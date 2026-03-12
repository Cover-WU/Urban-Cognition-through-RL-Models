"""
Model-Free (MF) RL estimation with TD learning.
"""

import numpy as np
import pandas as pd
import time

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any, Set
from collections import defaultdict
from scipy.special import logsumexp
from scipy.optimize import minimize

from .utils import compute_day_sequence, compute_time_angle, compute_reward_array, compute_time_discount_factor


@dataclass
class MFConfig:
    """Configuration for Model-Free RL estimation."""
    alpha_init: float = 0.1
    beta_init: float = 1.0
    epsilon_init: float = 0.1
    phi_init: float = 0.1
    reward_type: str = 'log'
    reward_param_init: float = 1.0
    visit_threshold: int = 3
    maxiter: int = 1000
    ftol: float = 1e-6
    ref_date: Tuple[int, int, int] = (2018, 12, 31)


def prepare_mf_data(user_df: pd.DataFrame,
                   config: MFConfig = None) -> Dict[str, Any]:
    """Prepare MF modeling data from user trajectory."""
    if config is None:
        config = MFConfig()

    df = user_df.sort_values('t_start').reset_index(drop=True)
    n_records = len(df)

    if 'date' not in df.columns:
        raise ValueError("Input DataFrame must contain 'date' column for MF model preparation.")

    date_array = df['date'].to_numpy()
    day_seq = compute_day_sequence(date_array, datetime(*config.ref_date))

    time_angles = np.zeros(n_records)
    for i, t_end in enumerate(df['t_end']):
        time_angles[i] = compute_time_angle(t_end)

    states = df['cluster_id'].astype(int).to_numpy()

    stay_minutes = (df['t_end'] - df['t_start']).dt.total_seconds() / 60.0
    stay_minutes = stay_minutes.to_numpy()
    stay_minutes = np.roll(stay_minutes, -1)
    stay_minutes[-1] = 0

    actions = np.zeros(n_records, dtype=int)
    actions[-1] = -9
    for t in range(n_records - 1):
        current_day = day_seq[t]
        next_day = day_seq[t + 1]

        if next_day > current_day:
            actions[t] = -9
        else:
            next_state = states[t + 1]
            if next_state == -1:
                actions[t] = -1
            elif next_state == 0:
                actions[t] = 0
            else:
                actions[t] = next_state

    same_day_next = np.zeros(n_records, dtype=bool)
    if n_records > 2:
        for t in range(n_records - 1):
            if day_seq[t] == day_seq[t + 1]:
                same_day_next[t] = True

    return {
        'states': states,
        'actions': actions,
        'day_seq': day_seq,
        'time_angles': time_angles,
        'stay_minutes': stay_minutes,
        'date_array': date_array,
        'n_records': n_records,
        'same_day_next': same_day_next,
        'df': df
    }


def unpack_params_mf(theta: np.ndarray,
                     feature_dim: int = 0,
                     has_reward_param: bool = True) -> Dict[str, float]:
    """Unpack MF model parameters from optimization vector."""
    idx = 0

    w = theta[:feature_dim] if feature_dim > 0 else np.array([])
    idx += feature_dim

    logit_alpha = theta[idx]; idx += 1
    alpha = 1.0 / (1.0 + np.exp(-logit_alpha))

    log_beta = theta[idx]; idx += 1
    beta = np.exp(log_beta)

    logit_epsilon = theta[idx]; idx += 1
    epsilon = 1.0 / (1.0 + np.exp(-logit_epsilon))

    logit_phi = theta[idx]; idx += 1
    phi = 1.0 / (1.0 + np.exp(-logit_phi))

    result = {
        'w': w,
        'alpha': alpha,
        'beta': beta,
        'epsilon': epsilon,
        'phi': phi
    }

    if has_reward_param:
        logit_reward = theta[idx]; idx += 1
        reward_param = np.exp(logit_reward)
        result['reward_param'] = reward_param

    return result


def pack_params_mf(alpha: float, beta: float, epsilon: float, phi: float,
                   reward_param: Optional[float] = None,
                   feature_dim: int = 0) -> np.ndarray:
    """Pack MF model parameters into optimization vector."""
    params = []

    if feature_dim > 0:
        params.extend([0.0] * feature_dim)

    params.append(np.log(alpha / (1.0 - alpha)))
    params.append(np.log(beta))
    params.append(np.log(epsilon / (1.0 - epsilon)))
    params.append(np.log(phi / (1.0 - phi)))

    if reward_param is not None:
        params.append(np.log(reward_param))

    return np.array(params, dtype=np.float64)


def simulate_and_loglik_mf(theta: np.ndarray,
                          mf_data: Dict[str, Any],
                          feature_dim: int = 0,
                          reward_type: str = 'log',
                          has_reward_param: bool = True,
                          visit_threshold: int = 3) -> float:
    """Compute negative log-likelihood for MF (TD) model."""
    params = unpack_params_mf(theta, feature_dim, has_reward_param)
    alpha = params['alpha']
    beta = params['beta']
    epsilon = params['epsilon']
    phi = params['phi']
    reward_param = params.get('reward_param', 1.0)

    states = mf_data['states']
    actions = mf_data['actions']
    day_seq = mf_data['day_seq']
    time_angles = mf_data['time_angles']
    n_records = mf_data['n_records']
    same_day_next = mf_data['same_day_next']

    reward_array = compute_reward_array(mf_data['stay_minutes'], reward_type, reward_param)

    Q_tables: Dict[int, Dict[int, float]] = defaultdict(lambda: defaultdict(float))
    visit_counts: Dict[int, int] = defaultdict(int)

    known_states: Set[int] = {-1, 0}
    known_actions: Set[int] = {-9, -1, 0}

    loglik = 0.0
    prev_day = None

    for t in range(n_records):
        s = int(states[t])
        a = int(actions[t])
        r_t = float(reward_array[t])
        current_day = int(day_seq[t])
        time_angle = float(time_angles[t])

        if prev_day is not None and current_day > prev_day:
            discount_factor = (1.0 - phi)
            for state_dict in Q_tables.values():
                for action_key in state_dict:
                    multi_day_discount = discount_factor ** (current_day - prev_day)
                    state_dict[action_key] *= multi_day_discount
        prev_day = current_day

        if a > 0:
            visit_counts[a] += 1
            if visit_counts[a] >= visit_threshold:
                known_states.add(a)
                known_actions.add(a)

        s_perc = s if s in known_states else -1
        a_perc = a if a in known_actions else -1

        time_scale = 1

        evaluated_actions = sorted([act for act in known_actions if act != -1])

        q_values = []
        for act in evaluated_actions:
            q_td = Q_tables[s_perc].get(act, 0.0)
            q_values.append(q_td * time_scale)

        q_values = np.asarray(q_values, dtype=np.float64)

        logits = beta * q_values
        probs_exploit = np.exp(logits - logsumexp(logits))

        if a_perc in evaluated_actions:
            idx_a = evaluated_actions.index(a_perc)
            if idx_a < len(probs_exploit):
                action_prob = (1.0 - epsilon) * probs_exploit[idx_a]
        elif a_perc == -1:
            action_prob = epsilon
        else:
            action_prob = 0

        loglik += np.log(action_prob + 1e-12)

        if t < n_records - 1 and same_day_next[t]:
            a_next = int(actions[t + 1])
            a_next_perc = a_next if a_next in known_actions else -1

            next_time_angle = float(time_angles[t + 1])
            next_time_scale = 1
            next_Q_td = Q_tables[a_perc].get(a_next_perc, 0.0) * next_time_scale
        else:
            next_Q_td = 0.0

        current_Q = Q_tables[s_perc].get(a_perc, 0.0) * time_scale
        delta = r_t + next_Q_td - current_Q

        Q_tables[s_perc][a_perc] = Q_tables[s_perc].get(a_perc, 0.0) + alpha * delta

    return -loglik


def fit_mf_model(user_df: pd.DataFrame,
                config: MFConfig = None,
                verbose: bool = True) -> Dict[str, Any]:
    """Fit MF model to a single user's trajectory data."""
    if config is None:
        config = MFConfig()

    mf_data = prepare_mf_data(user_df, config)
    n_records = mf_data['n_records']

    if n_records < 2:
        return {
            'n_records': int(n_records),
            'log_likelihood': np.nan,
            'AIC': np.nan,
            'BIC': np.nan,
            'alpha': np.nan,
            'beta': np.nan,
            'epsilon': np.nan,
            'phi': np.nan,
            'converged': False,
            'n_iterations': 0,
            'optimization_message': 'Insufficient records',
        }

    has_reward_param = config.reward_type in ('power', 'log')
    extra_params = 4 + (1 if has_reward_param else 0)
    param_dim = extra_params

    initial_theta = pack_params_mf(
        alpha=config.alpha_init,
        beta=config.beta_init,
        epsilon=config.epsilon_init,
        phi=config.phi_init,
        reward_param=config.reward_param_init if has_reward_param else None,
    )

    def objective(theta):
        return simulate_and_loglik_mf(
            theta,
            mf_data,
            reward_type=config.reward_type,
            has_reward_param=has_reward_param,
            visit_threshold=config.visit_threshold
        )

    result = minimize(
        objective,
        initial_theta,
        method='L-BFGS-B',
        options={'maxiter': config.maxiter, 'ftol': config.ftol}
    )

    params = unpack_params_mf(result.x, has_reward_param=has_reward_param)

    n_params = param_dim
    log_likelihood = -result.fun
    aic = 2 * n_params - 2 * log_likelihood
    bic = n_params * np.log(n_records) - 2 * log_likelihood

    return {
        'n_records': int(n_records),
        'log_likelihood': log_likelihood,
        'AIC': aic,
        'BIC': bic,
        'alpha': params['alpha'],
        'beta': params['beta'],
        'epsilon': params['epsilon'],
        'phi': params['phi'],
        'reward_param': params.get('reward_param', np.nan),
        'converged': result.success,
        'n_iterations': result.nit,
        'optimization_message': result.message,
    }


def fit_mf_for_all_users(users_dict: Dict[int, Any],
                        config: MFConfig = None,
                        sample_size: Optional[int] = None,
                        verbose: bool = True) -> pd.DataFrame:
    """Fit MF model for all users."""
    if config is None:
        config = MFConfig()

    user_ids = list(users_dict.keys())
    if sample_size is not None:
        user_ids = user_ids[:sample_size]

    results = []
    for i, user_id in enumerate(user_ids):
        if verbose:
            print(f"[{i+1}/{len(user_ids)}] Fitting MF for user {user_id}...", end='')

        user = users_dict[user_id]
        user_df = user.to_dataframe()

        try:
            t_start = time.time()
            result = fit_mf_model(user_df, config, verbose=False)
            elapsed = time.time() - t_start
            result['user_id'] = user_id
            result['fit_time_seconds'] = elapsed
            results.append(result)
            if verbose:
                print(f" done (LL: {result['log_likelihood']:.2f}, time: {elapsed:.2f}s)")
        except Exception as e:
            if verbose:
                print(f" error: {e}")
            results.append({
                'user_id': user_id,
                'fit_time_seconds': np.nan,
                'n_records': 0,
                'log_likelihood': np.nan,
                'AIC': np.nan,
                'BIC': np.nan,
                'alpha': np.nan,
                'beta': np.nan,
                'epsilon': np.nan,
                'phi': np.nan,
                'converged': False,
                'n_iterations': None,
                'optimization_message': str(e),
            })

    return pd.DataFrame(results)
