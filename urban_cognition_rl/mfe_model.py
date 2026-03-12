"""
Model-Free with Episodic Memory (MFE) RL estimation.
"""

import numpy as np
import pandas as pd
import warnings
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any, Set
from collections import defaultdict
from scipy.optimize import minimize

from .epi_memory import EntryRecord, EpisodicMemory
from .utils import compute_day_sequence, compute_time_angle, compute_reward_array, compute_time_kernel, prepare_trajectory_data, pack_params, unpack_params


@dataclass
class MFEpiConfig:
    """Configuration for MF + episodic memory model."""
    alpha_init: float = 0.1
    beta_init: float = 1.0
    epsilon_init: float = 0.1
    phi_init: float = 0.1
    sigma_t_init: float = 1.0 / 12.0
    reward_type: str = 'log'
    reward_param_init: float = 1.0
    visit_threshold: int = 3
    memory_threshold: float = 0.01
    maxiter: int = 1000
    ftol: float = 1e-6



@dataclass
class QRecord(EntryRecord):
    """One memory trace for a (state, action, time) tuple."""
    q_value: float


class QMemory(EpisodicMemory):
    """Q-Value episodic memory for MFE model."""

    def __init__(self, phi: float, config: Optional[MFEpiConfig] = None):
        super().__init__(phi, config if config is not None else MFEpiConfig())

    @property
    def Q_table(self) -> Dict[int, Dict[int, List[EntryRecord]]]:
        """Alias for memory field - Q-value table."""
        return self.memory


    def add_record(self, node_id: int, action_id: int, time_angle: float, q_value: float,
                  day_seq: int, record_date: int):
        rec = QRecord(
            q_value=float(q_value),
            time_angle=float(time_angle),
            day_seq=int(day_seq),
            record_date=int(record_date),
            strength=1.0,
        )
        self._add_record_base(rec, node_id, action_id)


    def retrieve_q(self, target_node: int, target_action: int, target_time: float,
                 sigma_t: Optional[float] = None) -> float:
        """Return Q-hat(target_node, target_action, target_time)."""
        sigma = self.config.sigma_t_init if sigma_t is None else float(sigma_t)
        recs = self.get_records_for_sa_pair(target_node, target_action)
        if not recs:
            return 0.0

        q_values = np.array([rec.q_value for rec in recs], dtype=np.float64)
        strengths = np.array([rec.strength for rec in recs], dtype=np.float64)
        similarities = np.array([
            compute_time_kernel(target_time, rec.time_angle, sigma)
            for rec in recs
        ])

        weights = similarities * strengths
        q_estimate = np.average(q_values, weights=weights)
        q_estimate *= self.memory_decay[target_node][target_action]

        return q_estimate


def prepare_mfe_data(user_df: pd.DataFrame, config: Optional[MFEpiConfig] = None) -> Dict[str, Any]:
    """Prepare trajectory arrays for MF + episodic model."""
    config = config if config is not None else MFEpiConfig()

    data = prepare_trajectory_data(
        user_df,
        config,
        config.reward_type,
        config.reward_param_init,
    )

    return data




def simulate_and_loglik_mfe(theta: np.ndarray,
                            mfe_data: Dict[str, Any],
                            config: Optional[MFEpiConfig] = None) -> float:
    """Negative log-likelihood for MF + episodic memory with TD updates."""
    config = config if config is not None else MFEpiConfig()

    params = unpack_params(theta)
    alpha = params['alpha']
    beta = params['beta']
    epsilon = params['epsilon']
    phi = params['phi']

    states = mfe_data['states']
    actions = mfe_data['actions']
    time_angles = mfe_data['time_angles']
    day_seq = mfe_data['day_seq']
    date_array = mfe_data['date_array']
    same_day_next = mfe_data['same_day_next']
    n_records = mfe_data['n_records']
    reward_array = mfe_data['reward_array']

    visit_counts: Dict[int, int] = defaultdict(int)
    known_states: Set[int] = {-1, 0}
    known_actions: Set[int] = {-9, -1, 0}

    memory = QMemory(phi, config)
    loglik = 0.0

    for t in range(n_records):
        s = int(states[t])
        a = int(actions[t])
        r_t = float(reward_array[t])
        current_day = int(day_seq[t])
        current_date = int(date_array[t])
        time_angle = float(time_angles[t])

        memory.decay(current_day)

        if a > 0:
            visit_counts[a] += 1
            if visit_counts[a] >= config.visit_threshold:
                known_states.add(a)
                known_actions.add(a)

        s_perc = s if s in known_states else -1
        a_perc = a if a in known_actions else -1

        evaluated_actions = sorted([act for act in known_actions if act != -1])
        if len(evaluated_actions) == 0:
            action_prob = np.clip(epsilon, 1e-12, 1.0)
        else:
            q_values = np.array([
                memory.retrieve_q(s_perc, act, time_angle, config.sigma_t_init)
                for act in evaluated_actions
            ], dtype=float)

            beta_q = beta * q_values
            beta_q -= np.max(beta_q)
            softmax = np.exp(beta_q)
            softmax /= (np.sum(softmax) + 1e-12)

            probs = (1.0 - epsilon) * softmax

            if a_perc in evaluated_actions:
                idx_a = evaluated_actions.index(a_perc)
                action_prob = float(np.clip(probs[idx_a], 1e-12, 1.0))
            else:
                action_prob = float(np.clip(epsilon, 1e-12, 1.0))

        loglik += np.log(action_prob)

        if t < n_records - 1 and same_day_next[t]:
            a_next = int(actions[t + 1])
            a_next_perc = a_next if a_next in known_actions else -1

            next_time_angle = float(time_angles[t + 1])
            next_q = memory.retrieve_q(a_perc, a_next_perc, next_time_angle, config.sigma_t_init)
        else:
            next_q = 0.0

        if a_perc != -1:
            current_q = memory.retrieve_q(s_perc, a_perc, time_angle, config.sigma_t_init)
            delta = r_t + next_q - current_q
            new_q = current_q + alpha * delta

            memory.add_record(s_perc, a_perc, time_angle, new_q, current_day, current_date)

    return float(-loglik)


def fit_mfe_model(user_df: pd.DataFrame,
                            config: Optional[MFEpiConfig] = None,
                            verbose: bool = True) -> Dict[str, Any]:
    """Fit MF + episodic memory model for one user trajectory."""
    config = config if config is not None else MFEpiConfig()

    episodic_data = prepare_mfe_data(user_df, config)
    n_records = episodic_data['n_records']

    if n_records < 2:
        return {
            'n_records': int(n_records),
            'log_likelihood': np.nan,
            'AIC': np.nan,
            'BIC': np.nan,
            'alpha_td': np.nan,
            'beta': np.nan,
            'epsilon_explore': np.nan,
            'phi_forget': np.nan,
            'converged': False,
            'n_iterations': 0,
            'optimization_message': 'Insufficient records',
        }

    theta_init = pack_params(
        alpha=np.clip(config.alpha_init, 1e-6, 1.0),
        beta=max(config.beta_init, 1e-6),
        epsilon=np.clip(config.epsilon_init, 1e-4, 1 - 1e-4),
        phi=np.clip(config.phi_init, 1e-4, 1 - 1e-4),
    )

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        result = minimize(
            simulate_and_loglik_mfe,
            theta_init,
            args=(episodic_data, config),
            method='L-BFGS-B',
            options={'maxiter': int(config.maxiter), 'ftol': float(config.ftol), 'disp': False},
        )

    fitted = unpack_params(result.x)
    log_likelihood = -float(result.fun)

    k_params = 4
    aic = 2 * k_params - 2 * log_likelihood
    bic = k_params * np.log(max(n_records, 1)) - 2 * log_likelihood

    summary = {
        'n_records': int(n_records),
        'log_likelihood': float(log_likelihood),
        'AIC': float(aic),
        'BIC': float(bic),
        'alpha_td': float(fitted['alpha']),
        'beta': float(fitted['beta']),
        'epsilon_explore': float(fitted['epsilon']),
        'phi_forget': float(fitted['phi']),
        'converged': bool(result.success),
        'n_iterations': int(result.nit),
        'optimization_message': str(result.message),
    }

    if verbose:
        print(f"MFE model fitting {'converged' if result.success else 'did not converge'}.")
        print(f"  Log-likelihood: {summary['log_likelihood']:.2f}, AIC: {summary['AIC']:.2f}, BIC: {summary['BIC']:.2f}")
        print(f"  alpha (TD rate): {summary['alpha_td']:.4f}")
        print(f"  beta (softmax temp): {summary['beta']:.4f}")
        print(f"  epsilon (explore): {summary['epsilon_explore']:.4f}")
        print(f"  phi (forgetting): {summary['phi_forget']:.4f}")

    return summary


def fit_mfe_for_all_users(users_dict: Dict[int, Any],
                          config: Optional[MFEpiConfig] = None,
                          sample_size: Optional[int] = None,
                          verbose: bool = True) -> pd.DataFrame:
    """Fit MF+episodic model for all users."""
    config = config if config is not None else MFEpiConfig()

    user_ids = list(users_dict.keys())
    if sample_size is not None:
        user_ids = user_ids[:sample_size]

    results = []
    for i, user_id in enumerate(user_ids):
        if verbose:
            print(f"[{i+1}/{len(user_ids)}] Fitting MFE for user {user_id}...", end='')

        user = users_dict[user_id]
        user_df = user.to_dataframe()

        try:
            t_start = time.time()
            result = fit_mfe_model(user_df, config, verbose=False)
            elapsed = time.time() - t_start
            result['user_id'] = user_id
            result['fit_time_seconds'] = elapsed
            results.append(result)
            if verbose:
                print(f"\tLL: {result['log_likelihood']:.2f}, time: {elapsed:.2f}s")
        except Exception as e:
            if verbose:
                print(f"\tError: {e}")
            results.append({
                'user_id': user_id,
                'fit_time_seconds': np.nan,
                'n_records': 0,
                'log_likelihood': np.nan,
                'AIC': np.nan,
                'BIC': np.nan,
                'alpha_td': np.nan,
                'beta': np.nan,
                'epsilon_explore': np.nan,
                'phi_forget': np.nan,
                'converged': False,
                'n_iterations': None,
                'optimization_message': str(e),
            })

    return pd.DataFrame(results)
