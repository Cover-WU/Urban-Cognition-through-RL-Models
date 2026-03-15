"""
Model-Based SR-Dyna RL estimation with Successor Representation.

Optimized version with vectorized operations and caching.
"""

import numpy as np
import warnings
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any, Set
from collections import defaultdict
from itertools import chain
from scipy.optimize import minimize

import pandas as pd

from .utils import compute_day_sequence, compute_time_angle, compute_reward_array, compute_time_kernel, prepare_trajectory_data, pack_params, unpack_params
from .epi_memory import EntryRecord, EpisodicMemory
from .srdyna_model_speed import simulate_and_loglik_sr_dyna as simulate_and_loglik_sr_dyna_fast
from .srdyna_model_speed import _warmup_jit


@dataclass
class SRDynaConfig:
    """Configuration for SR Dyna model."""
    alpha_init: float = 0.1
    alpha_plan: float = 0.1
    beta_init: float = 1.0
    epsilon_init: float = 0.1
    phi_init: float = 0.1
    n_planning_steps: int = 10
    reward_type: str = 'log'
    reward_param_init: float = 1.0
    visit_threshold: int = 3
    memory_threshold: float = 0.01
    selection_size: int = None
    sigma_t_init: float = 1.0 / 12.0
    time_slots: int = 48
    maxiter: int = 1000
    ftol: float = 1e-6


@dataclass
class SRRecord(EntryRecord):
    """A single SR memory record for one (state, action) pair at a specific time."""
    sr: np.ndarray = None


class WorldModel:
    """Online learning world model for SR-Dyna planning."""

    def __init__(self, config: Optional[SRDynaConfig] = None):
        self.config = config if config is not None else SRDynaConfig()
        self.sigma_t = config.sigma_t_init if config else 1.0 / 12.0
        self.DAY_THRESHOLD_HOUR = 3
        self.action_time_transitions: Dict[int, List[Tuple[float, float]]] = defaultdict(list)
        self.state_visit_times: Dict[int, List[float]] = defaultdict(list)
        self.state_rewards: Dict[int, List[float]] = defaultdict(list)
        self.state_visit_counts: Dict[int, int] = defaultdict(int)

    def update(self, state: int, action: int, time_angle: float,
               reward: float, day_continues: bool, next_time_angle: float = None):
        """Update world model with observed transition."""
        self.state_visit_counts[state] += 1
        self.state_visit_times[state].append(time_angle)
        self.state_rewards[state].append(reward)

        if day_continues and next_time_angle is not None:
            time_delta = next_time_angle - time_angle
            if time_delta < 0:
                time_delta += 1.0
            self.action_time_transitions[action].append((time_angle, time_delta))

    def predict_next_time(self, state: int, action: int, time_angle: float = None) -> float:
        """Predict time delta to reach next state."""
        transitions = self.action_time_transitions.get(action, [])
        remaining_time = 1.0 - time_angle
        kernel_threshold = 3.0 * self.sigma_t

        valid_transitions = []
        for t, delta in transitions:
            time_diff = abs(time_angle - t)
            if time_diff <= kernel_threshold:
                valid_transitions.append((t, delta))

        if len(valid_transitions) < 3:
            return remaining_time * 0.5

        total_weight = 0.0
        weighted_sum = 0.0
        for t, delta in valid_transitions:
            weight = compute_time_kernel(time_angle, t, self.sigma_t)
            total_weight += weight
            weighted_sum += weight * delta

        time_delta = weighted_sum / total_weight
        return float(np.clip(time_delta, 0, remaining_time * 0.95))

    def predict_reward(self, state: int) -> float:
        """Predict average reward at given state."""
        if state not in self.state_rewards or not self.state_rewards[state]:
            return 0.0
        return float(np.mean(self.state_rewards[state]))


class SRMemory(EpisodicMemory):
    """Successor Representation memory, inheriting common methods from EpisodicMemory."""

    def __init__(self, phi: float, config: Optional[SRDynaConfig] = None):
        super().__init__(phi, config if config is not None else SRDynaConfig())
        self.exploration_rewards: List[float] = []

    @property
    def SRvisit(self) -> Dict[int, Dict[int, List[EntryRecord]]]:
        """Alias for memory field - SR visit records."""
        return self.memory

    def add_record(self, node_id: int, action_id: int, time_angle: float,
                   sr_array: np.ndarray, day_seq: int, record_date: int,
                   strength: float = 1.0):
        """Add an SR record to memory."""
        rec = SRRecord(
            sr=np.array(sr_array, dtype=np.float64),
            time_angle=float(time_angle),
            day_seq=int(day_seq),
            record_date=int(record_date),
            strength=float(strength),
        )

        self._add_record_base(rec, node_id, action_id)

    def add_exploration_reward(self, reward: float):
        """Record a reward obtained from exploration action."""
        self.exploration_rewards.append(float(reward))

    def get_exploration_reward_mean(self) -> float:
        """Get average reward from all exploration actions."""
        if not self.exploration_rewards:
            return 0.0
        return float(np.mean(self.exploration_rewards))

    def update_records_for_sa_pair(self, node_id: int, action_id: int, new_records: List[SRRecord]):
        """Update SR records for (node_id, action_id) pair."""
        assert len(self.memory[node_id][action_id]) == len(new_records)
        self.memory[node_id][action_id] = new_records

    def retrieve_sr(self, node_id: int, action_id: int, target_time: float,
                 sigma_t: Optional[float] = None) -> np.ndarray:
        """Retrieve SR vector for (node_id, action_id) at target time.

        Optimized version with vectorized kernel computation.
        """
        sigma = self.config.sigma_t_init if sigma_t is None else float(sigma_t)

        if action_id == -9:
            return visited_onehot(action_id, node_id, self.config.selection_size)

        recs = self.get_records_for_sa_pair(node_id, action_id)
        if not recs:
            return np.zeros(self.config.selection_size + 2, dtype=np.float64)

        sr_arrays = np.array([rec.sr for rec in recs], dtype=np.float64)
        strengths = np.array([rec.strength for rec in recs], dtype=np.float64)
        similarities = np.array([
            compute_time_kernel(target_time, rec.time_angle, sigma)
            for rec in recs
        ])

        weights = similarities * strengths
        weights_sum = np.sum(weights)

        if weights_sum <= 0:
            return np.zeros(self.config.selection_size + 2, dtype=np.float64)

        sr_estimate = np.sum(sr_arrays * weights[:, np.newaxis], axis=0) / weights_sum
        return sr_estimate

    def sr_is_empty(self) -> bool:
        """Check if SR memory is empty."""
        return not any(chain(*[self.memory[s][a] for s in self.memory for a in self.memory[s]]))

    def sample_random_sa(self) -> Optional[Tuple[int, int]]:
        """Randomly sample a (state, action) pair from memory for planning."""
        if self.sr_is_empty():
            return None

        valid_pairs = []
        for s in self.memory:
            for a in self.memory[s]:
                if a > 0:
                    valid_pairs.append((s, a))

        states_in_pairs = list(set([s for s, a in valid_pairs]))
        state_weights = []
        for s in states_in_pairs:
            count = len([a for s_pair, a in valid_pairs if s == s_pair])
            state_weights.append(count)

        total = sum(state_weights)
        probs = [w / total for w in state_weights]
        sampled_state = np.random.choice(states_in_pairs, p=probs)
        actions_for_state = [a for s, a in valid_pairs if s == sampled_state]
        sampled_action = np.random.choice(actions_for_state)

        return (sampled_state, sampled_action)


def visited_onehot(action: int, current_state: int, selection_size: int) -> np.ndarray:
    """Create one-hot encoding for visited state."""
    idx = action if action != -9 else current_state
    return np.eye(selection_size + 2)[idx]


def compute_Q(node_id: int, action_id: int, target_time: float,
             memory: SRMemory, model: WorldModel,
             sigma_t: Optional[float] = None) -> float:
    """Compute Q(s, a) using Successor Representation and world model."""
    sigma = memory.config.sigma_t_init if sigma_t is None else float(sigma_t)
    selection_size = memory.config.selection_size

    sr_vector = memory.retrieve_sr(node_id, action_id, target_time, sigma)

    reward_vector = np.zeros(selection_size + 2, dtype=np.float64)
    for s in range(1, selection_size + 1):
        reward_vector[s] = model.predict_reward(s)

    exploration_reward = memory.get_exploration_reward_mean()
    reward_vector[-1] = exploration_reward

    q_value = float(np.dot(sr_vector, reward_vector))
    return q_value


def prepare_sr_dyna_data(user_df: pd.DataFrame,
                        config: Optional[SRDynaConfig] = None) -> Dict[str, Any]:
    """Prepare trajectory data for SR Dyna modeling."""
    config = config if config is not None else SRDynaConfig()

    data = prepare_trajectory_data(
        user_df,
        config,
        config.reward_type,
        config.reward_param_init,
    )

    anchor_size = int(np.max(data['states'])) if len(data['states']) > 0 else 0
    data['selection_size'] = anchor_size

    return data



def simulate_and_loglik_sr_dyna(theta: np.ndarray,
                                sr_data: Dict[str, Any],
                                config: Optional[SRDynaConfig] = None) -> float:
    """Negative log-likelihood for SR-Dyna model.

    Optimized version with reward caching and reduced redundant computations.
    """
    config = config if config is not None else SRDynaConfig()

    params = unpack_params(theta)
    alpha = params['alpha']
    beta = params['beta']
    epsilon = params['epsilon']
    phi = params['phi']

    states = sr_data['states']
    actions = sr_data['actions']
    time_angles = sr_data['time_angles']
    day_seq = sr_data['day_seq']
    date_array = sr_data['date_array']
    same_day_next = sr_data['same_day_next']
    n_records = sr_data['n_records']
    reward_array = sr_data['reward_array']

    selection_size = sr_data['selection_size']
    config.selection_size = selection_size

    visit_counts: Dict[int, int] = defaultdict(int)
    known_states: Set[int] = {-1, 0}
    known_actions: Set[int] = {-9, -1, 0}

    memory = SRMemory(phi, config)
    model = WorldModel(config)

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

        # Use cached SR retrieval for Q computation
        q_values = np.array([
            compute_Q(s_perc, act, time_angle, memory, model, config.sigma_t_init)
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

        # ========================
        # REAL EXPERIENCE UPDATE
        # ========================
        day_continues = same_day_next[t] and t < n_records - 1
        # Get next time angle for model update
        if day_continues:
            time_angle_next = float(time_angles[t + 1])
        else:
            time_angle_next = None
        # Update world model with observed transition
        model.update(s_perc, a_perc, time_angle, r_t, day_continues, time_angle_next)
        
        # Record exploration reward if action was exploration (a_perc == -1)
        if a_perc == -1:
            memory.add_exploration_reward(r_t)

        # Update SR from real transition
        if day_continues and a_perc >= 0:
            a_next = int(actions[t + 1])
            a_next_perc = a_next if a_next in known_actions else -1

            next_sr = memory.retrieve_sr(a_perc, a_next_perc, time_angle_next, config.sigma_t_init)
            current_sr = memory.retrieve_sr(s_perc, a_perc, time_angle, config.sigma_t_init)

            onehot = visited_onehot(a_perc, s_perc, selection_size)
            target_sr = onehot + next_sr
            delta_sr = target_sr - current_sr
            new_sr = current_sr + alpha * delta_sr

            memory.add_record(s_perc, a_perc, time_angle, new_sr, current_day, current_date)

        # Dyna planning
        valid_planning_steps = 0
        while valid_planning_steps < config.n_planning_steps:
            sampled = memory.sample_random_sa()
            if sampled is None:
                break
            plan_s, plan_a = sampled
            if plan_a < 0:
                continue

            plan_records = memory.get_records_for_sa_pair(plan_s, plan_a)
            # if not plan_records:
            #     continue

            new_sr_records = []
            for rec in plan_records:
                plan_time = rec.time_angle
                time_delta = model.predict_next_time(plan_s, plan_a, plan_time)
                plan_next_time = (plan_time + time_delta) % 1.0

                plan_a_actions = set(memory.SRvisit.get(plan_a, {}).keys())
                plan_a_actions = list(plan_a_actions | {0, -9})

                best_q = -np.inf
                optimal_a = plan_a_actions[0] if plan_a_actions else 0
                for a_next in plan_a_actions:
                    q = compute_Q(plan_a, a_next, plan_next_time, memory, model, config.sigma_t_init)
                    if q > best_q:
                        best_q = q
                        optimal_a = a_next

                next_sr_optimal = memory.retrieve_sr(plan_a, optimal_a, plan_next_time, config.sigma_t_init)
                onehot = visited_onehot(plan_a, plan_s, selection_size)
                target_sr_plan = onehot + next_sr_optimal
                current_sr_plan = memory.retrieve_sr(plan_s, plan_a, plan_time, config.sigma_t_init)
                delta_sr_plan = target_sr_plan - current_sr_plan
                new_sr_plan = current_sr_plan + config.alpha_plan * delta_sr_plan

                new_rec = SRRecord(
                    sr=new_sr_plan.copy(),
                    time_angle=rec.time_angle,
                    day_seq=rec.day_seq,
                    record_date=rec.record_date,
                    strength=rec.strength,
                )
                new_sr_records.append(new_rec)
            valid_planning_steps += 1
            memory.update_records_for_sa_pair(plan_s, plan_a, new_sr_records)

    return -loglik


def fit_sr_dyna_model(user_df: pd.DataFrame,
                     config: Optional[SRDynaConfig] = None,
                     verbose: bool = True,
                     high_performance: bool = False) -> Dict[str, Any]:
    """Fit SR Dyna model for one user."""
    config = config if config is not None else SRDynaConfig()

    sr_data = prepare_sr_dyna_data(user_df, config)
    n_records = sr_data['n_records']

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

    theta_init = pack_params(
        alpha=config.alpha_init,
        beta=config.beta_init,
        epsilon=config.epsilon_init,
        phi=config.phi_init,
    )

    if high_performance:
        states = sr_data['states']
        actions = sr_data['actions']
        time_angles = sr_data['time_angles']
        day_seq = sr_data['day_seq']
        date_array = sr_data['date_array']
        same_day_next = sr_data['same_day_next']
        reward_array = sr_data['reward_array']
        n_records = sr_data['n_records']
        selection_size = int(sr_data['selection_size'])
        sr_data_fast = (states, actions, time_angles, day_seq, date_array, same_day_next, n_records, reward_array)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        if high_performance:
            _warmup_jit()
            result = minimize(
                simulate_and_loglik_sr_dyna_fast,
                theta_init,
                args=(
                    sr_data_fast,
                    selection_size,
                    int(config.time_slots),
                    float(config.sigma_t_init),
                    int(config.visit_threshold),
                    int(config.n_planning_steps),
                    float(config.alpha_plan),
                ),
                method='L-BFGS-B',
                options={'maxiter': config.maxiter, 'ftol': config.ftol, 'disp': False},
            )
        else:
            result = minimize(
                simulate_and_loglik_sr_dyna,
                theta_init,
                args=(sr_data, config),
                method='L-BFGS-B',
                options={'maxiter': config.maxiter, 'ftol': config.ftol, 'disp': False},
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
        'alpha': float(fitted['alpha']),
        'beta': float(fitted['beta']),
        'epsilon': float(fitted['epsilon']),
        'phi': float(fitted['phi']),
        'converged': bool(result.success),
        'n_iterations': int(result.nit) if hasattr(result, 'nit') else 0,
        'optimization_message': str(result.message) if hasattr(result, 'message') else '',
    }

    if verbose:
        print(f"SR Dyna model fitting {'converged' if result.success else 'did not converge'}.")
        print(f"  Log-likelihood: {log_likelihood:.2f}, AIC: {aic:.2f}, BIC: {bic:.2f}")
        print(f"  alpha (SR rate): {fitted['alpha']:.4f}")
        print(f"  beta (softmax temp): {fitted['beta']:.4f}")
        print(f"  epsilon (explore): {fitted['epsilon']:.4f}")
        print(f"  phi (forgetting): {fitted['phi']:.4f}")
    
    return summary


def fit_sr_dyna_for_all_users(users_dict: Dict[int, Any],
                             config: Optional[SRDynaConfig] = None,
                             sample_size: Optional[int] = None,
                             verbose: bool = True,
                             high_performance: bool = False) -> pd.DataFrame:
    """Fit SR-Dyna model for all users."""
    config = config if config is not None else SRDynaConfig()

    user_ids = list(users_dict.keys())
    if sample_size is not None:
        user_ids = user_ids[:sample_size]

    results = []
    for i, user_id in enumerate(user_ids):
        if verbose:
            print(f"[{i+1}/{len(user_ids)}] Fitting SR-Dyna for user {user_id}...", end='')

        user = users_dict[user_id]
        user_df = user.to_dataframe()

        try:
            t_start = time.time()
            result = fit_sr_dyna_model(user_df, config, verbose=False, high_performance=high_performance)
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
                'alpha': np.nan,
                'beta': np.nan,
                'epsilon': np.nan,
                'phi': np.nan,
                'converged': False,
                'n_iterations': None,
                'optimization_message': str(e),
            })

    return pd.DataFrame(results)
