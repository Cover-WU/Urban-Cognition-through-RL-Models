"""
Utility functions shared across all models.
"""

import numpy as np
from numba import njit
from datetime import datetime
from typing import Optional, Dict, Any, Tuple
import pandas as pd


def compute_day_sequence(date_array: np.ndarray,
                         ref_date: Optional[datetime] = None) -> np.ndarray:
    """
    Compute day sequence as days since reference date.

    Parameters:
    - date_array: Array of dates in YYYYMMDD format
    - ref_date: Reference datetime

    Returns:
    - Array of day sequences (integers, monotonically increasing)
    """
    if ref_date is None:
        ref_date = datetime(2018, 12, 31)
    else:
        ref_date = datetime.strptime(str(ref_date), '%Y%m%d')

    day_seq = np.zeros(len(date_array), dtype=int)
    for i, d in enumerate(date_array):
        year = d // 10000
        month = (d // 100) % 100
        day = d % 100
        current_date = datetime(year, month, day)
        day_seq[i] = (current_date - ref_date).days

    return day_seq


def compute_time_angle(time_val: datetime,
                       angle_base_hour: int = 3) -> float:
    """
    Compute time angle scaled to [0, 1] where:
    - 0.0 = same day's 3 AM
    - 1.0 = next day's 3 AM (24 hours later)

    Example: 4 AM -> 1/24, Noon (12 PM) -> 9/24 = 0.375

    Parameters:
    - time_val: Datetime of stay end (departure time)
    - angle_base_hour: Base hour for day start (default: 3)

    Returns:
    - Time angle in [0, 1]
    """
    hour = time_val.hour
    minute = time_val.minute
    second = time_val.second

    seconds_in_day = 24 * 3600
    time_seconds = hour * 3600 + minute * 60 + second
    base_seconds = angle_base_hour * 3600

    delta = (time_seconds - base_seconds) % seconds_in_day
    time_angle = delta / seconds_in_day

    return time_angle


def compute_reward_array(stay_minutes: np.ndarray,
                         reward_type: str = 'log',
                         reward_param: Optional[float] = None) -> np.ndarray:
    """
    Compute reward array based on stay duration (time-based rewards).

    Parameters:
    - stay_minutes: Array of stay durations in minutes
    - reward_type: 'linear', 'power', or 'log'
    - reward_param: Parameter for power/log reward functions

    Returns:
    - Array of rewards
    """
    base = np.maximum(stay_minutes / 30.0, 0.0)

    if reward_type == 'linear' or reward_param is None:
        rewards = base
    elif reward_type == 'power':
        rewards = np.power(base, reward_param)
    elif reward_type == 'log':
        rewards = np.log1p(reward_param * base) / np.log1p(reward_param)
    else:
        rewards = base

    return np.nan_to_num(rewards, nan=0.0, posinf=0.0, neginf=0.0)


def compute_time_kernel(t1: float, t2: float, sigma: float) -> float:
    """
    Gaussian kernel for time similarity on normalized time angle [0, 1).

    Optimized version using vectorized numpy operations when possible.

    Parameters:
    - t1, t2: Time angles in [0, 1)
    - sigma: Kernel bandwidth

    Returns:
    - Similarity weight in [0, 1]
    """
    sigma = max(float(sigma), 1e-6)
    diff = abs(float(t1) - float(t2))
    # Handle wrap-around at day boundary
    return float(np.exp(-0.5 * (diff / sigma) ** 2))


def compute_time_discount_factor(time_angle: float) -> float:
    """
    Compute time discount factor based on time angle.

    The factor is: 1 / (1 - time_angle)
    This scales Q values so that values at different times of day are comparable.

    Example:
    - time_angle = 0 (3 AM): factor = 1.0
    - time_angle = 0.75 (9 PM): factor = 4.0
    - time_angle = 0.99 (2:58 AM next day): factor = 100.0

    Parameters:
    - time_angle: Time of day in [0, 1] scale

    Returns:
    - Time discount factor
    """
    return 1.0 / (1.0 - min(float(time_angle), 0.999))


def prepare_trajectory_data(user_df: pd.DataFrame,
                           config: Any = None,
                           reward_type: str = 'log',
                           reward_param_init: float = 1.0) -> Dict[str, Any]:
    """
    Prepare trajectory arrays for RL models. Shared by MFE and SR-Dyna.

    Parameters:
    - user_df: DataFrame with trajectory data
    - config: Model configuration (optional)
    - reward_type: Type of reward function
    - reward_param_init: Reward parameter

    Returns:
    - Dictionary with states, actions, time_angles, day_seq, etc.
    """
    df = user_df.sort_values(by=['t_start']).reset_index(drop=True)
    n_records = len(df)

    states = df['cluster_id'].astype(int).to_numpy()
    time_angles = df['t_end'].apply(compute_time_angle).to_numpy(dtype=float)
    date_array = df['date'].to_numpy()

    date_baseline = int(date_array.min())
    day_seq = compute_day_sequence(date_array, date_baseline)

    stay_minutes = (df['t_end'] - df['t_start']).dt.total_seconds() / 60.0
    stay_minutes = stay_minutes.to_numpy(dtype=float)
    stay_minutes = np.roll(stay_minutes, -1)
    stay_minutes[-1] = 0.0

    reward_array = compute_reward_array(
        stay_minutes,
        reward_type,
        reward_param_init,
    )

    actions = np.zeros(n_records, dtype=int)
    actions[-1] = -9
    for t in range(n_records - 1):
        if day_seq[t + 1] > day_seq[t]:
            actions[t] = -9
        else:
            actions[t] = int(states[t + 1])

    same_day_next = np.zeros(n_records, dtype=bool)
    for t in range(n_records - 1):
        same_day_next[t] = day_seq[t] == day_seq[t + 1]

    return {
        'states': states,
        'actions': actions,
        'day_seq': day_seq,
        'time_angles': time_angles,
        'date_array': date_array,
        'reward_array': reward_array,
        'same_day_next': same_day_next,
        'n_records': n_records,
    }


def pack_params(alpha: float, beta: float, epsilon: float, phi: float) -> np.ndarray:
    """
    Pack parameters for optimization (shared by MFE and SR-Dyna).

    Parameters:
    - alpha: Learning rate
    - beta: Softmax temperature
    - epsilon: Exploration rate
    - phi: Forgetting rate

    Returns:
    - Parameter vector in logit space
    """
    return np.array([
        np.log(alpha / (1.0 - alpha)),
        np.log(beta),
        np.log(epsilon / (1.0 - epsilon)),
        np.log(phi / (1.0 - phi)),
    ], dtype=float)


def unpack_params(theta: np.ndarray) -> Dict[str, float]:
    """
    Unpack parameters from optimization vector (shared by MFE and SR-Dyna).

    Parameters:
    - theta: Parameter vector in logit space

    Returns:
    - Dictionary with alpha, beta, epsilon, phi
    """
    idx = 0
    alpha = 1.0 / (1.0 + np.exp(-theta[idx])); idx += 1
    beta = np.exp(theta[idx]); idx += 1
    epsilon = 1.0 / (1.0 + np.exp(-theta[idx])); idx += 1
    phi = 1.0 / (1.0 + np.exp(-theta[idx])); idx += 1

    alpha = float(np.clip(alpha, 1e-6, 1.0))
    epsilon = float(np.clip(epsilon, 1e-6, 1.0 - 1e-6))

    return {
        'alpha': alpha,
        'beta': float(beta),
        'epsilon': epsilon,
        'phi': float(phi),
    }

@njit
def unpack_params_for_fast(theta: np.ndarray) -> Tuple:
    """
    Unpack parameters from optimization vector (shared by MFE and SR-Dyna).

    Parameters:
    - theta: Parameter vector in logit space

    Returns:
    - Dictionary with alpha, beta, epsilon, phi
    """
    idx = 0
    alpha = 1.0 / (1.0 + np.exp(-theta[idx])); idx += 1
    beta = np.exp(theta[idx]); idx += 1
    epsilon = 1.0 / (1.0 + np.exp(-theta[idx])); idx += 1
    phi = 1.0 / (1.0 + np.exp(-theta[idx])); idx += 1

    if alpha < 1e-6:
        alpha = 1e-6
    elif alpha > 1.0:
        alpha = 1.0

    if epsilon < 1e-6:
        epsilon = 1e-6
    elif epsilon > (1.0 - 1e-6):
        epsilon = 1.0 - 1e-6

    alpha = float(alpha)
    epsilon = float(epsilon)
    beta = float(beta)
    phi = float(phi)

    return alpha, beta, epsilon, phi
