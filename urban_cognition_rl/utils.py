"""
Utility functions shared across all models.
"""

import numpy as np
from datetime import datetime
from typing import Optional


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
