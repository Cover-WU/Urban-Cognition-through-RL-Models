"""
Model-Free with Episodic Memory (MFE) - Numba Accelerated Version.

This module contains the core computation classes and functions:
- QMemory (jitclass)
- simulate_and_loglik_mfe (njit)

Data preparation and model fitting functions are in mfe_model.py.
"""

import numpy as np
from numba import njit, float32, float64, int32, int64, boolean, types
from numba.experimental import jitclass
from typing import Any, Tuple
from .utils import unpack_params_for_fast
from math import floor


# QMemory spec for jitclass
qmemory_spec = [
    ('phi', float64),
    ('selection_size', int32),
    ('memory_threshold', float64),
    ('time_slots', int32),
    ('sigma_t', float64),
    ('q_table', float32[:,:,:]),
    ('mem_strength', float64[:,:,:]),
    ('mem_decay', float64[:,:]),
    ('time_similarity_kernel_matrix', float32[:,:]),
    ('last_day', int32),
]

@jitclass(qmemory_spec)
class QMemory:
    """Q-Value episodic memory for MFE model.
    
    Numba jitclass for high-performance compilation.
    """

    def __init__(self, selection_size: int, time_slots: int,
                 phi: float, memory_threshold: float, sigma_t: float):
        self.phi = phi
        self.selection_size = selection_size
        self.memory_threshold = memory_threshold
        self.time_slots = time_slots
        self.sigma_t = sigma_t
        
        # Total size: indices -1, 0, 1, 2, ..., selection_size
        # Mapped to array indices: 0, 1, 2, 3, ..., selection_size+1
        total_size = selection_size + 2
        
        self.q_table = np.zeros((total_size, total_size, time_slots), dtype=np.float32)
        self.mem_strength = np.zeros((total_size, total_size, time_slots), dtype=np.float64)
        self.mem_decay = np.zeros((total_size, total_size), dtype=np.float64)
        
        # Precompute time similarity kernel matrix
        self.time_similarity_kernel_matrix = np.empty((time_slots, time_slots), dtype=np.float32)
        for i in range(time_slots):
            for j in range(time_slots):
                dist = abs(i - j)
                kernel = np.exp(-0.5 * (dist / sigma_t * time_slots) ** 2)
                self.time_similarity_kernel_matrix[i, j] = kernel

        # Track last day for decay
        self.last_day = -1


    def decay(self, current_day: int):
        """Apply memory decay when day changes."""
        if self.last_day < 0:
            self.last_day = current_day
            return

        day_diff = current_day - self.last_day
        if day_diff > 0:
            factor = (1.0 - self.phi) ** day_diff
            self.mem_strength *= factor
            self.mem_decay *= factor

        self.last_day = current_day


    @staticmethod
    def time_slot_mapping(time_angle: float, time_slots: int) -> int:
        """Map continuous time angle to discrete slot."""
        if time_angle >= 1.0:
            time_angle = 1.0 - 1e-12
        return floor(time_angle * time_slots)


    def encode(self, node_id: int, action_id: int, time_angle: float, q_value: float):
        """Encode new memory record with incremental update."""
        time_slot = self.time_slot_mapping(time_angle, self.time_slots)
        
        current_Q = self.q_table[node_id, action_id, time_slot]
        self.mem_strength[node_id, action_id, time_slot] += 1.0
        difference = (q_value - current_Q) / self.mem_strength[node_id, action_id, time_slot]
        self.q_table[node_id, action_id, time_slot] = current_Q + difference
        self.mem_decay[node_id, action_id] = 1.0


    def retrieve(self, target_node: int, target_action: int, target_time: float) -> float:
        """Return Q-hat(target_node, target_action, target_time) with weighted average."""
        target_action = target_action if target_action != -9 else -1
        time_slot = self.time_slot_mapping(target_time, self.time_slots)
        similarity_kernel = self.time_similarity_kernel_matrix[time_slot, :]
        extracted_q_values = self.q_table[target_node, target_action, :]
        
        q_estimate = np.dot(extracted_q_values, similarity_kernel) / np.sum(similarity_kernel)
        q_estimate *= self.mem_decay[target_node, target_action]
        return q_estimate

    def retrieve_q_array(self, target_node: int, target_time: float) -> np.ndarray:
        """Retrieve Q values for all actions as array."""
        time_slot = self.time_slot_mapping(target_time, self.time_slots)
        similarity_kernel = self.time_similarity_kernel_matrix[time_slot, :]

        extracted_q_array = self.q_table[target_node, :, :]
        q_estimate_arr = np.dot(extracted_q_array, similarity_kernel).astype(np.float64) / np.sum(similarity_kernel)
        q_estimate_arr *= self.mem_decay[target_node, :]
        return q_estimate_arr


@njit(cache=True, fastmath=True)
def simulate_and_loglik_mfe(theta: np.ndarray,
                            mfe_data: Tuple,
                            selection_size: int,
                            time_slots: int,
                            sigma_t: float,
                            visit_threshold: int,
                            memory_threshold: float = 0.01) -> float:
    """Negative log-likelihood for MF + episodic memory with TD updates.
    
    Numba accelerated version using @njit.
    """
    
    alpha, beta, epsilon, phi = unpack_params_for_fast(theta)

    states, actions, time_angles, day_seq, date_array, same_day_next, n_records, reward_array = mfe_data

    # Index mapping: -1 -> 0, 0 -> 1, 1 -> 2, ...
    total_size = selection_size + 2
    
    # visit_counts and is_place_known use original indices as keys
    # but stored in array with offset +1
    visit_counts = np.zeros(total_size, dtype=np.int32)
    is_place_known = np.zeros(total_size, dtype=np.bool_)
    
    # Set is_place_known[-1] = True and is_place_known[0] = True
    is_place_known[0] = True   
    is_place_known[-1] = True   

    # Initialize QMemory (jitclass)
    memory = QMemory(selection_size, time_slots, phi, memory_threshold, sigma_t)

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
            # a is original index, use directly since a>0 means 1,2,3...
            visit_counts[a] += 1
            if visit_counts[a] >= visit_threshold:
                # Map original index to array index
                is_place_known[a] = True

        # Perception: s_perc = s if is_place_known[s] else -1
        if is_place_known[s]:
            s_perc = s
        else:
            s_perc = -1

        # Perception: a_perc = a if is_place_known[a] or a == -9 else -1
        if a == -9 or is_place_known[a]:
            a_perc = a
        else:
            a_perc = -1

        # Use cached retrieval
        q_values_array = memory.retrieve_q_array(s_perc, time_angle)
        # -9 is in the last index here, -1 changed into -9
        evaluated_actions = np.flatnonzero(is_place_known)
        q_values = q_values_array[evaluated_actions]

        # Compute softmax probabilities
        beta_q = beta * q_values
        beta_q -= np.max(beta_q)
        softmax = np.exp(beta_q)
        softmax /= (np.sum(softmax) + 1e-12)

        probs = (1.0 - epsilon) * softmax


        # Get action probability
        if a_perc == -1:
            action_prob = epsilon
        elif a_perc == -9:
            action_prob = probs[-1]
        else:
            # Find index of a_perc in eval_actions
            action_prob = epsilon
            for i, action in enumerate(evaluated_actions):
                if action == a_perc:
                    action_prob = probs[i]
                    break

        loglik += np.log(action_prob)

        # TD update
        if t < n_records - 1 and same_day_next[t]:
            a_next = int(actions[t + 1])
            a_next_perc = a_next if (a_next == -9 or is_place_known[a_next]) else -1

            next_time_angle = float(time_angles[t + 1])
            if a_next_perc != -1:
                next_q = memory.retrieve(a_perc, a_next_perc, next_time_angle)
            else:
                all_qs = memory.retrieve_q_array(a_perc, next_time_angle)
                all_ws = memory.mem_decay[a_perc, :]
                ws_sum = np.sum(all_ws)
                if ws_sum <= 1e-12:
                    next_q = 0.0
                else:
                    next_q = np.dot(all_qs, all_ws) / ws_sum
        else:
            next_q = 0.0

        if a_perc != -1:
            current_q = memory.retrieve(s_perc, a_perc, time_angle)
            delta = r_t + next_q - current_q
            new_q = current_q + alpha * delta

            memory.encode(s_perc, a_perc, time_angle, new_q)

    return float(-loglik)


# JIT compile on import (optional, can be called lazily)
def _warmup_jit():
    """Warm up JIT compilation with dummy data."""
    dummy_theta = np.array([0.1, 1.0, 0.1, 0.01], dtype=np.float64)
    dummy_states = np.array([0, 1, 2], dtype=np.int32)
    dummy_actions = np.array([1, 2, -9], dtype=np.int32)
    dummy_time_angles = np.array([0.0, 0.5, 1.0], dtype=np.float64)
    dummy_day_seq = np.array([0, 0, 1], dtype=np.int32)
    dummy_date_array = np.array([0, 0, 1], dtype=np.int32)
    dummy_same_day_next = np.array([True, False, False], dtype=np.bool_)
    dummy_n_records = int(3)
    dummy_reward_array = np.array([1.0, 0.5, 1.0], dtype=np.float64)
    dummy_data = (
        dummy_states,
        dummy_actions,
        dummy_time_angles,
        dummy_day_seq,
        dummy_date_array,
        dummy_same_day_next,
        dummy_n_records,
        dummy_reward_array,
    )
    try:
        simulate_and_loglik_mfe(dummy_theta, dummy_data, 5, 24, 0.1, 2, 0.01)
    except Exception:
        pass  # Ignore errors during warmup

# Uncomment to enable warmup on import
# _warmup_jit()
