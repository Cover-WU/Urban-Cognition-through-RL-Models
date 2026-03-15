"""
Model-Based SR-Dyna - Numba Accelerated Version.

This module mirrors the MFE acceleration style:
- SRMemory (jitclass)
- WorldModel (jitclass)
- simulate_and_loglik_sr_dyna (njit)

Data preparation and fitting entry remain in srdyna_model.py.
"""

import numpy as np
from math import floor
from numba import njit, float32, float64, int32, int64, boolean
from numba.experimental import jitclass
from typing import Tuple

from .utils import unpack_params_for_fast


@njit(cache=True)
def _time_kernel(t1: float, t2: float, sigma: float) -> float:
    if sigma < 1e-6:
        sigma = 1e-6
    diff = abs(t1 - t2)
    return np.exp(-0.5 * (diff / sigma) ** 2)


@njit(cache=True)
def _time_slot_mapping(time_angle: float, time_slots: int) -> int:
    if time_angle >= 1.0:
        time_angle = 1.0 - 1e-12
    return floor(time_angle * time_slots)


@njit(cache=True)
def _state_to_index(state_id: int) -> int:
    # state=-1 occupies the last slot by Python negative index convention.
    return -1 if state_id == -1 else state_id


@njit(cache=True)
def _action_to_index(action_id: int) -> int:
    # action=-9 occupies the last slot by Python negative index convention.
    return -1 if action_id == -9 else action_id


world_model_spec = [
    ('selection_size', int32),
    ('sigma_t', float64),
    ('max_transitions', int32),
    ('transition_times', float64[:, :]),
    ('transition_deltas', float64[:, :]),
    ('transition_counts', int32[:]),
    ('state_reward_sums', float64[:]),
    ('state_reward_counts', int32[:]),
]


@jitclass(world_model_spec)
class WorldModel:
    """Fixed-array world model for SR-Dyna planning."""

    def __init__(self, selection_size: int, sigma_t: float, max_transitions: int):
        self.selection_size = selection_size
        self.sigma_t = sigma_t
        self.max_transitions = max_transitions

        total_size = selection_size + 2
        self.transition_times = np.zeros((total_size, max_transitions), dtype=np.float64)
        self.transition_deltas = np.zeros((total_size, max_transitions), dtype=np.float64)
        self.transition_counts = np.zeros(total_size, dtype=np.int32)

        self.state_reward_sums = np.zeros(total_size, dtype=np.float64)
        self.state_reward_counts = np.zeros(total_size, dtype=np.int32)

    def update(self, state: int, action: int, time_angle: float,
               reward: float, day_continues: boolean, next_time_angle: float):
        state_idx = _state_to_index(state)
        self.state_reward_sums[state_idx] += reward
        self.state_reward_counts[state_idx] += 1

        # Exploration action (-1) is not represented in action-value memory.
        if action == -1:
            return

        if day_continues:
            action_idx = _action_to_index(action)
            cnt = self.transition_counts[action_idx]
            if cnt < self.max_transitions:
                delta = next_time_angle - time_angle
                if delta < 0.0:
                    delta += 1.0
                self.transition_times[action_idx, cnt] = time_angle
                self.transition_deltas[action_idx, cnt] = delta
                self.transition_counts[action_idx] = cnt + 1

    def predict_next_time(self, state: int, action: int, time_angle: float) -> float:
        _state = state  # keep signature aligned with python version
        _ = _state

        remaining_time = 1.0 - time_angle
        if remaining_time <= 0.0:
            return 0.0

        action_idx = _action_to_index(action)
        cnt = self.transition_counts[action_idx]
        if cnt <= 0:
            return remaining_time * 0.5

        kernel_threshold = 3.0 * self.sigma_t
        valid_n = 0
        total_weight = 0.0
        weighted_sum = 0.0
        for i in range(cnt):
            t = self.transition_times[action_idx, i]
            delta = self.transition_deltas[action_idx, i]
            if abs(time_angle - t) <= kernel_threshold:
                w = _time_kernel(time_angle, t, self.sigma_t)
                valid_n += 1
                total_weight += w
                weighted_sum += w * delta

        if valid_n < 3 or total_weight <= 1e-12:
            return remaining_time * 0.5

        pred = weighted_sum / total_weight
        upper = remaining_time * 0.95
        if pred < 0.0:
            pred = 0.0
        if pred > upper:
            pred = upper
        return pred

    def predict_reward(self, state: int) -> float:
        state_idx = _state_to_index(state)
        cnt = self.state_reward_counts[state_idx]
        if cnt <= 0:
            return 0.0
        return self.state_reward_sums[state_idx] / cnt


sr_memory_spec = [
    ('phi', float64),
    ('selection_size', int32),
    ('time_slots', int32),
    ('sigma_t', float64),
    ('sr_table', float32[:, :, :, :]),
    ('mem_strength', float64[:, :, :]),
    ('pair_active', boolean[:, :]),
    ('time_similarity_kernel_matrix', float32[:, :]),
    ('exploration_reward_sum', float64),
    ('exploration_reward_count', int64),
    ('last_day', int32),
]


@jitclass(sr_memory_spec)
class SRMemory:
    """Fixed-array SR memory with slot prototypes."""

    def __init__(self, selection_size: int, time_slots: int, phi: float, sigma_t: float):
        self.phi = phi
        self.selection_size = selection_size
        self.time_slots = time_slots
        self.sigma_t = sigma_t

        total_size = selection_size + 2
        self.sr_table = np.zeros((total_size, total_size, time_slots, total_size), dtype=np.float32)
        self.mem_strength = np.zeros((total_size, total_size, time_slots), dtype=np.float64)
        self.pair_active = np.zeros((total_size, total_size), dtype=np.bool_)

        self.time_similarity_kernel_matrix = np.empty((time_slots, time_slots), dtype=np.float32)
        for i in range(time_slots):
            for j in range(time_slots):
                dist = abs(i - j)
                kernel = np.exp(-0.5 * (dist / sigma_t * time_slots) ** 2)
                self.time_similarity_kernel_matrix[i, j] = kernel

        self.exploration_reward_sum = 0.0
        self.exploration_reward_count = 0
        self.last_day = -1

    def decay(self, current_day: int):
        if self.last_day < 0:
            self.last_day = current_day
            return

        day_diff = current_day - self.last_day
        if day_diff > 0:
            factor = (1.0 - self.phi) ** day_diff
            self.mem_strength *= factor
        self.last_day = current_day

    def add_exploration_reward(self, reward: float):
        self.exploration_reward_sum += reward
        self.exploration_reward_count += 1

    def get_exploration_reward_mean(self) -> float:
        if self.exploration_reward_count <= 0:
            return 0.0
        return self.exploration_reward_sum / self.exploration_reward_count

    def encode_sr(self, node_id: int, action_id: int, time_angle: float, sr_vec: np.ndarray):
        if action_id == -1:
            return

        node_idx = _state_to_index(node_id)
        action_idx = _action_to_index(action_id)
        slot = _time_slot_mapping(time_angle, self.time_slots)

        old_strength = self.mem_strength[node_idx, action_idx, slot]
        new_strength = old_strength + 1.0
        self.mem_strength[node_idx, action_idx, slot] = new_strength
        self.pair_active[node_idx, action_idx] = True

        for j in range(self.selection_size + 2):
            prev = self.sr_table[node_idx, action_idx, slot, j]
            self.sr_table[node_idx, action_idx, slot, j] = prev + (sr_vec[j] - prev) / new_strength

    def overwrite_slot_sr(self, node_id: int, action_id: int, slot: int, sr_vec: np.ndarray):
        if action_id == -1:
            return
        node_idx = _state_to_index(node_id)
        action_idx = _action_to_index(action_id)
        for j in range(self.selection_size + 2):
            self.sr_table[node_idx, action_idx, slot, j] = sr_vec[j]

    def retrieve_sr(self, node_id: int, action_id: int, target_time: float) -> np.ndarray:
        total_size = self.selection_size + 2
        out = np.zeros(total_size, dtype=np.float64)

        if action_id == -1:
            return out

        node_idx = _state_to_index(node_id)
        if action_id == -9:
            out[node_idx] = 1.0
            return out

        action_idx = action_id
        slot = _time_slot_mapping(target_time, self.time_slots)
        similarity_kernel = self.time_similarity_kernel_matrix[slot, :]

        weight_sum = 0.0
        for ts in range(self.time_slots):
            w = similarity_kernel[ts] * self.mem_strength[node_idx, action_idx, ts]
            if w <= 0.0:
                continue
            weight_sum += w
            for j in range(total_size):
                out[j] += self.sr_table[node_idx, action_idx, ts, j] * w

        if weight_sum <= 1e-12:
            out[:] = 0.0
            return out

        out /= weight_sum
        return out

    def sample_random_sa(self) -> Tuple[int, int]:
        total_size = self.selection_size + 2

        state_weights = np.zeros(total_size, dtype=np.int32)
        total_pairs = 0
        for s in range(total_size):
            cnt = 0
            for a in range(1, self.selection_size + 1):
                if self.pair_active[s, a]:
                    cnt += 1
            state_weights[s] = cnt
            total_pairs += cnt

        if total_pairs <= 0:
            return -99999, -99999

        r = np.random.random() * total_pairs
        acc = 0.0
        sampled_s_idx = -1
        for s in range(total_size):
            acc += state_weights[s]
            if r <= acc:
                sampled_s_idx = s
                break
        if sampled_s_idx < 0:
            return -99999, -99999

        n_actions = 0
        for a in range(1, self.selection_size + 1):
            if self.pair_active[sampled_s_idx, a]:
                n_actions += 1
        if n_actions <= 0:
            return -99999, -99999

        r2 = np.random.random() * n_actions
        acc2 = 0.0
        sampled_a = -1
        for a in range(1, self.selection_size + 1):
            if self.pair_active[sampled_s_idx, a]:
                acc2 += 1.0
                if r2 <= acc2:
                    sampled_a = a
                    break

        sampled_s = -1 if sampled_s_idx == total_size - 1 else sampled_s_idx
        return sampled_s, sampled_a


@njit(cache=True)
def compute_Q(node_id: int, action_id: int, target_time: float,
              memory: SRMemory, model: WorldModel) -> float:
    selection_size = memory.selection_size
    sr_vector = memory.retrieve_sr(node_id, action_id, target_time)

    q_value = 0.0
    for s in range(1, selection_size + 1):
        q_value += sr_vector[s] * model.predict_reward(s)

    q_value += sr_vector[-1] * memory.get_exploration_reward_mean()
    return q_value


@njit(cache=True, fastmath=True)
def simulate_and_loglik_sr_dyna(theta: np.ndarray,
                                sr_data: Tuple,
                                selection_size: int,
                                time_slots: int,
                                sigma_t: float,
                                visit_threshold: int,
                                n_planning_steps: int,
                                alpha_plan: float) -> float:
    """Numba accelerated SR-Dyna negative log-likelihood."""

    alpha, beta, epsilon, phi = unpack_params_for_fast(theta)

    states, actions, time_angles, day_seq, date_array, same_day_next, n_records, reward_array = sr_data

    memory = SRMemory(selection_size, time_slots, phi, sigma_t)
    model = WorldModel(selection_size, sigma_t, n_records)

    total_size = selection_size + 2
    visit_counts = np.zeros(total_size, dtype=np.int32)
    known_pos_states = np.zeros(total_size, dtype=np.bool_)
    known_pos_actions = np.zeros(total_size, dtype=np.bool_)
    known_pos_states[0] = True
    known_pos_actions[0] = True

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
            if visit_counts[a] >= visit_threshold:
                known_pos_states[a] = True
                known_pos_actions[a] = True

        if s == -1 or s == 0:
            s_perc = s
        elif s > 0 and known_pos_states[s]:
            s_perc = s
        else:
            s_perc = -1

        if a == -9:
            a_perc = -9
        elif a == 0:
            a_perc = 0
        elif a > 0 and known_pos_actions[a]:
            a_perc = a
        else:
            a_perc = -1

        evaluated_actions = np.empty(total_size, dtype=np.int32)
        n_eval = 0
        evaluated_actions[n_eval] = -9
        n_eval += 1
        evaluated_actions[n_eval] = 0
        n_eval += 1
        for ap in range(1, selection_size + 1):
            if known_pos_actions[ap]:
                evaluated_actions[n_eval] = ap
                n_eval += 1

        q_values = np.empty(n_eval, dtype=np.float64)
        for i in range(n_eval):
            q_values[i] = compute_Q(s_perc, evaluated_actions[i], time_angle, memory, model)

        beta_q = beta * q_values
        beta_q -= np.max(beta_q)
        softmax = np.exp(beta_q)
        softmax /= (np.sum(softmax) + 1e-12)
        probs = (1.0 - epsilon) * softmax

        action_prob = epsilon
        if a_perc != -1:
            found = False
            for i in range(n_eval):
                if evaluated_actions[i] == a_perc:
                    action_prob = probs[i]
                    found = True
                    break
            if not found:
                action_prob = epsilon

        if action_prob < 1e-12:
            action_prob = 1e-12
        loglik += np.log(action_prob)

        day_continues = False
        next_time_angle = 0.0
        if t < n_records - 1 and same_day_next[t]:
            day_continues = True
            next_time_angle = float(time_angles[t + 1])

        model.update(s_perc, a_perc, time_angle, r_t, day_continues, next_time_angle)

        if a_perc == -1:
            memory.add_exploration_reward(r_t)

        if day_continues and a_perc >= 0:
            a_next = int(actions[t + 1])
            if a_next == -9:
                a_next_perc = -9
            elif a_next == 0:
                a_next_perc = 0
            elif a_next > 0 and known_pos_actions[a_next]:
                a_next_perc = a_next
            else:
                a_next_perc = -1

            next_sr = memory.retrieve_sr(a_perc, a_next_perc, next_time_angle)
            current_sr = memory.retrieve_sr(s_perc, a_perc, time_angle)

            onehot = np.zeros(total_size, dtype=np.float64)
            onehot[a_perc] = 1.0
            target_sr = onehot + next_sr
            new_sr = current_sr + alpha * (target_sr - current_sr)

            memory.encode_sr(s_perc, a_perc, time_angle, new_sr)

        valid_planning_steps = 0
        while valid_planning_steps < n_planning_steps:
            plan_s, plan_a = memory.sample_random_sa()
            if plan_s == -99999:
                break

            plan_s_idx = _state_to_index(plan_s)
            plan_a_idx = plan_a
            has_record = False
            for ts_check in range(time_slots):
                if memory.mem_strength[plan_s_idx, plan_a_idx, ts_check] > 0.0:
                    has_record = True
                    break
            if not has_record:
                valid_planning_steps += 1
                continue

            for ts in range(time_slots):
                if memory.mem_strength[plan_s_idx, plan_a_idx, ts] <= 0.0:
                    continue

                plan_time = (ts + 0.5) / time_slots
                time_delta = model.predict_next_time(plan_s, plan_a, plan_time)
                plan_next_time = (plan_time + time_delta) % 1.0

                plan_actions = np.empty(total_size, dtype=np.int32)
                n_plan_actions = 0
                plan_actions[n_plan_actions] = -9
                n_plan_actions += 1
                plan_actions[n_plan_actions] = 0
                n_plan_actions += 1
                plan_a_idx_local = _state_to_index(plan_a)
                for cand in range(1, selection_size + 1):
                    if memory.pair_active[plan_a_idx_local, cand]:
                        plan_actions[n_plan_actions] = cand
                        n_plan_actions += 1

                best_q = -np.inf
                optimal_a = plan_actions[0]
                for i in range(n_plan_actions):
                    a_next = plan_actions[i]
                    q = compute_Q(plan_a, a_next, plan_next_time, memory, model)
                    if q > best_q:
                        best_q = q
                        optimal_a = a_next

                next_sr_optimal = memory.retrieve_sr(plan_a, optimal_a, plan_next_time)
                current_sr_plan = memory.retrieve_sr(plan_s, plan_a, plan_time)
                onehot_plan = np.zeros(total_size, dtype=np.float64)
                onehot_plan[plan_a] = 1.0
                target_sr_plan = onehot_plan + next_sr_optimal
                new_sr_plan = current_sr_plan + alpha_plan * (target_sr_plan - current_sr_plan)

                memory.overwrite_slot_sr(plan_s, plan_a, ts, new_sr_plan)

            valid_planning_steps += 1

    return float(-loglik)


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
        simulate_and_loglik_sr_dyna(
            dummy_theta,
            dummy_data,
            selection_size=5,
            time_slots=48,
            sigma_t=1.0 / 12.0,
            visit_threshold=2,
            n_planning_steps=2,
            alpha_plan=0.1,
        )
    except Exception:
        pass
