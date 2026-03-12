"""
Base class for memory records.
"""


from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Any
from collections import defaultdict

@dataclass
class EntryRecord:
    """Base class for memory records."""
    time_angle: float
    day_seq: int
    record_date: int
    strength: float = 1.0



class EpisodicMemory:
    """
    Base episodic memory with common methods for both Q-Memory and SR-Memory.
    """

    def __init__(self, phi: float, config: Any):
        self.config = config
        self.phi = phi
        self.memory: Dict[int, Dict[int, List[EntryRecord]]] = defaultdict(lambda: defaultdict(list))
        self.memory_decay: Dict[int, Dict[int, float]] = defaultdict(lambda: defaultdict(float))
        self.node_strength: Dict[int, float] = {}
        self.node_visits: Dict[int, int] = {}
        self.last_day: Optional[int] = None
        self._active_states: Optional[Set[int]] = None

    def _add_record_base(self, rec: EntryRecord, node_id: int, action_id: int):
        """Common record addition logic."""
        self.memory[node_id][action_id].append(rec)
        self.memory_decay[node_id][action_id] = 1.0
        self.node_strength[node_id] = self.node_strength.get(node_id, 0.0) + 1.0
        self.node_visits[node_id] = self.node_visits.get(node_id, 0) + 1
        self._active_states = None

    def decay(self, current_day: int):
        """Apply memory decay when day changes. Identical for both Q-Memory and SR-Memory."""
        if self.last_day is None:
            self.last_day = current_day
            return

        day_diff = int(current_day - self.last_day)
        if day_diff > 0:
            factor = (1.0 - self.phi) ** day_diff
            for s in self.memory:
                for a in self.memory[s]:
                    for rec in self.memory[s][a]:
                        rec.strength *= factor
            for node_id in self.node_strength:
                self.node_strength[node_id] *= factor
                for action_id in self.memory_decay[node_id]:
                    self.memory_decay[node_id][action_id] *= factor
            self._active_states = None

        self.last_day = current_day


    def get_records_for_sa_pair(self, node_id: int, action_id: int) -> List[EntryRecord]:
        return self.memory.get(int(node_id), {}).get(int(action_id), [])

    def get_active_states(self) -> Set[int]:
        """Get states with memory strength above threshold."""
        if self._active_states is not None:
            return self._active_states
        self._active_states = {
            state_id
            for state_id, strength in self.node_strength.items()
            if strength >= self.config.memory_threshold
            and self.node_visits.get(state_id, 0) >= self.config.visit_threshold
        }
        return self._active_states

