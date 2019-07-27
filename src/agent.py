import os
import pickle
import numpy as np
from abc import ABC, abstractmethod


class RLearningAgent(ABC):
    def __init__(self, possible_actions=["stick", "hit"]):
        self._possible_actions = possible_actions
        self._n_actions = len(possible_actions)
        self._action2id = dict(zip(possible_actions, range(self._n_actions)))
        self._id2action = dict(zip(range(self._n_actions), possible_actions))
        self._state_action_values = np.zeros(
            (11, 22, self._n_actions), dtype=np.float32
        )
        self._state_action_visits = np.zeros(
            (11, 22, self._n_actions), dtype=np.float32
        )

    def get_action_values(self):
        return self._state_action_values.copy()

    @abstractmethod
    def train(self, steps, environment):
        for e in range(steps):
            raise Exception("Running abstract training")
        return
