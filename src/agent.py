import os
import pickle
import numpy as np
from abc import ABC, abstractmethod
from .environment import Easy21


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
    def act(self, state, explore=True):
        return

    @abstractmethod
    def train(self, steps, env):
        for e in range(steps):
            raise Exception("Running abstract training")
        return

    def eval(self, steps: int, env: Easy21):
        """Evaluate performance of agent by playing a number of steps in the same environment

        Args:
            steps (int): number of episodes to play the game
            env (Easy21): environment

        Returns:
            float: percentage of the agent winning excluding the ties.
        """
        win = 0
        count = 0
        for i in range(steps):
            env.clear()
            cur_state = env.initial_step()
            while not env.has_terminated():
                action = self.act(cur_state, explore=False)
                cur_state, reward = env.step(action)
            if reward != 0:
                count += 1
                if reward != -1:
                    win += 1
        return win / count
