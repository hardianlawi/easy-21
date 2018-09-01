import random
import numpy as np


class MonteCarloAgent(object):

    def __init__(self, possible_actions, n_0=100):
        self._possible_actions = possible_actions
        self._n_actions = len(possible_actions)
        self._n_0 = n_0
        self._state_action_values = np.zeros(
            (11, 21, self._n_actions), dtype=np.float32)
        self._state_action_visits = np.zeros(
            (11, 21, self._n_actions), dtype=np.float32)
        self._past_action = None
        self._past_state = [None, None]

    def receive_feedback(self, reward):

        d_H, p_H = self._past_state
        a = self._past_action

        self._state_action_visits[d_H, p_H, a] += 1
        self._state_action_values[d_H, p_H, a] = \
            self._state_action_values[d_H, p_H, a] + \
            (reward - self._state_action_values[d_H, p_H, a]) / \
            self._state_action_visits[d_H, p_H, a]

    def take_action(self, state):

        n_0 = self._n_0

        epsilon = n_0 / (n_0 + self._state_action_visits.take(state).sum())

        # Choose action based on Epsilon-Greedy
        if random.random() < epsilon:
            action = random.choice(range(self._n_actions))
        else:
            action_values = self._state_action_values.take(state)
            action = np.argmax(action_values)

        self._past_state = state
        self._past_action = action

        return self._possible_actions[action]
