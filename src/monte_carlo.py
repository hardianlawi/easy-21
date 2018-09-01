import random
import numpy as np


class MonteCarloAgent(object):

    def __init__(self, possible_actions, n_0=100):
        self._possible_actions = possible_actions
        self._n_actions = len(possible_actions)
        self._n_0 = n_0
        self._state_action_values = np.zeros(10, 21, self._n_actions)
        self._state_action_visits = np.zeros(10, 21, self._n_actions)
        self._past_action = None
        self._past_state = [None, None]

    def feedback(self, reward):

        past_state_action = self._past_state + [self._past_action]

        self._state_action_visits[past_state_action] += 1
        self._state_action_values[past_state_action] = \
            self._state_action_values[past_state_action] + \
            (reward - self._state_action_values[past_state_action]) / \
            self._state_action_visits[past_state_action]

    def take_action(self, state):

        n_0 = self._n_0
        state_action_visits = self._state_action_visits

        epsilon = n_0 / (n_0 + state_action_visits[state])

        # Choose action based on Epsilon-Greedy
        if random.random() < epsilon:
            action = random.choice(range(self._n_actions))
        else:
            action_values = self._state_action_values[state]
            action = np.argmax(action_values)

        self._past_state = state
        self._past_action = action

        return self._possible_actions[action]
