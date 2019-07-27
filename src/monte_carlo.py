import os
import gc
import pickle
import random
import numpy as np
import matplotlib.pyplot as plt
from .environment import Easy21
from .agent import RLearningAgent


class MonteCarloAgent(RLearningAgent):
    def __init__(self, n_0=100, gamma=1, method="first", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.base_name = "monte_carlo_" + method
        self._gamma = gamma
        self._n_0 = n_0
        self._past_actions = []
        self._past_states = []
        self._past_returns = []

    def train(self, steps: int, env: Easy21):
        for i in range(steps):
            cur_state = env.initial_step()
            action = self.act(cur_state, explore=True)
            while not env.has_terminated():
                cur_state, reward = env.step(action)
                self.observe(reward)
            self._update()
            env.clear()

    def act(self, state, explore=True):
        n_0 = self._n_0
        epsilon = n_0 / (n_0 + self._state_action_visits.take(state).sum())
        # Choose action based on Epsilon-Greedy
        if explore and random.random() < epsilon:
            action_id = random.choice(range(self._n_actions))
        else:
            action_values = self._state_action_values.take(state)
            action_id = np.argmax(action_values)
        if explore:
            self._memorize(state, action_id)
        return self._id2action[action_id]

    def observe(self, reward):
        gamma = self._gamma
        past_returns = self._past_returns
        n_states = len(past_returns)
        for i, r in enumerate(past_returns):
            past_returns[i] = r + gamma ** (n_states - i) * reward
        past_returns.append(reward)

    def _memorize(self, state, action_id):
        self._past_states.append(state)
        self._past_actions.append(action_id)

    def _update(self):
        for (d_H, p_H), a, g in zip(
            self._past_states, self._past_actions, self._past_returns
        ):
            # Increase N(s, a)
            self._state_action_visits[d_H, p_H, a] += 1

            # Increase Q(s, a)
            # Monte Carlo Incremental update
            # q(s, a) = q(s, a) + 1 / n(s, a) * (g - q(s, a))
            self._state_action_values[d_H, p_H, a] += (
                g - self._state_action_values[d_H, p_H, a]
            ) / self._state_action_visits[d_H, p_H, a]

        self._clear_cache()

    def _clear_cache(self):
        self._past_actions = []
        self._past_states = []
        self._past_returns = []
        gc.collect()
