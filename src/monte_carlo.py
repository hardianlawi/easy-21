import os
import gc
import pickle
import random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from src.agent import RLearningAgent


class MonteCarloAgent(RLearningAgent):

    def __init__(self, possible_actions, n_0=100, gamma=1,
                 base_name='monte_carlo'):
        self._possible_actions = possible_actions
        self._n_actions = len(possible_actions)
        self._gamma = gamma
        self._n_0 = n_0
        self._base_name = base_name
        self._state_action_values = np.zeros(
            (11, 22, self._n_actions), dtype=np.float32)
        self._state_action_visits = np.zeros(
            (11, 22, self._n_actions), dtype=np.float32)
        self._past_actions = []
        self._past_states = []
        self._past_returns = []

    def take_action(self, state, explore=True):

        n_0 = self._n_0

        epsilon = n_0 / (n_0 + self._state_action_visits.take(state).sum())

        # Choose action based on Epsilon-Greedy
        if explore and random.random() < epsilon:
            action = random.choice(range(self._n_actions))
        else:
            action_values = self._state_action_values.take(state)
            action = np.argmax(action_values)

        self._past_states.append(state)
        self._past_actions.append(action)

        return self._possible_actions[action]

    def receive_feedback(self, reward):

        gamma = self._gamma
        past_returns = self._past_returns
        n_states = len(past_returns)

        for i, r in enumerate(past_returns):
            past_returns[i] = r + gamma**(n_states - i) * reward

        past_returns.append(reward)

    def update(self):

        for (d_H, p_H), a, g in zip(self._past_states,
                                    self._past_actions,
                                    self._past_returns):

            # Increase N(s, a)
            self._state_action_visits[d_H, p_H, a] += 1

            # Increase Q(s, a)
            # Monte Carlo Incremental update
            # q(s, a) = q(s, a) + 1 / n(s, a) * (g - q(s, a))
            self._state_action_values[d_H, p_H, a] += \
                ((g - self._state_action_values[d_H, p_H, a]) /
                 self._state_action_visits[d_H, p_H, a])

        self._clear_cache()

    def _clear_cache(self):
        self._past_actions = []
        self._past_states = []
        self._past_returns = []
        gc.collect()

    def save(self, output_dir, iteration=None):

        output_dir = os.path.join(output_dir, self._base_name)

        # Clear cache to avoid saving unnecessary data
        self._clear_cache()

        # Convert numpy array to list for pickability
        self._state_action_values = self._state_action_values.tolist()
        self._state_action_visits = self._state_action_visits.tolist()

        # Pickle class
        self._save_local(output_dir, iteration, extension='pkl')

        # Convert back to numpy array for reusability
        self._state_action_values = np.asarray(self._state_action_values)
        self._state_action_visits = np.asarray(self._state_action_visits)

        # Save plots
        self._save_plot(output_dir, iteration)

    def _save_plot(self, output_dir, iteration=None):

        x, y = np.meshgrid(np.arange(1, 11),
                           np.arange(1, 22))

        state_action_values = self._state_action_values[1:, 1:, 0].T
        state_action_visits = self._state_action_visits[1:, 1:, 0].T

        fig = plt.figure()

        ax = fig.add_subplot(1, 2, 1, projection='3d')
        ax.plot_surface(x, y, state_action_values,
                        cmap='viridis', edgecolor='none')
        ax.set_title('State action values')

        ax = fig.add_subplot(1, 2, 2, projection='3d')
        ax.plot_surface(x, y, state_action_visits,
                        cmap='viridis', edgecolor='none')
        ax.set_title('State action visits')

        self._save_local(output_dir, iteration, extension='png')

    def load(fname):
        with open(fname, 'rb') as f:
            self = pickle.load(f)
        self._state_action_values = np.asarray(self._state_action_values)
        self._state_action_visits = np.asarray(self._state_action_visits)
        return self
