import random
import numpy as np
from collections import deque
from .agent import RLearningAgent
from .environment import Easy21


class TDLearningAgent(RLearningAgent):
    def __init__(self, n_0=100, gamma=1):
        self._base_name = "Sarsa"
        self._gamma = gamma
        self._n_0 = n_0
        self._past_actions = deque([], 2)
        self._past_states = deque([], 2)
        self._past_returns = deque([], 2)

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
        self._past_returns.append(reward)

        # TODO: Fix here
        self._update()

    def _memorize(self, state, action_id):
        self._past_states.append(state)
        self._past_actions.append(action_id)

    def _update(self, terminate):

        gamma = self._gamma
        past_states = self._past_states
        past_actions = self._past_actions
        past_returns = self._past_returns

        if terminate and len(past_states) == 1:
            d_H_1, p_H_1 = past_states[0]
            a_1 = past_actions[0]
        else:
            (d_H, p_H), (d_H_1, p_H_1) = past_states[0], past_states[1]
            a, a_1 = past_actions

        r = past_returns.popleft()

        # Increase Q(s, a)
        if not terminate:
            self._state_action_visits[d_H, p_H, a] += 1
            # q(s, a) += 1 / n(s, a) * (r + gammba * q(s', a') - q(s, a))
            self._state_action_values[d_H, p_H, a] += (
                r
                + gamma * self._state_action_values[d_H_1, p_H_1, a_1]
                - self._state_action_values[d_H, p_H, a]
            ) / self._state_action_visits[d_H, p_H, a]
        else:
            self._state_action_visits[d_H_1, p_H_1, a_1] += 1
            # q(s, a) += 1 / n(s, a) * (r - q(s, a))
            self._state_action_values[d_H_1, p_H_1, a_1] += (
                r - self._state_action_values[d_H_1, p_H_1, a_1]
            ) / self._state_action_visits[d_H_1, p_H_1, a_1]

    def _clear_cache(self):
        self._past_actions = deque([], 2)
        self._past_states = deque([], 2)
        self._past_returns = deque([], 2)
