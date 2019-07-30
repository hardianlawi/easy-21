import random
import numpy as np
from .agent import RLearningAgent
from .environment import Easy21


class SarsaAgent(RLearningAgent):
    def __init__(self, n_0=100, gamma=1, alpha=0.5, l=0.0):
        super().__init__()
        self.base_name = "Sarsa"
        self._n_0 = n_0
        self._gamma = gamma
        self._alpha = alpha
        self._lmda = l
        self._e_trace = np.zeros((11, 22, self._n_actions), dtype=np.float32)

    def train(self, steps: int, env: Easy21):
        for i in range(steps):
            cur_state = env.initial_step()
            while not env.has_terminated():
                action = self.act(cur_state, explore=True)
                next_state, reward = env.step(action)
                next_action = self.act(next_state, explore=True)
                self._update(reward, cur_state, action, next_state, next_action)
                cur_state = next_state
                action = next_action
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
        return self._id2action[action_id]

    def _update(self, reward, cur_state, action, next_state, next_action):
        action_id = self._action2id[action]
        next_action_id = self._action2id[next_action]

        self._e_trace[cur_state[0], cur_state[1], action_id] += 1

        delta = (
            reward
            + self._gamma * self._state_action_values.take(next_state)[next_action_id]
            - self._state_action_values.take(cur_state)[action_id]
        )

        self._state_action_values += self._alpha * delta * self._e_trace
        self._e_trace *= self._gamma * self._lmda

    def _refresh_cache(self):
        self._e_trace = np.zeros((11, 22, self._n_actions), dtype=np.float32)
