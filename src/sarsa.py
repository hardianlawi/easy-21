import random
import numpy as np
from .agent import RLearningAgent
from .environment import Easy21


class SarsaAgent(RLearningAgent):
    def __init__(self, n_0=100, gamma=1.0, lmda=1.0):
        super().__init__()
        self.base_name = "Sarsa"
        self._n_0 = n_0
        self._gamma = gamma
        self._lmda = lmda
        self._e_trace = np.zeros((11, 22, self._n_actions), dtype=np.float32)

    def __str__(self):
        return self.base_name + "_gamma_%s_" % self._gamma + "_lmda+%s" % self._lmda

    def train(self, steps: int, env: Easy21):
        for i in range(steps):
            cur_state = env.initial_step()
            action = self.act(cur_state, explore=True)
            while not env.has_terminated():
                next_state, reward = env.step(action)
                if not env.has_terminated():
                    next_action = self.act(next_state, explore=True)
                    self._update(reward, cur_state, action, next_state, next_action)
                    cur_state = next_state
                    action = next_action
                else:
                    self._update(reward, cur_state, action)
            self._refresh_cache()
            env.clear()

    def act(self, state, explore=True) -> str:
        d_H, p_H = state
        n_0 = self._n_0
        epsilon = n_0 / (n_0 + self._state_action_visits[d_H, p_H].sum())
        # Choose action based on Epsilon-Greedy
        if explore and random.random() < epsilon:
            action_id = random.choice(range(self._n_actions))
        else:
            action_values = self._state_action_values[d_H, p_H]
            action_id = np.argmax(action_values)
        return self._id2action[action_id]

    def _update(
        self,
        reward: float,
        cur_state: tuple,
        action: str,
        next_state: tuple = None,
        next_action: str = None,
    ):

        action_id = self._action2id[action]

        self._state_action_visits[cur_state[0], cur_state[1], action_id] += 1
        self._e_trace[cur_state[0], cur_state[1], action_id] += 1

        delta = (
            reward - self._state_action_values[cur_state[0], cur_state[1], action_id]
        )

        if next_state is not None and next_action is not None:
            next_action_id = self._action2id[next_action]
            delta += (
                self._gamma
                * self._state_action_values[
                    next_state[0], next_state[1], next_action_id
                ]
            )

        update = delta * self._e_trace

        # Using the visits as alpha for the action values to converge to real values
        alpha = 1 / self._state_action_visits[cur_state[0], cur_state[1], action_id]

        self._state_action_values += alpha * update
        self._e_trace *= self._gamma * self._lmda

    def _refresh_cache(self):
        self._e_trace = np.zeros((11, 22, self._n_actions), dtype=np.float32)
