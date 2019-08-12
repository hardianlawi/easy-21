import unittest
from src.monte_carlo import MonteCarloAgent


class TestMonteCarlo(unittest.TestCase):
    def setUp(self):
        self.env = Easy21()

    def test_initial_step(self):
        for i in range(10000):
            d_H, p_H = self.env.initial_step()
            self.assertLessEqual(d_H, 10)
            self.assertLessEqual(p_H, 10)
            self.assertGreaterEqual(d_H, 1)
            self.assertGreaterEqual(p_H, 1)

    def test_move(self):
        s = self.env.initial_step()
        s, r = self.env.step("hit")
        self.assertListEqual(s, self.env._cur_state)

    def test_get_reward(self):

        d_H, p_H = 5, 5
        reward = self.env._get_reward(d_H, p_H)
        self.assertEqual(reward, 0)

        d_H, p_H = 22, 5
        reward = self.env._get_reward(d_H, p_H)
        self.assertEqual(reward, 1)

        d_H, p_H = 5, 22
        reward = self.env._get_reward(d_H, p_H)
        self.assertEqual(reward, -1)
