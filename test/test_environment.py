import unittest
from src.environment import Easy21


class TestEnvironment(unittest.TestCase):

    def setUp(self):
        self.env = Easy21()

    def test_initial_step(self):
        for i in range(10000):
            d_H, p_H = self.env.initial_step()
            self.assertLessEqual(d_H, 10)
            self.assertLessEqual(p_H, 10)
            self.assertGreaterEqual(d_H, 1)
            self.assertGreaterEqual(p_H, 1)

    def test_step(self):
        pass

    def test_get_reward(self):

        d_H, p_H = 5, 5
        reward = self.env._get_reward(d_H, p_H, False)
        self.assertEqual(reward, 0)

        d_H, p_H = 22, 5
        reward = self.env._get_reward(d_H, p_H, False)
        self.assertEqual(reward, 1)

        d_H, p_H = 5, 22
        reward = self.env._get_reward(d_H, p_H, False)
        self.assertEqual(reward, -1)

        d_H, p_H = 17, 18
        reward = self.env._get_reward(d_H, p_H, True)
        self.assertEqual(reward, 1)
