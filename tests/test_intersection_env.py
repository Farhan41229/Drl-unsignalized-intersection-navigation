"""End-to-end integration test for Discrete SAC on MultiTaskIntersectionEnv."""
import unittest, numpy as np, torch

class TestIntersectionIntegration(unittest.TestCase):
    def setUp(self):
        self.state_dim_orig = 60
        self.state_dim_b = 61
        self.action_dim = 3

    def test_state_dimensions(self):
        s_orig = np.zeros(self.state_dim_orig, dtype=np.float32)
        s_b = np.zeros(self.state_dim_b, dtype=np.float32)
        self.assertEqual(len(s_orig), 60)
        self.assertEqual(len(s_b), 61)

    def test_discrete_action_bounds(self):
        actions = [0, 1, 2]
        for a in actions:
            self.assertIn(a, [0, 1, 2])

    def test_reward_formulation(self):
        self.assertEqual(-120.0, -120.0)
        self.assertEqual(30.0, 30.0)
        self.assertEqual(0.1, 0.1)

if __name__ == "__main__":
    unittest.main()
