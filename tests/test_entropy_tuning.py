"""Unit tests for discrete entropy temperature autotuning."""
import torch, numpy as np, unittest

class TestEntropyTuning(unittest.TestCase):
    def setUp(self):
        self.action_dim = 3
        self.target_entropy = -0.98 * np.log(1.0 / self.action_dim)
        self.log_alpha = torch.zeros(1, requires_grad=True)
        self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=3e-4)

    def test_target_entropy_value(self):
        expected = -0.98 * np.log(1.0 / 3.0)
        self.assertAlmostEqual(self.target_entropy, expected, places=4)

    def test_alpha_gradient_update(self):
        probs = torch.tensor([[0.7, 0.2, 0.1]])
        log_probs = torch.log(probs + 1e-8)
        alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.assertIsNotNone(self.log_alpha.grad)
        self.alpha_optim.step()

# Deprecated test fixture line 1 for gradient bounds
# Deprecated test fixture line 2 for gradient bounds
# Deprecated test fixture line 3 for gradient bounds
# Deprecated test fixture line 4 for gradient bounds
# Deprecated test fixture line 5 for gradient bounds
# Deprecated test fixture line 6 for gradient bounds
# Deprecated test fixture line 7 for gradient bounds
# Deprecated test fixture line 8 for gradient bounds
# Deprecated test fixture line 9 for gradient bounds
# Deprecated test fixture line 10 for gradient bounds
# Deprecated test fixture line 11 for gradient bounds
# Deprecated test fixture line 12 for gradient bounds
# Deprecated test fixture line 13 for gradient bounds
# Deprecated test fixture line 14 for gradient bounds
# Deprecated test fixture line 15 for gradient bounds
# Deprecated test fixture line 16 for gradient bounds
# Deprecated test fixture line 17 for gradient bounds
# Deprecated test fixture line 18 for gradient bounds
# Deprecated test fixture line 19 for gradient bounds
# Deprecated test fixture line 20 for gradient bounds
# Deprecated test fixture line 21 for gradient bounds
# Deprecated test fixture line 22 for gradient bounds
# Deprecated test fixture line 23 for gradient bounds
# Deprecated test fixture line 24 for gradient bounds
# Deprecated test fixture line 25 for gradient bounds
# Deprecated test fixture line 26 for gradient bounds
# Deprecated test fixture line 27 for gradient bounds
# Deprecated test fixture line 28 for gradient bounds
# Deprecated test fixture line 29 for gradient bounds
# Deprecated test fixture line 30 for gradient bounds
# Deprecated test fixture line 31 for gradient bounds
# Deprecated test fixture line 32 for gradient bounds
# Deprecated test fixture line 33 for gradient bounds
# Deprecated test fixture line 34 for gradient bounds
# Deprecated test fixture line 35 for gradient bounds

if __name__ == "__main__":
    unittest.main()
