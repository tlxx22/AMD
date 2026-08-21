import unittest

import torch

from models.modules.modern_conv_refinement import (
    PeakPreservingModernConvRefinement,
)


class PMCRVariableIndependenceTests(unittest.TestCase):
    def _module(self):
        torch.manual_seed(2201)
        return PeakPreservingModernConvRefinement(
            hidden_dim=8,
            kernel_small=3,
            kernel_large=7,
            dropout=0.1,
            gamma_init=1e-3,
        ).eval()

    def test_single_variable_perturbation_cannot_change_other_deltas(self):
        module = self._module()
        hidden = torch.randn(2, 5, 12)
        changed = hidden.clone()
        changed[:, 2, :] += torch.linspace(-3.0, 4.0, 12)

        with torch.no_grad():
            before = module.compute_delta(hidden)
            after = module.compute_delta(changed)

        untouched = [0, 1, 3, 4]
        self.assertTrue(torch.equal(before[:, untouched], after[:, untouched]))
        self.assertGreater((before[:, 2] - after[:, 2]).abs().max().item(), 0.0)

    def test_variable_permutation_equivariance(self):
        module = self._module()
        hidden = torch.randn(3, 5, 12)
        permutation = torch.tensor([3, 0, 4, 1, 2])

        with torch.no_grad():
            expected = module.compute_delta(hidden)[:, permutation]
            actual = module.compute_delta(hidden[:, permutation])

        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
