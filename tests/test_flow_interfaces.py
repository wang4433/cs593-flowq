"""Regression checks for the frozen-flow boundary used by latent RL."""
import unittest

import torch

from flowq.flow import DTYPE, Flow, fingerprint, freeze, rk4


class FlowInterfaces(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(4)
        self.flow = freeze(Flow(3, hidden=16, max_velocity=6.))
        self.s = torch.randn(12, 4, dtype=DTYPE)  # extra time feature is ignored by BC

    def test_sample_preserves_legacy_rng_and_actions(self):
        g = torch.Generator().manual_seed(10)
        z = torch.randn(12, 6, dtype=DTYPE, generator=g)
        arm = rk4(lambda t, x: self.flow.velocity(t, x, self.s), z, 0, 1, 16).tanh()
        grip = torch.multinomial(self.flow.gripper_dist(self.s).probs, 1, generator=g)-1
        expected = torch.cat([arm, grip.to(DTYPE)], -1)
        actual = self.flow.sample(self.s, generator=torch.Generator().manual_seed(10))
        self.assertTrue(torch.equal(actual, expected))

    def test_roundtrip_and_frozen_inference(self):
        before = fingerprint(self.flow)
        z = torch.randn(12, 6, dtype=DTYPE, requires_grad=True)
        arm = self.flow.decode(self.s, z, steps=32)
        inverse = self.flow.encode(self.s, arm, steps=32)
        torch.testing.assert_close(inverse, z, atol=1e-7, rtol=1e-7)
        self.assertFalse(arm.requires_grad)
        self.assertEqual(before, fingerprint(self.flow))
        self.assertTrue(all(p.grad is None for p in self.flow.parameters()))

    def test_invalid_inputs_and_extreme_latents_fail_explicitly(self):
        a = torch.zeros(12, 7, dtype=DTYPE)
        a[0, 0] = float('nan')
        with self.assertRaises(FloatingPointError):
            self.flow.loss(self.s, a)
        with self.assertRaises(ValueError):
            self.flow.encode(self.s, torch.ones(12, 6, dtype=DTYPE))
        with self.assertRaises(FloatingPointError):
            self.flow.decode(self.s, torch.full((12, 6), 100., dtype=DTYPE))

    def test_configurable_depth_keeps_legacy_parameter_names(self):
        # Default depth is the original 2-layer network, so old checkpoints still load.
        legacy = Flow(3, hidden=16, max_velocity=6.)
        self.assertEqual(legacy.layers, 2)
        self.assertEqual(sorted(legacy.state_dict()), sorted(Flow(3, hidden=16, layers=2).state_dict()))
        deep = freeze(Flow(3, hidden=16, max_velocity=6., layers=4))
        linears = [m for m in deep.net if isinstance(m, torch.nn.Linear)]
        self.assertEqual(len(linears), 5)  # 4 hidden layers + output
        self.assertEqual(len([m for m in deep.gripper if isinstance(m, torch.nn.Linear)]), 5)
        with self.assertRaises(ValueError):
            Flow(3, hidden=16, layers=0)
        z = torch.randn(12, 6, dtype=DTYPE)
        inverse = deep.encode(self.s, deep.decode(self.s, z, steps=32), steps=32)
        torch.testing.assert_close(inverse, z, atol=1e-6, rtol=1e-6)


if __name__ == '__main__':
    unittest.main()
