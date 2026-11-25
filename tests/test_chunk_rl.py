"""Chunk-level DSRL / dual RL: pairing with chunked BC, chunk TD target, gripper
expectation, the chunk flow correction, and runner bookkeeping."""
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from torch import nn

from flowq.chunk import ChunkFlow, save as save_chunk_flow
from flowq.chunk_rl import (ChunkDSRLLearner, ChunkDualLearner, act, decode_chunk, evaluate,
                            load_checkpoint, train)
from flowq.flow import DTYPE, fingerprint, freeze, tensor
from test_dsrl import TinyTask, settings


def dual_block(**kw):
    d = dict(residual_bound=1., flow_every=1, flow_lr=.003, flow_tau=1., flow_samples=4,
             demo_weight=0., demo_batch=8)
    d.update(kw)
    return d


class GripperSumCritic(nn.Module):
    """Q = sum of the two gripper commands of a chunk of 2."""
    def forward(self, x, a):
        q = a[:, 6]+a[:, 13]
        return torch.stack([q, q])


class ChunkRLTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(5)
        self.flow = freeze(ChunkFlow(3, 2, hidden=16, max_velocity=6.))
        self.state = np.array([.1, -.2, .3, 1.])

    def test_baseline_and_fresh_learner_reproduce_chunked_bc(self):
        learner = ChunkDSRLLearner(self.flow, settings())
        expected = self.flow.sample(tensor(self.state)[None], 8, torch.Generator().manual_seed(9))[0].numpy()
        bc, _ = act(self.flow, learner, self.state, torch.Generator().manual_seed(9), 8, baseline=True)
        steered, _ = act(self.flow, learner, self.state, torch.Generator().manual_seed(9), 8)
        self.assertTrue(np.array_equal(bc, expected))
        self.assertTrue(np.array_equal(steered, expected))  # untrained noise policy = N(0, I)
        self.assertEqual(learner.actor.latent_dim, 12)

    def test_chunk_td_target_uses_discount_of_executed_steps(self):
        learner = ChunkDSRLLearner(self.flow, settings())
        learner.decoded_value = lambda critic, s, z: torch.full((len(s),), 10., dtype=DTYPE)
        discounts = torch.full((2,), .99**2, dtype=DTYPE)
        target = learner.td_target(torch.tensor([4., -1.5], dtype=DTYPE), torch.zeros(2, 4, dtype=DTYPE),
                                   torch.tensor([1., 0.], dtype=DTYPE), discounts)
        torch.testing.assert_close(target, torch.tensor([4., -1.5+.99**2*10], dtype=DTYPE))

    def test_gripper_expectation_is_monte_carlo_over_the_frozen_head(self):
        learner = ChunkDSRLLearner(self.flow, dict(settings(), gripper_samples=4000))
        with torch.no_grad():
            for p in self.flow.gripper.parameters():
                p.zero_()
            self.flow.gripper[-1].bias.copy_(torch.tensor([.2, .3, .5], dtype=DTYPE).log().repeat(2))
        s = torch.zeros(3, 4, dtype=DTYPE)
        value = learner.action_value(GripperSumCritic(), s, torch.zeros(3, 12, dtype=DTYPE))
        # Per step E[g] = -0.2 + 0.5 = 0.3; two steps -> 0.6 (MC standard error ~0.02).
        self.assertLess((value-.6).abs().max().item(), .08)

    def test_chunk_correction_starts_at_bc_and_trains(self):
        before = fingerprint(self.flow)
        learner = ChunkDualLearner(self.flow, dict(settings(), dual=dual_block(correction_hidden=8,
                                                                                correction_layers=3)))
        s = torch.randn(6, 4, dtype=DTYPE)
        z = torch.randn(6, 12, dtype=DTYPE)
        self.assertTrue(torch.equal(learner.flow.decode(s, z, 8), decode_chunk(self.flow, s, z, 8)))
        correction = fingerprint(learner.flow.correction)
        metrics = learner.update_flow(s)
        self.assertEqual(metrics['flow_updates'], 1)
        self.assertNotEqual(correction, fingerprint(learner.flow.correction))
        self.assertEqual(before, fingerprint(self.flow))
        with self.assertRaises(ValueError):  # anchor without demonstrations
            ChunkDualLearner(self.flow, dict(settings(), dual=dual_block(demo_weight=1.))).update_flow(s)

    def _cfg(self, dual=None):
        d = dict(settings(), seed=0, eval_seed=40000, batch_size=8, replay_capacity=100, warmup_episodes=1,
                 critic_warmup_updates=1, pretrain_updates=1, online_steps=4, utd=2, eval_episodes=2,
                 eval_interval=3, log_stride=1, train_flow_steps=4, gripper_samples=2)
        d.pop('gamma')
        if dual is not None:
            d['dual'] = dual
        return dict(task='lift', gamma=.99, horizon=3, versions={}, reward_scale=5., reward_bias=-1.,
                    boundary_epsilon=.001, flow_max_velocity=6., dsrl=d)

    def test_runner_bookkeeping_and_checkpoint_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            flow_path = root/'flow.pt'
            save_chunk_flow(self.flow, self._cfg(), {'env': {}, 'train_demos': []}, 1, flow_path)
            for name, dual in (('dsrl', None), ('dual', dual_block())):
                cfg = self._cfg(dual)
                report = train(cfg, flow_path, root/name, env_factory=TinyTask)
                # Horizon 3, chunk 2: warmup = 2 decisions / 3 steps; online stops at 5 steps
                # (2 + 1 [episode end] + 2) after 3 decisions; updates = 1 + 1 + 3 * utd 2.
                self.assertEqual(report['warmup_env_steps'], 3)
                self.assertEqual((report['online_env_steps'], report['online_decisions']), (5, 3))
                self.assertEqual(report['updates'], 8)
                self.assertEqual(report['retained_replay_transitions'], 5)
                arms = {'bc', 'dsrl'} | ({'reshaped'} if dual else set())
                self.assertEqual(set(report['evaluations'][-1]) & {'bc', 'reshaped', 'dsrl'}, arms)
                if dual:
                    self.assertEqual(report['flow_updates'], 7)  # 1 pretrain + 6 online
                loaded_cfg, base, _, learner = load_checkpoint(root/name/'chunk_rl.pt', flow_path)
                self.assertIsInstance(learner, ChunkDualLearner if dual else ChunkDSRLLearner)
                self.assertEqual(fingerprint(base), fingerprint(self.flow))
            # A fresh learner is exactly BC on every arm (paired seeds and noise).
            fresh = ChunkDualLearner(base, dict(loaded_cfg['dsrl'], gamma=loaded_cfg['gamma']))
            paired = evaluate(loaded_cfg, base, fresh, {}, TinyTask)
            self.assertEqual(paired['bc'], paired['reshaped'])
            self.assertEqual(paired['bc'], paired['dsrl'])


if __name__ == '__main__':
    unittest.main()
