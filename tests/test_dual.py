"""Dual RL invariants: the corrected flow starts at BC and stays bounded, the
prior-candidate critic-weighted update tilts its default distribution, and the
runner pairs BC / reshaped / steered arms and round-trips the correction."""
from pathlib import Path
import tempfile
import unittest

import torch
from torch import nn

from flowq.dsrl_runner import evaluate_pair, load_checkpoint, train
from flowq.dual import DualLearner, ResidualFlow
from flowq.flow import DTYPE, Flow, fingerprint, freeze
from test_dsrl import TinyTask, TwoModeFlow, settings


def dual_settings(**dual):
    d = dict(residual_bound=4., flow_every=1, flow_lr=.003, flow_tau=.3, flow_samples=16,
             demo_weight=0., demo_batch=8)
    d.update(dual)
    return dict(settings(), dual=d)


class SignCritic(nn.Module):
    """Rewards the arm mode whose sign matches the (normalized) state."""
    def forward(self, x, a):
        q = 3*a[:, 0]*torch.sign(x[:, 0])
        return torch.stack([q, q])


class DualTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(11)
        self.flow = freeze(Flow(3, hidden=16, max_velocity=6.))

    def test_residual_flow_starts_at_bc_and_is_bounded(self):
        flow = ResidualFlow(self.flow, .5)
        s = torch.randn(8, 4, dtype=DTYPE)
        z = torch.randn(8, 6, dtype=DTYPE)
        self.assertTrue(torch.equal(flow.decode(s, z, steps=8), self.flow.decode(s, z, steps=8)))
        self.assertTrue(all(not p.requires_grad for p in flow.frozen_parameters()))
        self.assertTrue(all(p.requires_grad for p in flow.correction.parameters()))
        with torch.no_grad():
            flow.correction[-1].bias.fill_(100.)
            x = torch.randn(8, 6, dtype=DTYPE)
            diff = flow.velocity(.3, x, s)-self.flow.velocity(.3, x, s)
        self.assertLessEqual(diff.abs().max().item(), .5+1e-12)
        self.assertGreater(diff.min().item(), .49)
        with self.assertRaises(ValueError):
            ResidualFlow(Flow(3, hidden=16), .5)  # base must be frozen
        with self.assertRaises(ValueError):
            ResidualFlow(self.flow, 0.)

    def test_flow_update_tilts_prior_toward_rewarded_mode(self):
        base = freeze(TwoModeFlow())
        before_base = fingerprint(base)
        learner = DualLearner(base, dual_settings())
        learner.action_critic = SignCritic()
        states = torch.ones(256, 2, dtype=DTYPE)
        states[:128, 0] = -1

        def rewarded_fraction():
            z = torch.randn(len(states), 6, dtype=DTYPE, generator=torch.Generator().manual_seed(5))
            arm = learner.flow.decode(states, z, steps=8)
            return (arm[:, 0]*states[:, 0] > 0).double().mean().item()

        start = rewarded_fraction()
        for _ in range(300):
            metrics = learner.update_flow(states)
        end = rewarded_fraction()
        print(f'\nPrior-noise rewarded-mode fraction: BC={start:.3f}, corrected flow={end:.3f}')
        self.assertGreater(end, .7)
        self.assertGreater(end-start, .15)
        self.assertEqual(before_base, fingerprint(base))
        self.assertTrue(1 <= metrics['flow_weight_ess'] <= 16)
        self.assertEqual(metrics['flow_updates'], 300)
        self.assertGreater(metrics['flow_displacement_mean'], 0)

    def test_demo_anchor_requires_demonstrations(self):
        learner = DualLearner(self.flow, dual_settings(demo_weight=1.))
        with self.assertRaises(ValueError):
            learner.update_flow(torch.zeros(4, 4, dtype=DTYPE))
        anchored = DualLearner(self.flow, dual_settings(demo_weight=1.),
                               demos=(torch.zeros(16, 3), torch.full((16, 7), .5)))
        metrics = anchored.update_flow(torch.zeros(4, 4, dtype=DTYPE))
        self.assertIsNotNone(metrics['flow_demo_loss'])

    def test_runner_three_arm_evaluation_and_checkpoint_roundtrip(self):
        d = dict(dual_settings(flow_samples=4), seed=0, eval_seed=40000, batch_size=8,
                 replay_capacity=100, warmup_episodes=1, critic_warmup_updates=1,
                 pretrain_updates=2, online_steps=4, utd=2, eval_episodes=2, eval_interval=3,
                 log_stride=1, actor_mean_bound=1.5, train_flow_steps=4)
        d.pop('gamma')
        cfg = dict(task='lift', gamma=.99, horizon=3, versions={}, reward_scale=5., reward_bias=-1.,
                   train_mask='train', valid_mask='valid', debug_demos=None, boundary_epsilon=.001,
                   flow_max_velocity=6., dsrl=d)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            flow_path = root/'flow.pt'
            torch.save(dict(trained_on_original_demonstrations=True, task='lift', config=cfg,
                            obs_dim=3, hidden=16, flow=self.flow.state_dict(), manifest={'env': {}}), flow_path)
            report = train(cfg, flow_path, root/'online', env_factory=TinyTask)
            # Flow updates follow the actor schedule only: 2 pretrain + 4 online * utd 2.
            self.assertEqual(report['flow_updates'], 10)
            self.assertEqual(report['updates'], 11)
            self.assertEqual(set(report['evaluations'][-1]) & {'bc', 'reshaped', 'dsrl'},
                             {'bc', 'reshaped', 'dsrl'})
            cfg2, flow, bc, learner = load_checkpoint(root/'online/dsrl.pt', flow_path)
            self.assertIsInstance(learner, DualLearner)
            self.assertEqual(learner.flow_updates, 10)
            self.assertEqual(fingerprint(flow), fingerprint(self.flow))  # BC untouched
            ck = torch.load(root/'online/dsrl.pt', weights_only=False)
            self.assertEqual(ck['schema'], 'flowq.dsrl.v3')
            for k, v in ck['learner']['correction'].items():
                self.assertTrue(torch.equal(learner.flow.correction.state_dict()[k], v))
            # Zero correction + prior actor: all three arms are exactly BC on paired seeds.
            fresh = DualLearner(flow, dict(cfg2['dsrl'], gamma=cfg2['gamma']))
            paired = evaluate_pair(cfg2, flow, fresh, {}, TinyTask)
            self.assertEqual(paired['bc'], paired['reshaped'])
            self.assertEqual(paired['bc'], paired['dsrl'])


if __name__ == '__main__':
    unittest.main()
