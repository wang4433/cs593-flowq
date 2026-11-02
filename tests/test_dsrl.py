"""Algorithmic invariants and a controlled multimodal latent-steering problem."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from flowq.dsrl import ActionReplay, DSRLLearner, LatentActor, state_features
from flowq.dsrl_runner import act, evaluate_pair, load_checkpoint, load_config, train, validate_config
from flowq.flow import DTYPE, Flow, fingerprint, freeze, tensor


def settings():
    return dict(hidden=32, lr=.003, gamma=.99, tau=.02, grad_clip=10., kl_beta=.03,
                flow_steps=8, prior_fraction=.5)


class TwoModeFlow(Flow):
    """Smooth invertible outward drift creates two arm modes; no learned weights."""
    def __init__(self):
        super().__init__(1, hidden=16, max_velocity=2.)

    def velocity(self, t, x, s):
        v = torch.zeros_like(x)
        v[:, 0] = 2*torch.tanh(3*x[:, 0])
        return v


class AffineCritic(nn.Module):
    def forward(self, x, a):
        q = a[:, 0]+2*a[:, -1]
        return torch.stack([q, q+1])


class TinyTask:
    """Three-step environment for runner bookkeeping, never a robotics benchmark."""
    def __init__(self, cfg, metadata):
        self.horizon = cfg['horizon']

    def reset(self, seed):
        np.random.seed(seed)
        self.t = 0
        return np.array([.1, -.1, .2, 1.])

    def step(self, a):
        self.t += 1
        done = self.t == self.horizon
        return np.array([.1, -.1, .2, 1-self.t/self.horizon]), -1., done, dict(success=False)

    def close(self):
        pass


class DSRLTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(7)
        np.random.seed(7)
        self.flow = freeze(Flow(3, hidden=16, max_velocity=6.))

    def test_actor_bounds_kl_and_gradients(self):
        actor = LatentActor(4, hidden=16, mean_bound=1.5)
        s = torch.ones(3, 4, dtype=DTYPE)
        with torch.no_grad():
            actor.net[-1].bias[:6].copy_(tensor([-100, -2, -.5, .5, 2, 100]))
            actor.net[-1].bias[6:].copy_(tensor([-100, -2, -.5, .5, 2, 100]))
        dist = actor.distribution(s)
        self.assertLessEqual(dist.mean.abs().max().item(), 1.5)
        self.assertLessEqual(dist.scale.max().item(), 1.)
        self.assertGreater(dist.scale.min().item(), 0.)
        prior = torch.distributions.Normal(torch.zeros_like(dist.mean), torch.ones_like(dist.scale))
        torch.testing.assert_close(actor.kl(s), torch.distributions.kl_divergence(dist, prior).sum(-1))
        actor.sample(s).sum().backward()
        grad = actor.net[-1].bias.grad
        self.assertTrue(torch.isfinite(grad).all())
        self.assertGreater(grad[2].abs().item(), 0.)  # mean gradient through tanh
        self.assertGreater(grad[8].abs().item(), 0.)  # std gradient inside cap

    def test_training_decoder_steps_are_separate_from_execution(self):
        cfg = dict(settings(), flow_steps=16, train_flow_steps=8)
        learner = DSRLLearner(self.flow, cfg)
        s = torch.zeros(4, 4, dtype=DTYPE)
        with patch.object(self.flow, 'decode', wraps=self.flow.decode) as decode:
            learner.update((s, torch.zeros(4, 7, dtype=DTYPE), torch.zeros(4, dtype=DTYPE),
                            s, torch.zeros(4, dtype=DTYPE)))
            self.assertEqual([c.kwargs['steps'] for c in decode.call_args_list], [8, 8])
            act(self.flow, learner, np.zeros(4), torch.Generator().manual_seed(5), 16)
            # action_from_noise passes steps positionally.
            self.assertEqual(decode.call_args.args[2], 16)

    def test_shipped_budgets_and_validation(self):
        root = Path(__file__).resolve().parents[1]
        for task in ('lift', 'can', 'square'):
            cfg = load_config(root/f'configs/{task}_dsrl.json')
            d = cfg['dsrl']
            self.assertEqual((d['warmup_episodes'], d['online_steps'], d['utd']), (100, 100000, 20))
            self.assertEqual((d['train_flow_steps'], d['flow_steps']), (8, 16))
            for key, value in (('actor_mean_bound', float('nan')), ('actor_mean_bound', 0),
                               ('train_flow_steps', 0), ('train_flow_steps', 1.5), ('kl_beta', -1)):
                bad = copy.deepcopy(cfg)
                bad['dsrl'][key] = value
                with self.assertRaises(ValueError):
                    validate_config(bad)

    def test_actor_starts_at_bc_prior_and_unbounded_replay(self):
        actor = LatentActor(4, hidden=16)
        s = torch.randn(4, 4, dtype=DTYPE)
        self.assertTrue(torch.equal(actor.kl(s), torch.zeros(4, dtype=DTYPE)))
        z = actor.sample(s, generator=torch.Generator().manual_seed(1))
        expected = torch.randn(4, 6, dtype=DTYPE, generator=torch.Generator().manual_seed(1))
        self.assertTrue(torch.equal(z, expected))
        replay = ActionReplay(4, capacity=2)
        for i in range(3):
            replay.add(np.zeros(4), [1., 0, 0, 0, 0, 0, -1], i, np.ones(4), True,
                       noise=np.full(6, 5.) if i == 0 else None)
        self.assertEqual(len(replay), 2)
        self.assertEqual(sorted(r[2] for r in replay.rows), [1, 2])
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/'replay.npz'
            replay.save(p, {'task': 'test'})
            restored = ActionReplay(4)
            self.assertEqual(restored.load(p, {'task': 'test'}), 2)
            self.assertTrue(all(z is None for z in restored.noises))
            with self.assertRaises(ValueError):
                restored.load(p, {'task': 'different'})
        with self.assertRaises(ValueError):
            replay.add(np.zeros(4), np.zeros(7), 0, np.zeros(4), float('nan'))

    def test_gripper_expectation_and_terminal_mask(self):
        learner = DSRLLearner(self.flow, settings())
        with torch.no_grad():
            for p in self.flow.gripper.parameters():
                p.zero_()
            self.flow.gripper[-1].bias.copy_(tensor([.2, .3, .5]).log())
        s = torch.zeros(5, 4, dtype=DTYPE)
        z = torch.zeros(5, 6, dtype=DTYPE)
        value = learner.decoded_value(AffineCritic(), s, z)
        expected = self.flow.decode(s, z, steps=8)[:, 0]+2*(.5-.2)
        torch.testing.assert_close(value, expected)
        self.assertFalse(value.requires_grad)
        learner.decoded_value = lambda *args: (_ for _ in ()).throw(AssertionError('terminal decode'))
        rewards = tensor([4., -1.])
        torch.testing.assert_close(learner.td_target(rewards, torch.full((2, 4), float('nan')),
                                                    torch.ones(2)), rewards)

    def test_td_target_nonterminal_and_update_isolation(self):
        learner = DSRLLearner(self.flow, settings())
        original = learner.decoded_value
        learner.decoded_value = lambda critic, s, z: torch.full((len(s),), 10., dtype=DTYPE)
        target = learner.td_target(tensor([4., -1.]), torch.zeros(2, 4, dtype=DTYPE), tensor([1, 0]))
        torch.testing.assert_close(target, tensor([4., 8.9]))
        learner.decoded_value = original
        replay = ActionReplay(4)
        for i in range(20):
            replay.add(np.zeros(4), np.zeros(7), i % 2, np.zeros(4), True)
        before = fingerprint(self.flow)
        actor_before = fingerprint(learner.actor)
        learner.update(replay.batch(16), train_actor=False)
        self.assertEqual(actor_before, fingerprint(learner.actor))
        learner.update(replay.batch(16))
        self.assertNotEqual(actor_before, fingerprint(learner.actor))
        self.assertEqual(before, fingerprint(self.flow))
        self.assertTrue(all(p.grad is None for p in self.flow.parameters()))
        clone = DSRLLearner(self.flow, settings())
        clone.load_state_dict(learner.state_dict())
        for name in ('actor', 'action_critic', 'noise_critic', 'target'):
            self.assertEqual(fingerprint(getattr(clone, name)), fingerprint(getattr(learner, name)))

    def test_noise_distillation_steers_two_modes_from_action_only_data(self):
        flow = freeze(TwoModeFlow())
        learner = DSRLLearner(flow, settings())
        replay = ActionReplay(2, capacity=1024, seed=4)
        states = torch.ones(1024, 2, dtype=DTYPE)
        states[:512, 0] = -1
        actions = flow.sample(states, steps=8)
        # Reward the left mode for one state and the right mode for the other.
        rewards = (actions[:, 0]*states[:, 0] > 0).to(DTYPE)
        for s, a, r in zip(states, actions, rewards):
            replay.add(s.numpy(), a.numpy(), r.item(), s.numpy(), True)  # no noise labels
        before = fingerprint(flow)
        for i in range(650):
            learner.update(replay.batch(64), train_actor=i >= 150)
        with torch.no_grad():
            z = learner.actor.sample(state_features(flow, states),
                                    generator=torch.Generator().manual_seed(123))
            steered = flow.decode(states, z, steps=8)
            score = (steered[:, 0]*states[:, 0] > 0).double().mean().item()
        self.assertGreater(score, .8)
        self.assertGreater(score-rewards.mean().item(), .25)
        self.assertEqual(before, fingerprint(flow))
        print(f'\nSynthetic two-mode success: BC={rewards.mean().item():.3f}, steered={score:.3f}')

    def test_runner_roundtrip_offline_replay_and_evaluation_rng(self):
        d = dict(settings(), seed=0, eval_seed=40000, batch_size=8, replay_capacity=100,
                 warmup_episodes=1, critic_warmup_updates=1, pretrain_updates=1,
                 online_steps=4, utd=3, eval_episodes=2, eval_interval=3, log_stride=1,
                 actor_mean_bound=1.5, train_flow_steps=4)
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
            self.assertEqual(report['training_env_steps'], 7)
            self.assertEqual(report['updates'], 14)
            self.assertGreater(report['mean_update_seconds'], 0)
            warmup = json.loads((root/'online/warmup.json').read_text())
            self.assertEqual(warmup['successes'], 0)
            self.assertEqual(report['retained_replay_transitions'], 7)
            self.assertEqual(report['partial_online_episode_steps'], 1)
            loaded_cfg, flow, bc, learner = load_checkpoint(root/'online/dsrl.pt', flow_path)
            self.assertEqual(learner.actor.mean_bound, 1.5)
            self.assertEqual(learner.actor.std_max, 1.)
            # v1 weights must still mean raw Gaussian means and std capped at e.
            legacy = DSRLLearner(flow, settings(), legacy_actor=True)
            with torch.no_grad():
                legacy.actor.net[-1].bias.fill_(2.)
            ck = torch.load(root/'online/dsrl.pt', weights_only=False)
            ck.update(schema='flowq.dsrl.v1', learner=legacy.state_dict())
            torch.save(ck, root/'legacy.pt')
            _, _, _, old = load_checkpoint(root/'legacy.pt', flow_path)
            x = torch.zeros(2, 4, dtype=DTYPE)
            torch.testing.assert_close(old.actor.distribution(x).mean, legacy.actor.distribution(x).mean)
            torch.testing.assert_close(old.actor.distribution(x).scale, legacy.actor.distribution(x).scale)
            np_before, torch_before = np.random.get_state(), torch.random.get_rng_state()
            first = evaluate_pair(loaded_cfg, flow, learner, {}, TinyTask)
            second = evaluate_pair(loaded_cfg, flow, learner, {}, TinyTask)
            self.assertEqual(first, second)
            self.assertTrue(np.array_equal(np_before[1], np.random.get_state()[1]))
            self.assertTrue(torch.equal(torch_before, torch.random.get_rng_state()))
            # Prior-initialized actor is exactly the BC baseline under paired seeds.
            fresh = DSRLLearner(flow, settings())
            paired = evaluate_pair(loaded_cfg, flow, fresh, {}, TinyTask)
            self.assertEqual(paired['bc'], paired['dsrl'])
            # Drop optional noise fields: offline training does not require them.
            with np.load(root/'online/replay.npz') as data:
                np.savez(root/'action_only.npz', **{k: data[k] for k in data.files
                                                 if k not in ('noise', 'noise_known')})
            offline_cfg = copy.deepcopy(cfg)
            offline_cfg['dsrl'].update(warmup_episodes=0, online_steps=0)
            off = train(offline_cfg, flow_path, root/'offline', root/'action_only.npz', TinyTask)
            self.assertEqual(off['training_env_steps'], 0)
            self.assertEqual(off['imported_transitions'], 7)
            self.assertEqual(off['updates'], 2)
            with self.assertRaises(FileExistsError):
                train(cfg, flow_path, root/'online', env_factory=TinyTask)


if __name__ == '__main__':
    unittest.main()
