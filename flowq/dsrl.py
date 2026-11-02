"""DSRL-inspired action-to-noise critic distillation for a frozen hybrid flow.

QA learns task returns from (s,a,r,s',done); QZ learns QA(s, decode(s,z)).
No inverse map, recorded latent, flow likelihood, or ODE gradient is needed.
The unchanged stochastic gripper is marginalized exactly in critic queries.

This is an adaptation of DSRL-NA, not an exact reproduction: Gaussian latent
actor, optional analytic KL to N(0,I), twin reward critics, and prior/actor
mixture distillation. See docs/DSRL.md for objectives and conventions.
"""
import copy
import json
import math

import numpy as np
import torch
from torch import nn

from .flow import DTYPE, finite, mlp, tensor
from .rl import Critic


def state_features(flow, states):
    if states.ndim != 2 or states.shape[-1] != flow.obs_dim + 1:
        raise ValueError('DSRL states must contain physical observations plus time remaining')
    return torch.cat([(states[:, :flow.obs_dim]-flow.mean)/flow.std, states[:, -1:]], -1)


class LatentActor(nn.Module):
    """Gaussian with bounded mean/std; samples themselves remain unbounded.

    mean_bound=None is reserved for loading legacy v1 checkpoints.
    """
    def __init__(self, obs_dim, latent_dim=6, hidden=128, mean_bound=2., std_max=1.):
        super().__init__()
        if mean_bound is not None and (not math.isfinite(mean_bound) or mean_bound <= 0):
            raise ValueError('Actor mean bound must be finite and positive')
        if not math.isfinite(std_max) or std_max < 1:
            raise ValueError('Std cap must include the initial unit-Gaussian prior')
        self.mean_bound, self.std_max = mean_bound, std_max
        self.latent_dim = latent_dim
        self.net = mlp(obs_dim, 2*latent_dim, hidden)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def distribution(self, states):
        mean, log_std = self.net(states).chunk(2, -1)
        if self.mean_bound is not None:
            mean = self.mean_bound*torch.tanh(mean)
        return torch.distributions.Normal(mean, log_std.clamp(-5, math.log(self.std_max)).exp())

    def sample(self, states, generator=None, deterministic=False):
        dist = self.distribution(states)
        if deterministic:
            return dist.mean
        eps = torch.randn(dist.mean.shape, dtype=states.dtype, device=states.device,
                          generator=generator)
        return dist.mean + dist.scale*eps

    def kl(self, states):
        dist = self.distribution(states)
        return .5*(dist.mean.square()+dist.scale.square()-1-2*dist.scale.log()).sum(-1)


class ActionReplay:
    """Raw action transitions; noise is optional diagnostic data, never a label.

    Unlike density queries, action critics can accept demonstration commands at
    +/-1. Unbounded noise is stored independently of bounded executed commands.
    """
    def __init__(self, obs_dim, capacity=100000, seed=0):
        if capacity < 1:
            raise ValueError('Replay capacity must be positive')
        self.obs_dim, self.capacity = obs_dim, capacity
        self.rows, self.noises, self.pos = [], [], 0
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.rows)

    def add(self, s, a, r, ns, done, noise=None):
        s, a, ns = [np.asarray(v, dtype=np.float64) for v in (s, a, ns)]
        if s.shape != (self.obs_dim,) or ns.shape != s.shape or a.shape != (7,):
            raise ValueError('Invalid transition dimensions')
        for name, value in [('state', s), ('action', a), ('reward', r), ('next state', ns)]:
            finite(value, name)
        if np.any(np.abs(a[:6]) > 1) or a[-1] not in (-1, 0, 1):
            raise ValueError('Invalid executed hybrid command')
        if done not in (0, 1, False, True):
            raise ValueError('Terminal flag must be boolean')
        if noise is not None:
            noise = np.asarray(noise, dtype=np.float64)
            if noise.shape != (6,):
                raise ValueError('Expected six-dimensional arm noise')
            finite(noise, 'noise')
            noise = noise.copy()
        row = (s.copy(), a.copy(), float(r), ns.copy(), float(done))
        if len(self) < self.capacity:
            self.rows.append(row); self.noises.append(noise)
        else:
            self.rows[self.pos], self.noises[self.pos] = row, noise
        self.pos = (self.pos+1) % self.capacity

    def batch(self, size):
        if not self.rows or size < 1:
            raise ValueError('Cannot sample empty replay or a nonpositive batch')
        indices = self.rng.integers(len(self), size=size)
        return tuple(tensor(np.stack(x)) for x in zip(*(self.rows[i] for i in indices)))

    def save(self, path, contract):
        if not self.rows:
            raise ValueError('Cannot export empty replay')
        states, actions, rewards, next_states, terminals = map(np.stack, zip(*self.rows))
        np.savez_compressed(path, states=states, actions=actions, rewards=rewards,
                            next_states=next_states, terminals=terminals,
                            noise=np.stack([np.zeros(6) if z is None else z for z in self.noises]),
                            noise_known=np.array([z is not None for z in self.noises]),
                            metadata=json.dumps(dict(schema='flowq.dsrl.replay.v1',
                                purpose='training', contract=contract)))

    def load(self, path, contract):
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(str(data['metadata']))
            if (meta.get('schema') != 'flowq.dsrl.replay.v1' or meta.get('purpose') != 'training'
                    or meta.get('contract') != contract):
                raise ValueError('Replay schema, training provenance, or environment contract mismatch')
            keys = ('states', 'actions', 'rewards', 'next_states', 'terminals')
            arrays = [data[k] for k in keys]
            n = len(arrays[0])
            if not n or any(len(v) != n for v in arrays):
                raise ValueError('Inconsistent or empty replay arrays')
            if any(k in data for k in ('noise', 'noise_known')):
                if ('noise' not in data or 'noise_known' not in data
                        or data['noise'].shape != (n, 6) or data['noise_known'].shape != (n,)):
                    raise ValueError('Inconsistent optional noise arrays')
            for i, row in enumerate(zip(*arrays)):
                z = data['noise'][i] if 'noise_known' in data and data['noise_known'][i] else None
                self.add(*row, noise=z)
        return n


class DSRLLearner:
    def __init__(self, flow, cfg, legacy_actor=False):
        if (not flow.hybrid or flow.continuous_dim != 6 or flow.action_dim != 7
                or flow.training or any(p.requires_grad for p in flow.parameters())):
            raise ValueError('DSRL requires a frozen, evaluation-mode 6D arm + gripper flow')
        self.flow, self.cfg = flow, cfg
        n = flow.obs_dim+1
        self.actor = LatentActor(n, 6, cfg['hidden'],
                                mean_bound=None if legacy_actor else cfg.get('actor_mean_bound', 2.),
                                std_max=math.e if legacy_actor else 1.)
        self.action_critic = Critic(n, 7, cfg['hidden'])
        self.noise_critic = Critic(n, 6, cfg['hidden'])
        self.target = copy.deepcopy(self.action_critic).requires_grad_(False)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg['lr'])
        self.action_opt = torch.optim.Adam(self.action_critic.parameters(), lr=cfg['lr'])
        self.noise_opt = torch.optim.Adam(self.noise_critic.parameters(), lr=cfg['lr'])
        self.updates = 0

    @torch.no_grad()
    def decoded_value(self, critic, states, z):
        """E_gripper[min_i QA_i(s, (decode(s,z), gripper))]."""
        arm = self.flow.decode(states, z, steps=self.cfg.get('train_flow_steps', self.cfg['flow_steps']))
        probs = self.flow.gripper_dist(states).probs
        grip = torch.arange(-1, 2, dtype=states.dtype, device=states.device)
        actions = torch.cat([arm[:, None, :].expand(-1, 3, -1),
                             grip.expand(len(states), 3)[..., None]], -1)
        x = state_features(self.flow, states)
        values = critic(x[:, None, :].expand(-1, 3, -1).reshape(-1, x.shape[-1]),
                        actions.reshape(-1, 7)).min(0).values.reshape(-1, 3)
        return (probs*values).sum(-1)

    @torch.no_grad()
    def td_target(self, rewards, next_states, terminals):
        # Do not decode terminal states: there is no next action at success or
        # the finite horizon, and terminal observations can be out of distribution.
        target = rewards.clone()
        active = terminals == 0
        if active.any():
            ns = next_states[active]
            z = self.actor.sample(state_features(self.flow, ns))
            target[active] += self.cfg['gamma']*self.decoded_value(self.target, ns, z)
        return target

    def _step(self, loss, optimizer, module, label):
        finite(loss, label)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(module.parameters(), self.cfg['grad_clip'],
                                       error_if_nonfinite=True)
        optimizer.step()

    def update(self, batch, train_actor=True):
        states, actions, rewards, next_states, terminals = batch
        x = state_features(self.flow, states)
        target = self.td_target(rewards, next_states, terminals)
        action_loss = (self.action_critic(x, actions)-target).square().mean()
        self._step(action_loss, self.action_opt, self.action_critic, 'action critic loss')

        # Distill freshly decoded actions, not the logged action/noise pair.
        with torch.no_grad():
            z = torch.randn(len(states), 6, dtype=states.dtype, device=states.device)
            actor_z = self.actor.sample(x)
            prior = torch.rand(len(states), 1, device=states.device) < self.cfg['prior_fraction']
            z = torch.where(prior, z, actor_z)
            teacher = self.decoded_value(self.action_critic, states, z)
        noise_loss = (self.noise_critic(x, z)-teacher).square().mean()
        self._step(noise_loss, self.noise_opt, self.noise_critic, 'noise critic loss')

        actor_loss = None
        if train_actor:
            self.noise_critic.requires_grad_(False)
            try:
                z = self.actor.sample(x)
                actor_loss = (self.cfg['kl_beta']*self.actor.kl(x)
                              - self.noise_critic(x, z).min(0).values).mean()
                self._step(actor_loss, self.actor_opt, self.actor, 'latent actor loss')
            finally:
                self.noise_critic.requires_grad_(True)
        with torch.no_grad():
            for target_p, p in zip(self.target.parameters(), self.action_critic.parameters()):
                target_p.lerp_(p, self.cfg['tau'])
            kl = self.actor.kl(x).mean().item()
            dist = self.actor.distribution(x)
        self.updates += 1
        return dict(action_critic_loss=action_loss.item(), noise_critic_loss=noise_loss.item(),
                    actor_loss=None if actor_loss is None else actor_loss.item(),
                    latent_kl=kl, teacher_mean=teacher.mean().item(),
                    latent_mean_abs_max=dist.mean.abs().max().item(),
                    latent_std_min=dist.scale.min().item(), latent_std_max=dist.scale.max().item(),
                    td_target_mean=target.mean().item())

    def state_dict(self):
        return dict(actor=self.actor.state_dict(), action_critic=self.action_critic.state_dict(),
                    noise_critic=self.noise_critic.state_dict(), target=self.target.state_dict(),
                    actor_optimizer=self.actor_opt.state_dict(),
                    action_optimizer=self.action_opt.state_dict(),
                    noise_optimizer=self.noise_opt.state_dict(), updates=self.updates)

    def load_state_dict(self, state):
        for name in ('actor', 'action_critic', 'noise_critic', 'target'):
            getattr(self, name).load_state_dict(state[name])
        for name in ('actor', 'action', 'noise'):
            getattr(self, name+'_opt').load_state_dict(state[name+'_optimizer'])
        self.updates = state['updates']
