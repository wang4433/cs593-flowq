"""Chunk-level DSRL and dual RL on a frozen action-chunked Flow BC policy (flowq/chunk.py).

One decision = one chunk of H commands, executed open loop, then replan (execute = H, the
best setting in docs/CHUNK_BC.md). The noise policy outputs 6H latent dimensions, the
action critic scores the whole (H, 7) chunk, and the TD target sums the discounted rewards
of the n executed steps:  y = sum_{k<n} gamma^k r_k + gamma^n (1 - done) Q(x', A').
Everything else is the single-step learner (flowq/dsrl.py, flowq/dual.py) with these
chunk-level pieces swapped in. The gripper expectation is a Monte Carlo average over
``gripper_samples`` sequences from the frozen per-step gripper head.

With a fresh noise policy (= N(0, I)) and zero correction, every evaluation arm reproduces
flowq.chunk.evaluate exactly (same seeds, same noise and gripper draws, same machine type),
so existing BC evaluations of the same checkpoint are the paired baseline.

  python -m flowq.chunk_rl train --config configs/square_chunk_dual_small.json \
      --flow runs/chunk/h4_wide/flow_50000.pt --out runs/chunk_rl/dual_seed0
  python -m flowq.chunk_rl eval --checkpoint runs/chunk_rl/dual_seed0/chunk_rl.pt \
      --flow runs/chunk/h4_wide/flow_50000.pt --episodes 200 --seed 60000 --out runs/chunk_rl/dual_seed0/confirm.json
"""
import argparse
import copy
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .chunk import load as load_chunk_flow, load_chunks
from .data import sha256
from .dsrl import DSRLLearner, LatentActor, state_features
from .dsrl_runner import device_info, load_config, resolve_device, validate_config as validate_dsrl
from .dual import DualLearner
from .env import Task
from .flow import DTYPE, finite, fingerprint, generator_device, mlp, rk4, tensor
from .rl import Critic
from .runner import seed, write_json


def validate_config(cfg):
    validate_dsrl(cfg)
    d = cfg['dsrl']
    if type(d.get('gripper_samples', 8)) is not int or d.get('gripper_samples', 8) < 1:
        raise ValueError('gripper_samples must be a positive integer')
    if type(d.get('bc_ema', False)) is not bool:
        raise ValueError('bc_ema must be true or false')
    if 'density' in cfg and d['flow_steps'] != cfg['density']['steps']:
        # flowq.chunk.evaluate decodes with density.steps; matching it keeps BC paired.
        raise ValueError('flow_steps must equal density.steps to reproduce chunked BC evaluations')


def decode_chunk(flow, states, z, steps):
    """Arm commands (batch, 6H) from explicit noise through any chunk velocity field."""
    arm = torch.tanh(rk4(lambda t, x: flow.velocity(t, x, states), z, 0, 1, steps))
    if (arm.abs() >= 1).any():
        raise FloatingPointError('Floating-point tanh saturation')
    return arm


class ChunkResidualFlow(nn.Module):
    """Frozen chunked BC flow plus a bounded velocity correction (chunk ResidualFlow)."""
    def __init__(self, base, bound, hidden=None, layers=None, layer_norm=False):
        super().__init__()
        if not math.isfinite(bound) or bound <= 0:
            raise ValueError('Residual bound must be finite and positive')
        if base.training or any(p.requires_grad for p in base.parameters()):
            raise ValueError('ChunkResidualFlow requires a frozen, evaluation-mode base flow')
        self.base, self.bound = base, float(bound)
        self.obs_dim, self.chunk, self.dim = base.obs_dim, base.chunk, base.dim
        self.continuous_dim = base.dim  # name used by DualLearner.update_flow
        self.correction = mlp(self.obs_dim+self.dim+1, self.dim, hidden or base.hidden,
                              layers or base.layers, layer_norm).to(base.mean.device)
        nn.init.zeros_(self.correction[-1].weight)
        nn.init.zeros_(self.correction[-1].bias)
        self.eval()

    @property
    def mean(self):
        return self.base.mean

    @property
    def std(self):
        return self.base.std

    def frozen_parameters(self):
        return self.base.parameters()

    def gripper_logits(self, s):
        return self.base.gripper_logits(s)

    def velocity(self, t, x, s):
        time = torch.full_like(x[..., :1], float(t)) if not torch.is_tensor(t) else t
        g = self.correction(torch.cat([self.base.normalized(s), x, time], -1))
        return self.base.velocity(t, x, s)+self.bound*torch.tanh(g)

    @torch.no_grad()
    def base_latent(self, s, z, steps=16):
        return rk4(lambda t, x: self.base.velocity(t, x, s), z, 0, 1, steps)

    @torch.no_grad()
    def decode(self, s, z, steps=16):
        return decode_chunk(self, s, z, steps)

    def matching_loss(self, s, y1, weights=None):
        finite(y1, 'flow target')
        x0 = torch.randn_like(y1)
        t = torch.rand_like(y1[:, :1])
        err = (self.velocity(t, (1-t)*x0+t*y1, s)-(y1-x0)).square().mean(-1)
        return err.mean() if weights is None else (weights*err).sum()/weights.sum()


class ChunkDSRLLearner(DSRLLearner):
    """DSRL over whole chunks: 6H-dim noise, critics over flattened (H, 7) command chunks."""
    def __init__(self, flow, cfg):
        frozen = flow.frozen_parameters() if hasattr(flow, 'frozen_parameters') else flow.parameters()
        if flow.training or any(p.requires_grad for p in frozen):
            raise ValueError('Chunk DSRL requires a frozen, evaluation-mode chunked flow')
        self.flow, self.cfg, self.device, self.chunk = flow, cfg, flow.mean.device, flow.chunk
        n = flow.obs_dim+1
        net = dict(layers=cfg.get('layers', 2), layer_norm=cfg.get('layer_norm', False))
        self.actor = LatentActor(n, 6*self.chunk, cfg['hidden'], mean_bound=cfg.get('actor_mean_bound', 2.),
                                 std_min=cfg.get('actor_std_min', math.exp(-5)), **net).to(self.device)
        self.action_critic = Critic(n, 7*self.chunk, cfg['hidden'], **net).to(self.device)
        self.noise_critic = Critic(n, 6*self.chunk, cfg['hidden'], **net).to(self.device)
        self.target = copy.deepcopy(self.action_critic).requires_grad_(False)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg['lr'])
        self.action_opt = torch.optim.Adam(self.action_critic.parameters(), lr=cfg['lr'])
        self.noise_opt = torch.optim.Adam(self.noise_critic.parameters(), lr=cfg['lr'])
        self.updates = 0

    @torch.no_grad()
    def decoded_value(self, critic, states, z):
        arm = decode_chunk(self.flow, states, z, self.cfg.get('train_flow_steps', self.cfg['flow_steps']))
        return self.action_value(critic, states, arm)

    @torch.no_grad()
    def action_value(self, critic, states, arm):
        """Monte Carlo E_gripper[min_i Q_i(x, chunk)] over the frozen per-step gripper head."""
        b, h, k = len(states), self.chunk, self.cfg.get('gripper_samples', 8)
        probs = torch.softmax(self.flow.gripper_logits(states), -1).reshape(b*h, 3)
        grip = (torch.multinomial(probs, k, replacement=True)-1).to(arm.dtype).reshape(b, h, k).transpose(1, 2)
        chunks = torch.cat([arm.reshape(b, 1, h, 6).expand(-1, k, -1, -1), grip[..., None]], -1)
        x = state_features(self.flow, states).repeat_interleave(k, 0)
        return critic(x, chunks.reshape(b*k, 7*h)).min(0).values.reshape(b, k).mean(-1)


class ChunkDualLearner(DualLearner, ChunkDSRLLearner):
    """Dual RL over chunks: chunk DSRL steering plus the bounded, critic-weighted correction.

    update, update_flow and the state dicts are DualLearner's; decoding and the critic
    value are ChunkDSRLLearner's (method resolution order).
    """
    def __init__(self, base, cfg, demos=None):
        self.dual = cfg['dual']
        flow = ChunkResidualFlow(base, self.dual['residual_bound'], self.dual.get('correction_hidden'),
                                 self.dual.get('correction_layers'), self.dual.get('correction_layer_norm', False))
        ChunkDSRLLearner.__init__(self, flow, cfg)
        self.base = base
        self.demo_states = self.demo_targets = None
        if demos is not None:
            states, chunks = (torch.as_tensor(v, dtype=DTYPE, device=self.device) for v in demos)
            eps = cfg.get('boundary_epsilon', 1e-3)
            # Same inward smoothing and coordinates as chunked BC training (ChunkFlow.loss).
            self.demo_states = states
            self.demo_targets = torch.atanh(chunks[..., :6]*(1-eps)).flatten(1)
        self.init_correction_training()


def make_learner(base, cfg, demos=None):
    learner_cfg = dict(cfg['dsrl'], gamma=cfg['gamma'], boundary_epsilon=cfg.get('boundary_epsilon', 1e-3))
    if cfg['dsrl'].get('dual') is not None:
        return ChunkDualLearner(base, learner_cfg, demos)
    return ChunkDSRLLearner(base, learner_cfg)


class ChunkReplay:
    """(state, flattened (H, 7) chunk, discounted reward sum, next state, done, gamma^n)."""
    def __init__(self, obs_dim, chunk, capacity=200000, seed=0):
        if capacity < 1:
            raise ValueError('Replay capacity must be positive')
        self.obs_dim, self.chunk, self.capacity = obs_dim, chunk, capacity
        self.rows, self.pos = [], 0
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.rows)

    def add(self, s, chunk, reward, ns, done, discount):
        s, chunk, ns = [np.asarray(v, dtype=np.float64) for v in (s, chunk, ns)]
        if s.shape != (self.obs_dim,) or ns.shape != s.shape or chunk.shape != (self.chunk, 7):
            raise ValueError('Invalid chunk transition dimensions')
        for name, value in [('state', s), ('chunk', chunk), ('reward', reward), ('next state', ns)]:
            finite(value, name)
        if np.any(np.abs(chunk[:, :6]) > 1) or not np.isin(chunk[:, 6], (-1, 0, 1)).all():
            raise ValueError('Invalid command chunk')
        if done not in (0, 1, False, True) or not 0 < discount <= 1:
            raise ValueError('Invalid terminal flag or discount')
        row = (s.copy(), chunk.reshape(-1).copy(), float(reward), ns.copy(), float(done), float(discount))
        if len(self) < self.capacity:
            self.rows.append(row)
        else:
            self.rows[self.pos] = row
        self.pos = (self.pos+1) % self.capacity

    def batch(self, size):
        if not self.rows or size < 1:
            raise ValueError('Cannot sample empty replay or a nonpositive batch')
        indices = self.rng.integers(len(self), size=size)
        return tuple(tensor(np.stack(x)) for x in zip(*(self.rows[i] for i in indices)))


@torch.no_grad()
def act(flow, learner, state, generator, steps, baseline=False):
    """One (H, 7) command chunk. The draw order (noise, then gripper) matches
    ChunkFlow.sample, so baseline=True on the BC flow, or a fresh learner, reproduces
    flowq.chunk.evaluate exactly."""
    s = tensor(state)[None].to(flow.mean.device)
    if baseline:
        z = torch.randn(1, flow.dim, dtype=s.dtype, device=generator_device(generator, s.device),
                        generator=generator).to(s.device)
    else:
        z = learner.actor.sample(state_features(flow, s), generator=generator)
    arm = decode_chunk(flow, s, z, steps).reshape(flow.chunk, 6)
    probs = torch.softmax(flow.gripper_logits(s), -1).flatten(0, 1)
    grip = torch.multinomial(probs.to(generator_device(generator, probs.device)), 1, generator=generator)-1
    plan = torch.cat([arm, grip.to(arm.device, arm.dtype)], -1)
    return plan.cpu().numpy(), z[0].cpu().numpy()


def evaluate(cfg, base, learner, metadata, env_factory=Task):
    """Paired arms on the same seeds and base noise: bc (BC flow, prior noise), reshaped
    (corrected flow, prior noise; dual only), dsrl (learner's flow, learned noise)."""
    d, gamma = cfg['dsrl'], cfg['gamma']
    arms = [('bc', base, True)]
    if learner.flow is not base:
        arms.append(('reshaped', learner.flow, True))
    arms.append(('dsrl', learner.flow, False))
    modules = (base, learner.flow, learner.actor, learner.action_critic, learner.noise_critic, learner.target)
    before, modes = [fingerprint(m) for m in modules], [m.training for m in modules]
    updates, np_state, result = learner.updates, np.random.get_state(), {}
    try:
        for m in modules:
            m.eval()
        with torch.random.fork_rng():
            for method, flow, baseline in arms:
                env, episodes = env_factory(cfg, metadata), []
                try:
                    for ep in range(d['eval_episodes']):
                        reset_seed = d['eval_seed']+ep
                        g = torch.Generator().manual_seed(reset_seed+100000)
                        s = env.reset(reset_seed)
                        initial = hashlib.sha256(s.tobytes()).hexdigest()
                        done, reward_sum, discounted, n = False, 0., 0., 0
                        while not done:
                            plan, _ = act(flow, learner, s, g, d['flow_steps'], baseline)
                            for command in plan:
                                s, r, done, info = env.step(command)
                                discounted += gamma**n*r; reward_sum += r; n += 1
                                if done:
                                    break
                        episodes.append(dict(seed=reset_seed, success=bool(info['success']), steps=n,
                                             return_sum=reward_sum, discounted_return=discounted,
                                             initial_observation_sha256=initial))
                finally:
                    env.close()
                result[method] = dict(success_rate=float(np.mean([e['success'] for e in episodes])),
                    mean_discounted_return=float(np.mean([e['discounted_return'] for e in episodes])),
                    environment_steps=sum(e['steps'] for e in episodes), episodes=episodes)
    finally:
        np.random.set_state(np_state)
        for m, mode in zip(modules, modes):
            m.train(mode)
    assert before == [fingerprint(m) for m in modules] and updates == learner.updates
    assert all(a['initial_observation_sha256'] == b['initial_observation_sha256']
               for method, _, _ in arms[1:] for a, b in zip(result['bc']['episodes'], result[method]['episodes']))
    result.update(evaluation_did_not_train=True, chunk=base.chunk,
                  evaluation_steps=sum(result[m]['environment_steps'] for m, _, _ in arms))
    return result


def save_checkpoint(path, cfg, flow_path, base, learner, training_steps):
    torch.save(dict(schema='flowq.chunk_rl.v1', config=cfg, flow_sha256=sha256(flow_path),
                    flow_fingerprint=fingerprint(base), learner=learner.state_dict(),
                    training_env_steps=training_steps), path)


def load_checkpoint(path, flow_path, device='cpu'):
    ck = torch.load(path, map_location='cpu', weights_only=False)
    if ck.get('schema') != 'flowq.chunk_rl.v1' or ck['flow_sha256'] != sha256(flow_path):
        raise ValueError('Chunk RL checkpoint schema or frozen flow identity mismatch')
    cfg = ck['config']
    validate_config(cfg)
    base, bc = load_chunk_flow(flow_path, cfg, ema=cfg['dsrl'].get('bc_ema', False))
    base.to(device)
    if ck['flow_fingerprint'] != fingerprint(base):
        raise ValueError('Frozen model fingerprint mismatch')
    learner = make_learner(base, cfg)
    learner.load_state_dict(ck['learner'])
    return cfg, base, bc, learner


def train(cfg, flow_path, out, env_factory=Task, device='cpu', demos=None):
    validate_config(cfg)
    d, gamma = cfg['dsrl'], cfg['gamma']
    dual = d.get('dual')
    if not d['warmup_episodes']:
        raise ValueError('Chunk RL needs BC warmup episodes (replay import is not supported)')
    out = Path(out)
    if out.exists():
        raise FileExistsError(out)
    seed(d['seed'])
    device = torch.device(device)
    base, bc = load_chunk_flow(flow_path, cfg, ema=d.get('bc_ema', False))
    base.to(device)
    initial = fingerprint(base)
    if dual is not None and dual['demo_weight'] > 0 and demos is None:
        # Anchor on the BC training split only; validation demonstrations stay held out.
        demos = load_chunks(cfg, bc['manifest']['train_demos'], base.chunk)
    learner = make_learner(base, cfg, demos)
    replay = ChunkReplay(base.obs_dim+1, base.chunk, d['replay_capacity'], d['seed']+1)
    out.mkdir(parents=True, exist_ok=False)
    write_json(out/'config.json', cfg)
    write_json(out/'device.json', device_info(device))
    start, noise_rng = time.perf_counter(), torch.Generator().manual_seed(d['seed']+100)
    warmup_steps = online_steps = evaluation_steps = decisions = 0
    episodes, evaluations, update_seconds = [], [], 0.

    def run_chunk(env, s, flow, baseline):
        """Execute one chunk; store one transition. Returns next state, steps, raw reward, done, info."""
        plan, _ = act(flow, learner, s, noise_rng, d['flow_steps'], baseline)
        discounted, raw, n = 0., 0., 0
        for command in plan:
            ns, r, done, info = env.step(command)
            discounted += gamma**n*r; raw += r; n += 1
            if done:
                break
        replay.add(s, plan, discounted, ns, done, gamma**n)
        return ns, n, raw, done, info

    env = env_factory(cfg, bc['manifest']['env'])
    try:
        for ep in range(d['warmup_episodes']):
            s, done, n, ret = env.reset(d['seed']+1000+ep), False, 0, 0.
            while not done:
                s, k, r, done, info = run_chunk(env, s, base, True)
                n += k; ret += r
            warmup_steps += n
            episodes.append(dict(stage='warmup', episode=ep, steps=n, success=bool(info['success']), return_sum=ret))
            print(json.dumps(episodes[-1]), flush=True)
        warmup = [e for e in episodes if e['stage'] == 'warmup']
        write_json(out/'warmup.json', dict(episodes=len(warmup), successes=sum(e['success'] for e in warmup),
            environment_steps=warmup_steps, decisions=len(replay),
            retained_positive_reward_transitions=sum(row[2] > 0 for row in replay.rows)))

        with (out/'updates.jsonl').open('w') as log:
            def update(train_actor, stage):
                nonlocal update_seconds
                tick = time.perf_counter()
                metrics = learner.update(replay.batch(d['batch_size']), train_actor=train_actor)
                elapsed = time.perf_counter()-tick
                update_seconds += elapsed
                if learner.updates % d['log_stride'] == 0 or learner.updates == 1:
                    log.write(json.dumps(dict(stage=stage, update=learner.updates,
                        training_env_steps=warmup_steps+online_steps, update_seconds=elapsed,
                        **metrics), allow_nan=False)+'\n')
                    log.flush()

            def evaluate_now():
                nonlocal evaluation_steps
                ev = evaluate(cfg, base, learner, bc['manifest']['env'], env_factory)
                ev['training_env_steps'] = warmup_steps+online_steps
                ev['online_env_steps'] = online_steps
                evaluations.append(ev); evaluation_steps += ev['evaluation_steps']
                write_json(out/'evaluations.json', evaluations)
                save_checkpoint(out/'chunk_rl.pt', cfg, flow_path, base, learner, warmup_steps+online_steps)

            for _ in range(d['critic_warmup_updates']):
                update(False, 'critic_warmup')
            for _ in range(d['pretrain_updates']):
                update(True, 'pretrain')

            ep, n, ret = 0, 0, 0.
            next_eval = d['eval_interval']
            s = env.reset(d['seed']+200000)
            while online_steps < d['online_steps']:
                s, k, r, done, info = run_chunk(env, s, learner.flow, False)
                online_steps += k; decisions += 1; n += k; ret += r
                for _ in range(d['utd']):
                    update(True, 'online')
                if done:
                    episodes.append(dict(stage='online', episode=ep, steps=n, success=bool(info['success']),
                                         return_sum=ret, training_env_steps=warmup_steps+online_steps))
                    print(json.dumps(episodes[-1]), flush=True)
                    ep += 1; n, ret = 0, 0.
                    if online_steps < d['online_steps']:
                        s = env.reset(d['seed']+200000+ep)
                if online_steps >= next_eval or online_steps >= d['online_steps']:
                    evaluate_now()
                    while next_eval <= online_steps:
                        next_eval += d['eval_interval']
            if not d['online_steps']:
                evaluate_now()
    finally:
        env.close()
    assert fingerprint(base) == initial
    assert all(not p.requires_grad and p.grad is None for p in base.parameters())
    assert learner.updates == d['critic_warmup_updates']+d['pretrain_updates']+decisions*d['utd']
    save_checkpoint(out/'chunk_rl.pt', cfg, flow_path, base, learner, warmup_steps+online_steps)
    report = dict(algorithm=('Chunk dual RL' if dual is not None else 'Chunk DSRL'), chunk=base.chunk,
        flow_sha256=sha256(flow_path), bc_ema=d.get('bc_ema', False), frozen_unchanged=True, config=cfg,
        flow_updates=getattr(learner, 'flow_updates', 0), warmup_env_steps=warmup_steps,
        online_env_steps=online_steps, online_decisions=decisions, training_env_steps=warmup_steps+online_steps,
        evaluation_env_steps=evaluation_steps, updates=learner.updates, retained_replay_transitions=len(replay),
        learner_update_seconds=update_seconds, mean_update_seconds=update_seconds/max(learner.updates, 1),
        partial_online_episode_steps=n, compute=device_info(device), episodes=episodes,
        evaluations=evaluations, seconds=time.perf_counter()-start)
    write_json(out/'summary.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    t = sub.add_parser('train', help='BC warmup, then chunk DSRL or chunk dual RL')
    t.add_argument('--config', required=True)
    t.add_argument('--flow', required=True, help='Chunked flow checkpoint (flowq.chunk)')
    t.add_argument('--out', required=True)
    t.add_argument('--seed', type=int)
    t.add_argument('--device', default='auto')
    e = sub.add_parser('eval', help='Paired evaluation of a saved learner against its BC flow')
    e.add_argument('--checkpoint', required=True)
    e.add_argument('--flow', required=True)
    e.add_argument('--out', required=True)
    e.add_argument('--episodes', type=int)
    e.add_argument('--seed', type=int)
    e.add_argument('--device', default='auto')
    args = parser.parse_args()
    device = resolve_device(args.device)
    if args.command == 'train':
        cfg = load_config(args.config)
        if args.seed is not None:
            cfg['dsrl']['seed'] = args.seed
        train(cfg, args.flow, args.out, device=device)
    else:
        if Path(args.out).exists():
            raise FileExistsError(args.out)
        cfg, base, bc, learner = load_checkpoint(args.checkpoint, args.flow, device)
        cfg = copy.deepcopy(cfg)
        if args.episodes is not None:
            cfg['dsrl']['eval_episodes'] = args.episodes
        if args.seed is not None:
            cfg['dsrl']['eval_seed'] = args.seed
        validate_config(cfg); seed(cfg['dsrl']['seed'])
        report = evaluate(cfg, base, learner, bc['manifest']['env'])
        report.update(compute=device_info(device), config=cfg, checkpoint_sha256=sha256(args.checkpoint),
                      flow_sha256=sha256(args.flow))
        write_json(args.out, report)


if __name__ == '__main__':
    main()
