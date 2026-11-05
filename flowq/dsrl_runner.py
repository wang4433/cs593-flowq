"""Train/evaluate DSRL-inspired steering without modifying the BC checkpoint.

Run ``python -m flowq.dsrl_runner --help``. Float64 everywhere; ``--device auto`` (default)
runs the networks on CUDA when available, otherwise CPU. Environment stepping and
all seeded noise generators stay on the CPU, so paired BC/DSRL evaluations keep
identical noise streams on either device.
Evaluation uses separate environments, seeds and RNGs and never enters replay.
"""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from .data import KEYS, sha256
from .dsrl import ActionReplay, DSRLLearner, state_features
from .env import Task
from .flow import fingerprint, tensor
from .runner import load_flow, seed, write_json


def load_config(path):
    path = Path(path)
    settings = json.loads(path.read_text())
    base = (path.parent/settings['bc_config']).resolve()
    cfg = json.loads(base.read_text())
    cfg['dsrl'] = settings['dsrl']
    validate_config(cfg)
    return cfg


def validate_config(cfg):
    d = cfg['dsrl']
    if cfg['task'] not in ('lift', 'can', 'square'):
        raise ValueError('Unsupported task')
    positive = ('hidden', 'batch_size', 'flow_steps', 'replay_capacity', 'eval_episodes',
                'eval_interval', 'log_stride', 'utd')
    nonnegative = ('warmup_episodes', 'online_steps', 'critic_warmup_updates', 'pretrain_updates',
                   'seed', 'eval_seed')
    for key in positive + nonnegative:
        if type(d[key]) is not int or d[key] < (1 if key in positive else 0):
            raise ValueError(f'Invalid DSRL setting: {key}')
    train_steps = d.get('train_flow_steps', d['flow_steps'])
    if type(train_steps) is not int or train_steps < 1:
        raise ValueError('train_flow_steps must be a positive integer')
    bound = d.get('actor_mean_bound', 2.)
    if not math.isfinite(bound) or bound <= 0:
        raise ValueError('actor_mean_bound must be finite and positive')
    for key in ('lr', 'grad_clip', 'kl_beta', 'prior_fraction', 'tau'):
        if not math.isfinite(d[key]):
            raise ValueError(f'Nonfinite DSRL setting: {key}')
    if (d['lr'] <= 0 or d['grad_clip'] <= 0 or d['kl_beta'] < 0
            or not 0 <= d['prior_fraction'] <= 1 or not 0 < d['tau'] <= 1):
        raise ValueError('Invalid DSRL optimizer or distribution settings')
    if not 0 < cfg['gamma'] <= 1 or type(cfg['horizon']) is not int or cfg['horizon'] < 1:
        raise ValueError('Invalid reward horizon or discount')
    for key in ('reward_scale', 'reward_bias'):
        if not math.isfinite(cfg[key]):
            raise ValueError('Nonfinite reward convention')


def resolve_device(name='auto'):
    if name == 'auto':
        name = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(name)
    if device.type == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA requested but torch.cuda.is_available() is False')
        torch.zeros(1, dtype=torch.float64, device=device)  # fail fast on unsupported GPUs
    return device


def device_info(device):
    info = dict(device=str(device), torch=torch.__version__)
    if device.type == 'cuda':
        info.update(gpu=torch.cuda.get_device_name(device), cuda=torch.version.cuda)
    return info


def learner_config(cfg):
    return dict(cfg['dsrl'], gamma=cfg['gamma'])


def replay_contract(cfg, flow):
    return dict(task=cfg['task'], horizon=cfg['horizon'], versions=cfg['versions'],
                reward_scale=cfg['reward_scale'], reward_bias=cfg['reward_bias'],
                observation_keys=KEYS, physical_obs_dim=flow.obs_dim,
                time_feature='1-t/horizon', terminal='success_or_horizon_or_env_done',
                actions='six_arm_commands_and_sign_gripper')


@torch.no_grad()
def act(flow, learner, state, generator, steps, baseline=False, mean_noise=False):
    s = tensor(state)[None].to(flow.mean.device)
    if baseline:
        z = torch.randn(1, 6, dtype=s.dtype, generator=generator).to(s.device)
    else:
        z = learner.actor.sample(state_features(flow, s), generator=generator,
                                 deterministic=mean_noise)
    a = flow.action_from_noise(s, z, steps=steps, generator=generator)
    return a[0].cpu().numpy(), z[0].cpu().numpy()


def evaluate_pair(cfg, flow, learner, metadata, env_factory=Task, mean_noise=False):
    """Paired reset seeds; stochastic learned policy is the default estimand."""
    d = cfg['dsrl']
    modules = (flow, learner.actor, learner.action_critic, learner.noise_critic, learner.target)
    before = [fingerprint(m) for m in modules]
    modes = [m.training for m in modules]
    updates = learner.updates
    np_state = np.random.get_state()
    result = {}
    try:
        for m in modules:
            m.eval()
        with torch.random.fork_rng():
            for method in ('bc', 'dsrl'):
                env = env_factory(cfg, metadata)
                episodes = []
                try:
                    for ep in range(d['eval_episodes']):
                        reset_seed = d['eval_seed']+ep
                        g = torch.Generator().manual_seed(reset_seed+100000)
                        s = env.reset(reset_seed)
                        initial = hashlib.sha256(s.tobytes()).hexdigest()
                        done, reward_sum, discounted, n = False, 0., 0., 0
                        while not done:
                            a, _ = act(flow, learner, s, g, d['flow_steps'],
                                       baseline=method == 'bc', mean_noise=mean_noise)
                            s, r, done, info = env.step(a)
                            discounted += cfg['gamma']**n*r
                            reward_sum += r; n += 1
                        episodes.append(dict(seed=reset_seed, success=bool(info['success']),
                            steps=n, return_sum=reward_sum, discounted_return=discounted,
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
               for a, b in zip(result['bc']['episodes'], result['dsrl']['episodes']))
    result.update(evaluation_did_not_train=True,
                  dsrl_sampling='mean_noise' if mean_noise else 'stochastic',
                  evaluation_steps=sum(result[m]['environment_steps'] for m in ('bc', 'dsrl')))
    return result


def save_checkpoint(path, cfg, flow_path, flow, learner, training_steps):
    torch.save(dict(schema='flowq.dsrl.v2', config=cfg, flow_sha256=sha256(flow_path),
                    flow_fingerprint=fingerprint(flow), learner=learner.state_dict(),
                    training_env_steps=training_steps), path)


def load_checkpoint(path, flow_path, device='cpu'):
    ck = torch.load(path, map_location='cpu', weights_only=False)
    if ck.get('schema') not in ('flowq.dsrl.v1', 'flowq.dsrl.v2') or ck['flow_sha256'] != sha256(flow_path):
        raise ValueError('DSRL checkpoint schema or frozen flow identity mismatch')
    cfg = ck['config']
    validate_config(cfg)
    flow, bc = load_flow(flow_path, cfg)
    flow.to(device)
    if ck['flow_fingerprint'] != fingerprint(flow):
        raise ValueError('Frozen model fingerprint mismatch')
    # A tanh mean applied to old weights would silently change the old policy.
    learner = DSRLLearner(flow, learner_config(cfg), legacy_actor=ck['schema'] == 'flowq.dsrl.v1')
    learner.load_state_dict(ck['learner'])
    return cfg, flow, bc, learner


def train(cfg, flow_path, out, replay_path=None, env_factory=Task, device='cpu'):
    validate_config(cfg)
    d = cfg['dsrl']
    if not (d['warmup_episodes'] or replay_path):
        raise ValueError('Provide warmup episodes or a training replay file')
    if not (d['critic_warmup_updates'] or d['pretrain_updates'] or d['online_steps']):
        raise ValueError('No training updates requested')
    out = Path(out)
    if out.exists():
        raise FileExistsError(out)
    seed(d['seed'])
    device = torch.device(device)
    flow, bc = load_flow(flow_path, cfg)
    flow.to(device)
    initial_flow = fingerprint(flow)
    learner = DSRLLearner(flow, learner_config(cfg))
    replay = ActionReplay(flow.obs_dim+1, d['replay_capacity'], d['seed']+1)
    contract = replay_contract(cfg, flow)
    imported = replay.load(replay_path, contract) if replay_path else 0
    out.mkdir(parents=True, exist_ok=False)
    write_json(out/'config.json', cfg)
    write_json(out/'device.json', device_info(device))
    print(json.dumps(dict(stage='start', **device_info(device))), flush=True)
    start = time.perf_counter()
    noise_rng = torch.Generator().manual_seed(d['seed']+100)
    warmup_steps, online_steps, evaluation_steps = 0, 0, 0
    episodes, evaluations = [], []
    update_seconds = 0.
    env = None
    try:
        if d['warmup_episodes'] or d['online_steps']:
            env = env_factory(cfg, bc['manifest']['env'])
        for ep in range(d['warmup_episodes']):
            s = env.reset(d['seed']+1000+ep)
            done, n, ret = False, 0, 0.
            while not done:
                a, z = act(flow, learner, s, noise_rng, d['flow_steps'], baseline=True)
                ns, r, done, info = env.step(a)
                replay.add(s, a, r, ns, done, noise=z)
                s = ns; n += 1; ret += r; warmup_steps += 1
            row = dict(stage='warmup', episode=ep, steps=n, success=bool(info['success']), return_sum=ret)
            episodes.append(row)
            print(json.dumps(row), flush=True)

        warmup = [e for e in episodes if e['stage'] == 'warmup']
        write_json(out/'warmup.json', dict(episodes=len(warmup),
            successes=sum(e['success'] for e in warmup), environment_steps=warmup_steps,
            retained_positive_reward_transitions=sum(row[2] > 0 for row in replay.rows),
            imported_transitions=imported))

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

            for _ in range(d['critic_warmup_updates']):
                update(False, 'critic_warmup')
            for _ in range(d['pretrain_updates']):
                update(True, 'pretrain')

            ep, n, ret = 0, 0, 0.
            if d['online_steps']:
                s = env.reset(d['seed']+200000)
            for step in range(d['online_steps']):
                a, z = act(flow, learner, s, noise_rng, d['flow_steps'])
                ns, r, done, info = env.step(a)
                replay.add(s, a, r, ns, done, noise=z)
                s = ns; online_steps += 1; n += 1; ret += r
                for _ in range(d['utd']):
                    update(True, 'online')
                if done:
                    row = dict(stage='online', episode=ep, steps=n, success=bool(info['success']),
                               return_sum=ret, training_env_steps=warmup_steps+online_steps)
                    episodes.append(row); print(json.dumps(row), flush=True)
                    ep += 1; n, ret = 0, 0.
                    if step+1 < d['online_steps']:
                        s = env.reset(d['seed']+200000+ep)
                if online_steps % d['eval_interval'] == 0 or step+1 == d['online_steps']:
                    ev = evaluate_pair(cfg, flow, learner, bc['manifest']['env'], env_factory)
                    ev['training_env_steps'] = warmup_steps+online_steps
                    evaluations.append(ev); evaluation_steps += ev['evaluation_steps']
                    write_json(out/'evaluations.json', evaluations)
                    save_checkpoint(out/'dsrl.pt', cfg, flow_path, flow, learner, warmup_steps+online_steps)
            if not d['online_steps']:
                ev = evaluate_pair(cfg, flow, learner, bc['manifest']['env'], env_factory)
                ev['training_env_steps'] = warmup_steps
                evaluations.append(ev); evaluation_steps += ev['evaluation_steps']
                write_json(out/'evaluations.json', evaluations)
    finally:
        if env is not None:
            env.close()
    assert fingerprint(flow) == initial_flow
    assert all(not p.requires_grad and p.grad is None for p in flow.parameters())
    assert learner.updates == d['critic_warmup_updates']+d['pretrain_updates']+online_steps*d['utd']
    assert len(replay) == min(imported+warmup_steps+online_steps, d['replay_capacity'])
    save_checkpoint(out/'dsrl.pt', cfg, flow_path, flow, learner, warmup_steps+online_steps)
    replay.save(out/'replay.npz', contract)
    report = dict(algorithm='DSRL-inspired noise-critic distillation',
        flow_sha256=sha256(flow_path), frozen_unchanged=True, config=cfg,
        imported_transitions=imported, retained_replay_transitions=len(replay),
        replay_source_sha256=sha256(replay_path) if replay_path else None,
        warmup_env_steps=warmup_steps, online_env_steps=online_steps,
        training_env_steps=warmup_steps+online_steps, evaluation_env_steps=evaluation_steps,
        all_simulator_steps=warmup_steps+online_steps+evaluation_steps,
        partial_online_episode_steps=n, updates=learner.updates,
        learner_update_seconds=update_seconds,
        mean_update_seconds=update_seconds/max(learner.updates, 1),
        compute=device_info(device),
        episodes=episodes, evaluations=evaluations, seconds=time.perf_counter()-start)
    write_json(out/'summary.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    train_p = sub.add_parser('train', help='BC warmup and/or replay, then online latent RL')
    train_p.add_argument('--config', required=True)
    train_p.add_argument('--flow', required=True)
    train_p.add_argument('--out', required=True)
    train_p.add_argument('--replay', help='Training replay.npz; action transitions need no noise labels')
    integer_overrides = ('warmup-episodes', 'online-steps', 'critic-warmup-updates',
                         'pretrain-updates', 'eval-episodes', 'utd', 'train-flow-steps', 'seed')
    for name in integer_overrides:
        train_p.add_argument('--'+name, type=int)
    train_p.add_argument('--kl-beta', type=float, help='Prior KL weight; suggested sweep: 1, 3, 10')
    train_p.add_argument('--actor-mean-bound', type=float, help='Mean = bound*tanh(raw_mean); default 2')
    train_p.add_argument('--eval-interval', type=int, help='Online steps between evaluations')
    train_p.add_argument('--log-stride', type=int, help='Updates between updates.jsonl rows')
    train_p.add_argument('--device', default='auto', help="'auto' (CUDA if available), 'cpu', 'cuda' or 'cuda:N'")
    eval_p = sub.add_parser('eval', help='Compare a saved latent actor against its exact BC checkpoint')
    eval_p.add_argument('--checkpoint', required=True)
    eval_p.add_argument('--flow', required=True)
    eval_p.add_argument('--out', required=True)
    eval_p.add_argument('--episodes', type=int)
    eval_p.add_argument('--seed', type=int)
    eval_p.add_argument('--mean-noise', action='store_true', help='Separate diagnostic; not stochastic policy evaluation')
    eval_p.add_argument('--device', default='auto', help="'auto' (CUDA if available), 'cpu', 'cuda' or 'cuda:N'")
    args = parser.parse_args()
    device = resolve_device(args.device)
    if args.command == 'train':
        cfg = load_config(args.config)
        for name in (tuple(n.replace('-', '_') for n in integer_overrides)
                     + ('kl_beta', 'actor_mean_bound', 'eval_interval', 'log_stride')):
            value = getattr(args, name)
            if value is not None:
                cfg['dsrl'][name] = value
        train(cfg, args.flow, args.out, args.replay, device=device)
    else:
        if Path(args.out).exists():
            raise FileExistsError(args.out)
        cfg, flow, bc, learner = load_checkpoint(args.checkpoint, args.flow, device)
        cfg = copy.deepcopy(cfg)
        if args.episodes is not None:
            cfg['dsrl']['eval_episodes'] = args.episodes
        if args.seed is not None:
            cfg['dsrl']['eval_seed'] = args.seed
        validate_config(cfg); seed(cfg['dsrl']['seed'])
        report = evaluate_pair(cfg, flow, learner, bc['manifest']['env'], mean_noise=args.mean_noise)
        report.update(compute=device_info(device), config=cfg, checkpoint_sha256=sha256(args.checkpoint), flow_sha256=sha256(args.flow))
        write_json(args.out, report)


if __name__ == '__main__':
    main()
