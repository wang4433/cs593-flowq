"""Action-chunked Flow BC: predict the next H commands, execute E of them, replan.

Standalone diagnostic for "does the BC policy need action chunking, and at what
horizon". It reuses the stage-1 recipe (same data, masks, state normalization,
atanh arm targets, bounded velocity, categorical gripper) and only widens the output
from one command to H. With H=1 the model and loss are the stage-1 Flow's.

Chunk targets never cross a demonstration boundary; the tail of each demonstration is
padded by repeating its last command (Diffusion Policy convention).

These checkpoints are NOT loadable by flowq.runner.load_flow: the DSRL/dual code
assumes a single 6-D arm command.

  python -m flowq.chunk train --config configs/square_bc_deep.json --chunk 8 \
      --updates 50000 --snapshots 10000 50000 --out runs/chunk/deep_h8
  python -m flowq.chunk eval --config configs/square_bc_deep.json \
      --flow runs/chunk/deep_h8/flow_50000.pt --episodes 50 --out runs/chunk/deep_h8/eval_50000
"""
import argparse
import copy
import json
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from torch import nn

from .data import KEYS, audit, sha256
from .flow import DTYPE, finite, fingerprint, freeze, mlp, rk4, tensor


def write_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False)+'\n')


def load_chunks(cfg, names, chunk):
    """States (N, obs) and command chunks (N, chunk, 7), one per demonstration step."""
    states, chunks = [], []
    with h5py.File(cfg['dataset'], 'r') as f:
        for name in names:
            d = f['data'][name]
            s = np.concatenate([d['obs'][k][:] for k in KEYS], -1)
            a = d['actions'][:]
            # Index t+k clamped to the last step: repeat the final command past the end.
            index = np.minimum(np.arange(len(a))[:, None]+np.arange(chunk)[None], len(a)-1)
            states.append(s); chunks.append(a[index])
    s, a = np.concatenate(states), np.concatenate(chunks)
    if not np.isfinite(s).all() or not np.isfinite(a).all() or np.max(np.abs(a)) > 1:
        raise ValueError('Invalid original observations/actions')
    return tensor(s), tensor(a)


class ChunkFlow(nn.Module):
    """Flow over chunk*6 arm commands plus an independent 3-way gripper head per step."""
    def __init__(self, obs_dim, chunk=1, hidden=128, layers=2, max_velocity=None,
                 gripper_hidden=None, gripper_layers=None):
        super().__init__()
        if type(chunk) is not int or chunk < 1:
            raise ValueError('chunk must be a positive integer')
        self.obs_dim, self.chunk, self.hidden, self.layers = obs_dim, chunk, hidden, layers
        self.max_velocity = max_velocity
        self.dim = 6*chunk
        self.net = mlp(obs_dim+self.dim+1, self.dim, hidden, layers)
        # The gripper head can be sized separately: it overfits long before the arm flow does.
        self.gripper_hidden, self.gripper_layers = gripper_hidden or hidden, gripper_layers or layers
        self.gripper = mlp(obs_dim, 3*chunk, self.gripper_hidden, self.gripper_layers)
        self.register_buffer('mean', torch.zeros(obs_dim, dtype=DTYPE))
        self.register_buffer('std', torch.ones(obs_dim, dtype=DTYPE))

    def normalized(self, s):
        return (s[..., :self.obs_dim]-self.mean)/self.std

    def gripper_logits(self, s):
        return self.gripper(self.normalized(s)).reshape(*s.shape[:-1], self.chunk, 3)

    def velocity(self, t, x, s):
        time = torch.full_like(x[..., :1], float(t)) if not torch.is_tensor(t) else t
        v = self.net(torch.cat([self.normalized(s), x, time], -1))
        return v if self.max_velocity is None else self.max_velocity*torch.tanh(v/self.max_velocity)

    def loss(self, s, actions, boundary_epsilon=1e-3):
        """Flow matching on atanh arm chunks + mean gripper cross-entropy (as Flow.loss)."""
        if actions.shape != (len(s), self.chunk, 7):
            raise ValueError('Expected (batch, chunk, 7) command chunks')
        x1 = torch.atanh(actions[..., :6]*(1-boundary_epsilon)).flatten(1)
        x0 = torch.randn_like(x1)
        t = torch.rand_like(x1[:, :1])
        fm = ((self.velocity(t, (1-t)*x0+t*x1, s)-(x1-x0))**2).mean()
        labels = actions[..., -1].sign().long()+1
        ce = nn.functional.cross_entropy(self.gripper_logits(s).flatten(0, 1), labels.flatten())
        return fm+ce, fm.detach(), ce.detach()

    @torch.no_grad()
    def sample(self, s, steps=16, generator=None):
        """One (chunk, 7) command chunk per state: decoded arm + sampled gripper."""
        z = torch.randn((len(s), self.dim), dtype=s.dtype, generator=generator)
        arm = torch.tanh(rk4(lambda t, x: self.velocity(t, x, s), z, 0, 1, steps))
        if (arm.abs() >= 1).any():
            raise FloatingPointError('Floating-point tanh saturation')
        probs = torch.softmax(self.gripper_logits(s), -1).flatten(0, 1)
        grip = (torch.multinomial(probs, 1, generator=generator)-1).reshape(len(s), self.chunk, 1)
        return torch.cat([arm.reshape(len(s), self.chunk, 6), grip.to(arm.dtype)], -1)


def save(flow, cfg, manifest, step, path, ema=None, options=None):
    torch.save(dict(flow=flow.state_dict(), ema=None if ema is None else ema.state_dict(),
                    obs_dim=flow.obs_dim, chunk=flow.chunk, hidden=flow.hidden, layers=flow.layers,
                    gripper_hidden=flow.gripper_hidden, gripper_layers=flow.gripper_layers,
                    task=cfg['task'], manifest=manifest, config=cfg, updates=step, options=options or {},
                    schema='flowq.chunk.v1', trained_on_original_demonstrations=True), path)


def load(path, cfg, ema=False):
    """Frozen chunked flow; ``ema=True`` loads the averaged weights instead of the raw ones."""
    ck = torch.load(path, map_location='cpu', weights_only=False)
    if ck.get('schema') != 'flowq.chunk.v1' or ck['task'] != cfg['task']:
        raise ValueError('Not a chunked flow checkpoint for this task')
    if ema and ck.get('ema') is None:
        raise ValueError('Checkpoint has no EMA weights')
    flow = ChunkFlow(ck['obs_dim'], ck['chunk'], ck['hidden'], ck['layers'],
                     ck['config'].get('flow_max_velocity'), ck.get('gripper_hidden'), ck.get('gripper_layers'))
    flow.load_state_dict(ck['ema'] if ema else ck['flow'])
    return freeze(flow), ck


def train(cfg, chunk, updates, snapshots, out, hidden=None, layers=None, gripper_hidden=None,
          gripper_layers=None, weight_decay=0., cosine=False, ema_decay=0.):
    """Stage-1 recipe by default. Optional: network size overrides, AdamW weight decay,
    cosine learning-rate decay to zero, and an EMA copy of the weights (saved alongside)."""
    out = Path(out)
    if weight_decay < 0 or not 0 <= ema_decay < 1:
        raise ValueError('Invalid weight decay or EMA decay')
    options = dict(weight_decay=weight_decay, cosine=cosine, ema_decay=ema_decay, total_updates=updates)
    if out.exists():
        raise FileExistsError(out)
    if any(n < 1 or n > updates for n in snapshots):
        raise ValueError('Snapshots must lie in [1, updates]')
    np.random.seed(cfg['seed']); torch.manual_seed(cfg['seed']); torch.set_num_threads(1)
    start = time.perf_counter()
    manifest = audit(cfg)
    s, a = load_chunks(cfg, manifest['train_demos'], chunk)
    vs, va = load_chunks(cfg, manifest['valid_demos'], chunk)
    flow = ChunkFlow(s.shape[-1], chunk, hidden or cfg['flow_hidden'], layers or cfg.get('flow_layers', 2),
                     cfg.get('flow_max_velocity'), gripper_hidden, gripper_layers)
    flow.mean.copy_(s.mean(0)); flow.std.copy_(s.std(0).clamp_min(cfg.get('obs_std_floor', 1e-3)))
    # weight_decay=0 without cosine is plain constant-rate Adam, i.e. the stage-1 optimizer.
    opt = (torch.optim.AdamW(flow.parameters(), lr=cfg['lr'], weight_decay=weight_decay) if weight_decay
           else torch.optim.Adam(flow.parameters(), lr=cfg['lr']))
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(opt, updates) if cosine else None
    ema = copy.deepcopy(flow).requires_grad_(False) if ema_decay else None
    out.mkdir(parents=True)
    logs, interval = [], cfg.get('bc_log_interval', 100)
    valid_rng = torch.Generator().manual_seed(cfg['seed']+1)
    for step in range(1, updates+1):
        i = torch.randint(len(s), (cfg['batch_size'],))
        loss, fm, ce = flow.loss(s[i], a[i], cfg['boundary_epsilon'])
        finite(loss, 'BC loss'); opt.zero_grad(); loss.backward(); opt.step()
        if schedule is not None:
            schedule.step()
        if ema is not None:
            with torch.no_grad():
                # Warm-up so early averages are not dominated by the random initialization.
                decay = min(ema_decay, (1+step)/(10+step))
                for average, p in zip(ema.parameters(), flow.parameters()):
                    average.lerp_(p, 1-decay)
        if step % interval == 0 or step == updates:
            with torch.no_grad():
                # Fixed validation subset and noise seed, so the curve is comparable across steps.
                j = torch.randint(len(vs), (min(2048, len(vs)),), generator=valid_rng)
                state = torch.get_rng_state(); torch.manual_seed(cfg['seed']+2)
                _, vfm, vce = flow.loss(vs[j], va[j], cfg['boundary_epsilon'])
                torch.set_rng_state(state)
            logs.append(dict(step=step, loss=loss.item(), fm=fm.item(), gripper_ce=ce.item(),
                             valid_fm=vfm.item(), valid_gripper_ce=vce.item()))
            print(json.dumps(logs[-1]), flush=True)
        if step in snapshots or step == updates:
            save(flow, cfg, manifest, step, out/f'flow_{step}.pt', ema, options)
    write_json(out/'bc.json', dict(chunk=chunk, updates=logs, train_samples=len(s), valid_samples=len(vs),
        parameters=sum(p.numel() for p in flow.parameters()), seconds=time.perf_counter()-start,
        hidden=flow.hidden, layers=flow.layers, gripper_hidden=flow.gripper_hidden,
        gripper_layers=flow.gripper_layers, options=options, dataset_sha256=manifest['sha256']))


def evaluate(cfg, flow_path, out, episodes, seed, execute=None, ema=False):
    from .env import Task
    out = Path(out)
    if out.exists():
        raise FileExistsError(out)
    torch.set_num_threads(1)
    flow, ck = load(flow_path, cfg, ema)
    execute = flow.chunk if execute is None else execute
    if not 1 <= execute <= flow.chunk:
        raise ValueError('execute must lie in [1, chunk]')
    before, start = fingerprint(flow), time.perf_counter()
    env, rows = Task(cfg, ck['manifest']['env']), []
    try:
        for ep in range(episodes):
            env_seed = seed+ep
            generator = torch.Generator().manual_seed(env_seed+100000)
            s, done, steps = env.reset(env_seed), False, 0
            while not done:
                plan = flow.sample(tensor(s)[None], cfg['density']['steps'], generator)[0].numpy()
                for command in plan[:execute]:
                    s, _, done, info = env.step(command); steps += 1
                    if done:
                        break
            rows.append(dict(seed=env_seed, success=info['success'], steps=steps))
            print(json.dumps(rows[-1]), flush=True)
    finally:
        env.close()
    assert fingerprint(flow) == before
    n, k = len(rows), sum(r['success'] for r in rows)
    z = 1.96; p = k/n; denom = 1+z*z/n
    center, half = (p+z*z/(2*n))/denom, z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/denom
    report = dict(task=cfg['task'], checkpoint=str(Path(flow_path).resolve()), checkpoint_sha256=sha256(flow_path),
        chunk=flow.chunk, execute=execute, hidden=flow.hidden, layers=flow.layers, updates=ck['updates'],
        ema=ema, gripper_hidden=flow.gripper_hidden, gripper_layers=flow.gripper_layers,
        options=ck.get('options', {}),
        successes=k, episodes_count=n, success_rate=p, wilson_95_interval=[center-half, center+half],
        mean_steps=float(np.mean([r['steps'] for r in rows])), eval_seed=seed,
        elapsed_seconds=time.perf_counter()-start, episodes=rows)
    write_json(out/'summary.json', report)
    print(json.dumps({k: v for k, v in report.items() if k != 'episodes'}), flush=True)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='command', required=True)
    t = sub.add_parser('train')
    t.add_argument('--config', required=True)
    t.add_argument('--chunk', type=int, required=True, help='Prediction horizon H')
    t.add_argument('--updates', type=int, help='Default: bc_updates from the config')
    t.add_argument('--snapshots', type=int, nargs='*', default=[], help='Extra checkpoints at these updates')
    t.add_argument('--out', required=True)
    for name in ('hidden', 'layers', 'gripper-hidden', 'gripper-layers'):
        t.add_argument('--'+name, type=int, help='Override the network size from the config')
    t.add_argument('--weight-decay', type=float, default=0., help='AdamW weight decay; 0 = plain Adam')
    t.add_argument('--cosine', action='store_true', help='Cosine learning-rate decay to zero')
    t.add_argument('--ema', type=float, default=0., help='EMA decay for an averaged copy, e.g. 0.999; 0 = off')
    e = sub.add_parser('eval')
    e.add_argument('--config', required=True)
    e.add_argument('--flow', required=True)
    e.add_argument('--out', required=True)
    e.add_argument('--episodes', type=int, default=50)
    e.add_argument('--seed', type=int, default=30000)
    e.add_argument('--execute', type=int, help='Commands executed per plan; default: the whole chunk')
    e.add_argument('--ema', action='store_true', help='Evaluate the EMA weights')
    args = p.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    if args.command == 'train':
        train(cfg, args.chunk, args.updates or cfg['bc_updates'], args.snapshots, args.out, args.hidden,
              args.layers, args.gripper_hidden, args.gripper_layers, args.weight_decay, args.cosine, args.ema)
    else:
        evaluate(cfg, args.flow, args.out, args.episodes, args.seed, args.execute, args.ema)


if __name__ == '__main__':
    main()
