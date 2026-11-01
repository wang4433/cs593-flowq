"""Is the flow-matching BC term learning the conditional action distribution?

No simulator, no policy updates after training. Two parts:
  synthetic  train Flow with the real loss and stage-1 hyperparameters on a known
             bimodal state-conditioned density; compare the learned density, mode
             weights, per-mode moments and gripper probabilities to the oracle.
  demos      held-out diagnostics of a trained checkpoint on original validation
             demonstrations, against Gaussian maximum-likelihood baselines.
Arm densities are in the flow's training coordinates x=atanh((1-eps)*a), so the
tanh Jacobian (identical for every model) never enters a comparison.
"""
import argparse
import json
import math
import time
from pathlib import Path
import numpy as np
import torch
from scipy.stats import chi2, wasserstein_distance
from flowq.data import load_demos, sha256
from flowq.flow import DTYPE, Flow, fingerprint, freeze, log_jacobian, mlp, rk4, tensor
from flowq.runner import load_flow, seed, write_json

EPS = 1e-3


def to_x(a):
    return torch.atanh(a[..., :6]*(1-EPS))


def grip_index(a):
    return a[..., -1].sign().long()+1


def flow_arm_logpdf(flow, s, x, g, steps=16, batch=512):
    # Uses the production log_prob path, then removes the gripper and tanh terms.
    out = []
    for i in range(0, len(s), batch):
        ss, xx, gg = s[i:i+batch], x[i:i+batch], g[i:i+batch]
        lp = flow.log_prob(ss, torch.cat([xx.tanh(), (gg-1).to(DTYPE)[:, None]], -1), steps=steps)
        out.append(lp-flow.gripper_dist(ss).log_prob(gg)+log_jacobian(xx))
    return torch.cat(out)


@torch.no_grad()
def latents(flow, s, x, steps=16):
    return rk4(lambda t, y: flow.velocity(t, y, s), x, 1, 0, steps)


class Gauss(torch.nn.Module):
    """Conditional diagonal Gaussian over x plus categorical gripper; same MLP and normalization."""
    def __init__(self, obs_dim, hidden, mean, std):
        super().__init__()
        self.net = mlp(obs_dim, 15, hidden)
        self.register_buffer('mean', mean.clone()); self.register_buffer('std', std.clone())

    def dists(self, s):
        o = self.net((s-self.mean)/self.std)
        return (torch.distributions.Normal(o[:, :6], o[:, 6:12].clamp(-7, 2).exp()),
                torch.distributions.Categorical(logits=o[:, 12:]))

    def loss(self, s, a):
        d, g = self.dists(s)
        return -(d.log_prob(to_x(a)).sum(-1)+g.log_prob(grip_index(a))).mean()


def fit(model, s, a, updates, lr, batch_size):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(updates):
        i = torch.randint(len(s), (batch_size,))
        loss = model.loss(s[i], a[i])
        opt.zero_grad(); loss.backward(); opt.step()
    return freeze(model)


def new_flow(s, hidden, max_velocity, std_floor):
    flow = Flow(s.shape[-1], hidden=hidden, max_velocity=max_velocity)
    flow.mean.copy_(s.mean(0)); flow.std.copy_(s.std(0).clamp_min(std_floor))
    return flow


# ---------------------------------------------------------------- synthetic oracle
DIRECTION = tensor([.8, -.8, .5, 0., .6, -.4])
STD = .25


def oracle(s):
    weight = torch.sigmoid(2*s[:, 0])
    centre = .3*s[:, 2:3]
    offset = (1+.3*torch.tanh(s[:, 1:2]))*DIRECTION
    logits = torch.stack([s[:, 0], torch.zeros_like(s[:, 0]), s[:, 1]-s[:, 0]], -1)
    return weight, centre, offset, logits


def oracle_sample(s, generator):
    weight, centre, offset, logits = oracle(s)
    sign = (torch.rand(len(s), generator=generator, dtype=DTYPE) < weight).to(DTYPE)*2-1
    x = centre+sign[:, None]*offset+STD*torch.randn(len(s), 6, generator=generator, dtype=DTYPE)
    if x.abs().max() >= math.atanh(1-EPS):
        raise ValueError('Oracle sample outside the representable training range')
    grip = torch.multinomial(torch.softmax(logits, -1), 1, generator=generator)[:, 0]-1
    # Chosen so the training transform atanh((1-eps)a) returns x exactly.
    return torch.cat([x.tanh()/(1-EPS), grip.to(DTYPE)[:, None]], -1), x, sign


def oracle_logpdf(s, x):
    weight, centre, offset, _ = oracle(s)
    normal = lambda m: torch.distributions.Normal(m, STD).log_prob(x).sum(-1)
    return torch.logsumexp(torch.stack([weight.log()+normal(centre+offset),
                                        (1-weight).log()+normal(centre-offset)]), 0)


def synthetic(args):
    seed(args.seed); start = time.perf_counter()
    g = torch.Generator().manual_seed(args.seed)
    s = torch.randn(args.n_train, 3, generator=g, dtype=DTYPE)
    a, _, _ = oracle_sample(s, g)
    vs = torch.randn(args.n_eval, 3, generator=g, dtype=DTYPE)
    va, vx, _ = oracle_sample(vs, g)
    vg = grip_index(va)
    truth = oracle_logpdf(vs, vx)
    flow = new_flow(s, args.hidden, args.max_velocity, args.std_floor)
    untrained = freeze(new_flow(s, args.hidden, args.max_velocity, args.std_floor))
    opt = torch.optim.Adam(flow.parameters(), lr=args.lr); curve = []
    for step in range(args.updates):
        i = torch.randint(len(s), (args.batch_size,))
        loss = flow.loss(s[i], a[i], EPS)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 1000 == 0 or step == args.updates-1:
            curve.append(dict(step=step+1, loss=loss.item()))
            print(json.dumps(curve[-1]), flush=True)
    freeze(flow)
    gauss = fit(Gauss(3, args.hidden, flow.mean, flow.std), s, a, args.updates, args.lr, args.batch_size)
    report = dict(part='synthetic', n_train=args.n_train, n_eval=args.n_eval, updates=args.updates,
                  oracle_log_density_mean=truth.mean().item())
    # 1. Density: KL(p||q) = E_p[log p - log q] on held-out (s, x) from the oracle.
    with torch.no_grad():
        d, c = gauss.dists(vs)
        gauss_lp = d.log_prob(vx).sum(-1)
    for name, lp in [('flow', flow_arm_logpdf(flow, vs, vx, vg)), ('untrained_flow', flow_arm_logpdf(untrained, vs, vx, vg)),
                     ('gaussian', gauss_lp)]:
        kl = truth-lp
        report[f'{name}_arm_kl_nats'] = kl.mean().item()
        report[f'{name}_arm_kl_se'] = (kl.std()/math.sqrt(len(kl))).item()
    # 2. Gripper: total variation and KL against the true categorical.
    true_p = torch.softmax(oracle(vs)[3], -1)
    with torch.no_grad():
        for name, q in [('flow', flow.gripper_dist(vs).probs), ('gaussian', c.probs)]:
            report[f'{name}_gripper_tv'] = (.5*(true_p-q).abs().sum(-1)).mean().item()
            report[f'{name}_gripper_kl_nats'] = (true_p*(true_p.log()-q.log())).sum(-1).mean().item()
    # 3. Sample-based checks, independent of log_prob: mode weights and per-mode moments.
    k = args.mode_samples; states = vs[:args.mode_states]
    rep = states.repeat_interleave(k, 0)
    # Flow samples tanh(y) with y in training coordinates, so invert with plain atanh.
    sampled = torch.atanh(flow.sample(rep, steps=16, generator=torch.Generator().manual_seed(args.seed+1))[:, :6])
    with torch.no_grad():
        d, _ = gauss.dists(rep); gauss_x = d.sample()
    _, oracle_x, _ = oracle_sample(rep, torch.Generator().manual_seed(args.seed+3))
    weight, centre, offset, _ = oracle(rep)
    for name, xs in [('oracle', oracle_x), ('flow', sampled), ('gaussian', gauss_x)]:
        proj = ((xs-centre)*offset).sum(-1)/(offset*offset).sum(-1)  # +1 / -1 at the mode centres
        plus = (proj > 0).to(DTYPE).reshape(-1, k).mean(-1)
        report[f'{name}_mode_weight_mae'] = (plus-weight.reshape(-1, k)[:, 0]).abs().mean().item()
        report[f'{name}_fraction_between_modes'] = (proj.abs() < .5).to(DTYPE).mean().item()
        sign = torch.where(proj > 0, 1., -1.).to(DTYPE)
        resid = (xs-centre-sign[:, None]*offset)
        report[f'{name}_within_mode_mean_abs_error'] = resid.reshape(-1, k, 6).mean(1).abs().mean().item()
        report[f'{name}_within_mode_std'] = resid.std().item()
    report.update(loss_curve=curve, flow_fingerprint=fingerprint(flow), seconds=time.perf_counter()-start)
    return report


# ---------------------------------------------------------------- real demonstrations
def fm_loss_terms(flow, s, a, repeats, batch=2048):
    arm = grip = 0.
    for _ in range(repeats):
        for i in range(0, len(s), batch):
            ss, aa = s[i:i+batch], a[i:i+batch]
            with torch.no_grad():
                total = flow.loss(ss, aa, EPS).item()
                ce = -flow.gripper_dist(ss).log_prob(grip_index(aa)).mean().item()
            arm += (total-ce)*len(ss); grip += ce*len(ss)
    n = repeats*len(s)
    return arm/n, grip/n


def demos(args):
    cfg = json.loads(Path(args.config).read_text()); seed(cfg['seed']); start = time.perf_counter()
    flow, ck = load_flow(args.flow, cfg); before = fingerprint(flow)
    (s, a), (vs, va) = load_demos(cfg, ck['manifest'])
    s, a, vs, va = map(tensor, (s, a, vs, va))
    x, vx, vg = to_x(a), to_x(va), grip_index(va)
    report = dict(part='demos', task=cfg['task'], flow=args.flow, flow_sha256=sha256(args.flow),
                  train_transitions=len(s), valid_transitions=len(vs))
    # 0. Data: saturated arm commands and gripper support.
    report['train_arm_fraction_abs_ge_0999'] = (a[:, :6].abs() >= .999).to(DTYPE).mean(0).tolist()
    report['train_arm_fraction_abs_eq_1'] = (a[:, :6].abs() == 1).to(DTYPE).mean(0).tolist()
    report['train_gripper_values'] = {str(k): int(v) for k, v in zip(*np.unique(a[:, -1].numpy(), return_counts=True))}
    # 1. Flow-matching objective on train vs held-out (fixed noise and times).
    torch.manual_seed(args.seed)
    report['fm_arm_loss_train'], report['gripper_ce_train'] = fm_loss_terms(flow, s, a, args.fm_repeats)
    report['fm_arm_loss_valid'], report['gripper_ce_valid'] = fm_loss_terms(flow, vs, va, args.fm_repeats)
    # 2. Held-out negative log-likelihood against Gaussian baselines.
    gauss = fit(Gauss(s.shape[-1], cfg['flow_hidden'], flow.mean, flow.std), s, a,
                cfg['bc_updates'], cfg['lr'], cfg['batch_size'])
    flow_lp = flow_arm_logpdf(flow, vs, vx, vg)
    with torch.no_grad():
        d, c = gauss.dists(vs)
        gauss_lp = d.log_prob(vx).sum(-1)
        uncond = torch.distributions.MultivariateNormal(x.mean(0), covariance_matrix=torch.cov(x.T)+1e-6*torch.eye(6, dtype=DTYPE))
        report.update(valid_arm_nll_flow=-flow_lp.mean().item(), valid_arm_nll_flow_se=(flow_lp.std()/math.sqrt(len(vs))).item(),
                      valid_arm_nll_conditional_gaussian=-gauss_lp.mean().item(),
                      valid_arm_nll_unconditional_gaussian=-uncond.log_prob(vx).mean().item(),
                      valid_arm_nll_flow_minus_gaussian_se=((gauss_lp-flow_lp).std()/math.sqrt(len(vs))).item(),
                      valid_gripper_nll_flow=-flow.gripper_dist(vs).log_prob(vg).mean().item(),
                      valid_gripper_nll_gaussian=-c.log_prob(vg).mean().item(),
                      valid_gripper_accuracy_flow=(flow.gripper_dist(vs).probs.argmax(-1) == vg).to(DTYPE).mean().item(),
                      valid_gripper_accuracy_gaussian=(c.probs.argmax(-1) == vg).to(DTYPE).mean().item())
    # 3. Inverse map: held-out demo actions pushed back through the ODE should look N(0, I).
    z = latents(flow, vs, vx)
    sq = (z*z).sum(-1)
    report.update(valid_latent_mean=z.mean(0).tolist(), valid_latent_std=z.std(0).tolist(),
                  valid_latent_sq_norm_mean=sq.mean().item(), expected_sq_norm_mean=6.0,
                  valid_latent_fraction_beyond_chi2_99=(sq > chi2.ppf(.99, 6)).to(DTYPE).mean().item(),
                  expected_fraction_beyond_chi2_99=.01)
    # 4. Action prediction from K samples, in the original command space.
    k = args.k; rep = vs.repeat_interleave(k, 0); arm = va[:, :6]
    fs = flow.sample(rep, steps=cfg['density']['steps'], generator=torch.Generator().manual_seed(args.seed))[:, :6].reshape(-1, k, 6)
    with torch.no_grad():
        d, _ = gauss.dists(rep); gs = d.sample().tanh().reshape(-1, k, 6)
    for name, samples in [('flow', fs), ('gaussian', gs)]:
        report[f'valid_mse_sample_mean_{name}'] = ((samples.mean(1)-arm)**2).mean().item()
        report[f'valid_mse_single_sample_{name}'] = ((samples[:, 0]-arm)**2).mean().item()
        report[f'valid_mse_best_of_{k}_{name}'] = ((samples-arm[:, None])**2).mean(-1).min(1).values.mean().item()
        report[f'valid_marginal_w1_{name}'] = [wasserstein_distance(samples[:, 0, j].numpy(), arm[:, j].numpy()) for j in range(6)]
    report['valid_mse_gaussian_mean'] = ((torch.tanh(gauss.dists(vs)[0].mean)-arm)**2).mean().item()
    report['marginal_w1_train_vs_valid_demos'] = [wasserstein_distance(a[:, j].numpy(), arm[:, j].numpy()) for j in range(6)]
    # 5. ODE discretization: same noise, increasing RK4 steps, versus 64 steps.
    sub = vs[:512]; noise = torch.randn(len(sub), 6, generator=torch.Generator().manual_seed(args.seed+2), dtype=DTYPE)
    with torch.no_grad():
        decode = lambda n: rk4(lambda t, y: flow.velocity(t, y, sub), noise, 0, 1, n).tanh()
        reference = decode(64)
        report['ode_max_abs_action_error_vs_64'] = {n: (decode(n)-reference).abs().max().item() for n in (2, 4, 8, 16, 32)}
    assert fingerprint(flow) == before
    report.update(flow_unchanged=True, seconds=time.perf_counter()-start)
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument('part', choices=['synthetic', 'demos'])
    p.add_argument('--out', required=True)
    p.add_argument('--config'); p.add_argument('--flow')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--k', type=int, default=16)
    p.add_argument('--fm_repeats', type=int, default=8)
    # synthetic: defaults copy the stage-1 BC configuration
    p.add_argument('--n_train', type=int, default=20000); p.add_argument('--n_eval', type=int, default=2000)
    p.add_argument('--updates', type=int, default=10000); p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--lr', type=float, default=3e-4); p.add_argument('--hidden', type=int, default=128)
    p.add_argument('--max_velocity', type=float, default=6.0); p.add_argument('--std_floor', type=float, default=.02)
    p.add_argument('--mode_states', type=int, default=200); p.add_argument('--mode_samples', type=int, default=200)
    args = p.parse_args()
    if Path(args.out).exists(): raise FileExistsError(args.out)
    if args.part == 'demos' and not (args.config and args.flow): p.error('demos needs --config and --flow')
    report = synthetic(args) if args.part == 'synthetic' else demos(args)
    write_json(args.out, report)
    print(json.dumps({k: v for k, v in report.items() if not isinstance(v, (list, dict))}, indent=1), flush=True)


if __name__ == '__main__': main()
