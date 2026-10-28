"""Single-step conditional flow matching and continuous change of variables.

All densities are in nats w.r.t. the action command, after tanh: Lebesgue
measure for arm commands and counting measure for the sign-discretized gripper.
No clipping is performed at inference. Divergence traces are detached because
these queries are for a frozen policy, never a likelihood-training objective.
"""
import hashlib
import math
import torch
from torch import nn

DTYPE = torch.float64


def tensor(x):
    return torch.as_tensor(x, dtype=DTYPE)


def mlp(n, m, hidden=128):
    return nn.Sequential(nn.Linear(n, hidden), nn.SiLU(), nn.Linear(hidden, hidden),
                         nn.SiLU(), nn.Linear(hidden, m)).to(DTYPE)


def fingerprint(module):
    h = hashlib.sha256()
    for k, v in sorted(module.state_dict().items()):
        h.update(k.encode()); h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def freeze(module):
    module.eval()
    module.requires_grad_(False)
    return module


def finite(x, name):
    if not torch.isfinite(torch.as_tensor(x)).all():
        raise FloatingPointError(f"Nonfinite {name}")


def log_jacobian(y):
    # log |d tanh(y) / dy|, stable for large |y|
    return (2 * (math.log(2) - y - torch.nn.functional.softplus(-2*y))).sum(-1)


def rk4(fn, y, start, stop, steps):
    if steps < 1:
        raise ValueError("ODE steps must be positive")
    h = (stop-start)/steps
    for i in range(steps):
        t = start+i*h
        k1 = fn(t, y)
        k2 = fn(t+h/2, y+h*k1/2)
        k3 = fn(t+h/2, y+h*k2/2)
        k4 = fn(t+h, y+h*k3)
        y = (y+h*(k1+2*k2+2*k3+k4)/6).detach()
    finite(y, "ODE solution")
    return y


class Flow(nn.Module):
    def __init__(self, obs_dim, action_dim=7, hidden=128, hybrid=True, max_velocity=None):
        super().__init__()
        self.obs_dim, self.action_dim, self.hidden = obs_dim, action_dim, hidden
        self.hybrid = hybrid
        self.max_velocity = max_velocity
        if max_velocity is not None and max_velocity <= 0:
            raise ValueError('max_velocity must be positive')
        self.continuous_dim = action_dim-int(hybrid)
        self.net = mlp(obs_dim+self.continuous_dim+1, self.continuous_dim, hidden)
        self.gripper = mlp(obs_dim, 3, hidden) if hybrid else None
        self.register_buffer('mean', torch.zeros(obs_dim, dtype=DTYPE))
        self.register_buffer('std', torch.ones(obs_dim, dtype=DTYPE))

    def gripper_dist(self, s):
        return torch.distributions.Categorical(logits=self.gripper((s[..., :self.obs_dim]-self.mean)/self.std))

    def velocity(self, t, x, s):
        s = (s[..., :self.obs_dim]-self.mean)/self.std
        time = torch.full_like(x[..., :1], float(t)) if not torch.is_tensor(t) else t
        v = self.net(torch.cat([s, x, time], -1))
        # A bounded smooth vector field prevents OOD states from driving the
        # tanh action transform to floating-point saturation. This changes the
        # trained model, not the sampled action; it preserves an invertible ODE.
        return v if self.max_velocity is None else self.max_velocity*torch.tanh(v/self.max_velocity)

    def loss(self, s, actions, boundary_epsilon=1e-3):
        # Deterministic inward smoothing of empirical boundary commands, training only.
        if (actions.abs()>1).any():
            raise ValueError('Demonstration action outside [-1,1]')
        if not 0 < boundary_epsilon < 1:
            raise ValueError('Invalid boundary smoothing epsilon')
        arm = actions[..., :self.continuous_dim]
        x1 = torch.atanh(arm*(1-boundary_epsilon))
        x0 = torch.randn_like(x1)
        t = torch.rand_like(x1[:, :1])
        loss = ((self.velocity(t, (1-t)*x0+t*x1, s)-(x1-x0))**2).mean()
        if self.hybrid:
            loss = loss-self.gripper_dist(s).log_prob(actions[..., -1].sign().long()+1).mean()
        return loss

    @torch.no_grad()
    def sample(self, s, steps=16, generator=None):
        z = torch.randn((*s.shape[:-1], self.continuous_dim), dtype=s.dtype,
                        device=s.device, generator=generator)
        y = rk4(lambda t,x:self.velocity(t,x,s), z, 0, 1, steps)
        a = torch.tanh(y)
        if (a.abs()>=1).any():
            raise FloatingPointError('Floating-point tanh saturation; action density is undefined')
        if self.hybrid:
            grip = torch.multinomial(self.gripper_dist(s).probs, 1, generator=generator)-1
            a = torch.cat([a, grip.to(a.dtype)], -1)
        return a

    def log_prob(self, s, a, steps=16, trace='exact', trace_samples=1, generator=None):
        if a.shape[-1] != self.action_dim or (a[..., :self.continuous_dim].abs()>=1).any():
            raise ValueError('log_prob requires an unsaturated continuous command in (-1,1)')
        finite(a, 'action')
        grip_lp = 0
        if self.hybrid:
            if not ((a[..., -1] == -1) | (a[..., -1] == 0) | (a[..., -1] == 1)).all():
                raise ValueError('Gripper command must already be discretized to -1, 0, or 1')
            grip_lp = self.gripper_dist(s).log_prob(a[..., -1].long()+1).detach()
            a = a[..., :-1]
        if trace not in ('exact', 'hutchinson') or trace_samples < 1:
            raise ValueError('Invalid trace settings')
        y = torch.atanh(a)
        probes = None
        if trace == 'hutchinson':
            # Fixed probes along each entire ODE path (not resampled per stage).
            probes = torch.randint(2, (trace_samples, *a.shape), generator=generator,
                                   device=a.device).to(a.dtype)*2-1

        def rhs(t, aug):
            with torch.enable_grad():
                x = aug[..., :-1].detach().requires_grad_(True)
                v = self.velocity(t, x, s)
                if trace == 'exact':
                    div = sum(torch.autograd.grad(v[..., j].sum(), x, retain_graph=True)[0][..., j]
                              for j in range(self.continuous_dim))
                else:
                    div = sum((torch.autograd.grad((v*e).sum(), x, retain_graph=True)[0]*e).sum(-1)
                              for e in probes)/trace_samples
            return torch.cat([v.detach(), div.detach()[..., None]], -1)

        result = rk4(rhs, torch.cat([y, torch.zeros_like(y[..., :1])], -1), 1, 0, steps)
        z, negative_integral = result[..., :-1], result[..., -1]
        lp = -0.5*(z*z+math.log(2*math.pi)).sum(-1)+negative_integral-log_jacobian(y)
        finite(lp, 'log density')
        return lp.detach()+grip_lp

    def entropy(self, s, samples=8, generator=None, **density):
        if samples < 1:
            raise ValueError('Entropy samples must be positive')
        repeated = s.repeat_interleave(samples, dim=0)
        a = self.sample(repeated, steps=density.get('steps', 16), generator=generator)
        h = -self.log_prob(repeated, a, generator=generator, **density).reshape(len(s),samples)
        se = h.std(-1, unbiased=True)/math.sqrt(samples) if samples>1 else torch.full_like(h[:,0], float('nan'))
        return h.mean(-1), se
