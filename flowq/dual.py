"""Dual RL on a flow BC policy: DSRL noise steering plus a bounded, critic-weighted flow correction.

The executed arm command decodes learned noise z ~ pi_omega(.|s) through the flow
v = v_BC + r*tanh(g_theta). DSRL trains pi_omega and both critics exactly as before.
The correction g_theta is trained by flow matching on candidates a = f_BC(s, z),
z ~ N(0,I) through the FROZEN BC flow, weighted per state by softmax(Q/tau). Its
target is pi_BC * exp(Q/tau): one tilt anchored to BC, so repeated flow updates do
not compound and steering does not count twice (Option A). An optional unweighted
demonstration term anchors the correction further. See docs/DUAL.md.

Optional trust region (``dual.displacement_target``): tau is the multiplier of
KL(p_theta || pi_BC) in max E[Q] - tau*KL, so it is adapted by dual ascent to hold the
mean action displacement |f_theta(s,z) - f_BC(s,z)| at the target: tau shrinks
(stronger tilt) while the correction moves less than the target and grows back
(tilt toward BC itself) when it moves more. Without the key tau stays fixed.
"""
import math

import torch
from torch import nn

from .dsrl import DSRLLearner
from .flow import DTYPE, Flow, finite, mlp, rk4


class ResidualFlow(Flow):
    """Frozen BC flow plus a bounded velocity correction; starts exactly at BC.

    Every Flow method (decode, encode, sample, log_prob) runs through the corrected
    velocity. The base flow and its gripper head are never updated.
    """
    def __init__(self, base, bound, hidden=None, layers=None, layer_norm=False):
        """``hidden``/``layers`` default to the base flow's; the correction can also be sized
        independently (e.g. wider than a small base flow), with optional LayerNorm."""
        nn.Module.__init__(self)
        if not math.isfinite(bound) or bound <= 0:
            raise ValueError('Residual bound must be finite and positive')
        if base.training or any(p.requires_grad for p in base.parameters()):
            raise ValueError('ResidualFlow requires a frozen, evaluation-mode base flow')
        self.base, self.bound = base, float(bound)
        self.obs_dim, self.action_dim, self.hidden = base.obs_dim, base.action_dim, base.hidden
        self.hybrid, self.max_velocity, self.continuous_dim = base.hybrid, base.max_velocity, base.continuous_dim
        self.layers = base.layers
        self.correction = mlp(self.obs_dim+self.continuous_dim+1, self.continuous_dim,
                              hidden or base.hidden, layers or base.layers, layer_norm).to(base.mean.device)
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

    def gripper_dist(self, s):
        return self.base.gripper_dist(s)

    def velocity(self, t, x, s):
        time = torch.full_like(x[..., :1], float(t)) if not torch.is_tensor(t) else t
        g = self.correction(torch.cat([(s[..., :self.obs_dim]-self.mean)/self.std, x, time], -1))
        return self.base.velocity(t, x, s)+self.bound*torch.tanh(g)

    @torch.no_grad()
    def base_latent(self, s, z, steps=16):
        """Pre-tanh BC arm command ODE_BC(s, z): the candidates of the flow update."""
        return rk4(lambda t, x: self.base.velocity(t, x, s), z, 0, 1, steps)

    def matching_loss(self, s, y1, weights=None):
        """Flow matching toward pre-tanh targets y1, optionally weighted per sample."""
        finite(y1, 'flow target')
        x0 = torch.randn_like(y1)
        t = torch.rand_like(y1[:, :1])
        err = (self.velocity(t, (1-t)*x0+t*y1, s)-(y1-x0)).square().mean(-1)
        return err.mean() if weights is None else (weights*err).sum()/weights.sum()


class DualLearner(DSRLLearner):
    """DSRL steering of a ResidualFlow whose bounded correction is trained by the action critic."""
    def __init__(self, base, cfg, demos=None):
        self.dual = cfg['dual']
        super().__init__(ResidualFlow(base, self.dual['residual_bound'], self.dual.get('correction_hidden'),
                                      self.dual.get('correction_layers'),
                                      self.dual.get('correction_layer_norm', False)), cfg)
        self.base = base
        self.demo_states = self.demo_targets = None
        if demos is not None:
            states, actions = (torch.as_tensor(v, dtype=DTYPE, device=self.device) for v in demos)
            eps = cfg.get('boundary_epsilon', 1e-3)
            # Same inward smoothing and coordinates as BC training (Flow.loss).
            self.demo_states = states
            self.demo_targets = torch.atanh(actions[:, :base.continuous_dim]*(1-eps))
        self.init_correction_training()

    def init_correction_training(self):
        """Optimizer, counters and the (possibly adaptive) temperature of the correction."""
        self.flow_opt = torch.optim.Adam(self.flow.correction.parameters(), lr=self.dual['flow_lr'])
        self.flow_updates = 0
        self.flow_tau = float(self.dual['flow_tau'])
        self.displacement_ema = None
        self.flow_metrics = dict(flow_updates=0, flow_tau=self.flow_tau)

    def adapt_tau(self, displacement):
        """Dual ascent on log tau toward the mean-displacement target (trust region of BC)."""
        d = self.dual
        target = d.get('displacement_target')
        if target is None:
            return
        self.displacement_ema = (displacement if self.displacement_ema is None
                                 else .9*self.displacement_ema+.1*displacement)
        error = min(max((self.displacement_ema-target)/target, -1.), 1.)
        self.flow_tau = min(max(self.flow_tau*math.exp(d.get('tau_lr', .005)*error),
                                d.get('tau_min', .01)), d.get('tau_max', 100.))

    def update(self, batch, train_actor=True):
        metrics = super().update(batch, train_actor)
        # The correction trains on the actor schedule, never during critic-only warmup.
        if train_actor and self.updates % self.dual['flow_every'] == 0:
            self.update_flow(batch[0].to(self.device))
        metrics.update(self.flow_metrics)
        return metrics

    def update_flow(self, states):
        d, m = self.dual, self.dual['flow_samples']
        if d.get('flow_states') is not None:
            # Replay batches are uniform samples, so the first rows are a random subset. Fewer
            # states with more candidates each keeps the cost (states x candidates) unchanged
            # while letting the per-state best candidate lie further from typical BC.
            states = states[:d['flow_states']]
        steps = self.cfg.get('train_flow_steps', self.cfg['flow_steps'])
        with torch.no_grad():
            rep = states.repeat_interleave(m, 0)
            z = torch.randn(len(rep), self.flow.continuous_dim, dtype=states.dtype, device=states.device)
            y = self.flow.base_latent(rep, z, steps)
            q = self.action_value(self.action_critic, rep, torch.tanh(y)).reshape(-1, m)
            # Within-state weights: candidates compete only with each other, so no V(s) is needed.
            w = torch.softmax(q/self.flow_tau, -1)
        loss = self.flow.matching_loss(rep, y, w.reshape(-1))
        tilt_loss, anchor = loss.item(), None
        if d['demo_weight'] > 0:
            if self.demo_states is None:
                raise ValueError('demo_weight > 0 requires demonstrations')
            i = torch.randint(len(self.demo_states), (d['demo_batch'],), device=self.device)
            anchor = self.flow.matching_loss(self.demo_states[i], self.demo_targets[i])
            loss = loss+d['demo_weight']*anchor
        self._step(loss, self.flow_opt, self.flow.correction, 'flow correction loss')
        self.flow_updates += 1
        with torch.no_grad():
            # Same noise through the corrected and the BC flow: how far actions moved.
            k = min(len(states), 64)
            moved = (self.flow.decode(states[:k], z[::m][:k], steps)-torch.tanh(y[::m][:k])).abs()
        tau_used = self.flow_tau
        self.adapt_tau(moved.mean().item())
        self.flow_metrics = dict(flow_updates=self.flow_updates, flow_tilt_loss=tilt_loss,
            flow_tau=tau_used, flow_displacement_ema=self.displacement_ema,
            flow_demo_loss=None if anchor is None else anchor.item(),
            flow_weight_ess=(1/w.square().sum(-1)).mean().item(),
            flow_q_spread=q.std(-1).mean().item(),
            flow_displacement_mean=moved.mean().item(), flow_displacement_max=moved.max().item())
        return self.flow_metrics

    def state_dict(self):
        state = super().state_dict()
        state.update(correction=self.flow.correction.state_dict(),
                     flow_optimizer=self.flow_opt.state_dict(), flow_updates=self.flow_updates,
                     flow_tau=self.flow_tau, displacement_ema=self.displacement_ema)
        return state

    def load_state_dict(self, state):
        super().load_state_dict(state)
        self.flow.correction.load_state_dict(state['correction'])
        self.flow_opt.load_state_dict(state['flow_optimizer'])
        self.flow_updates = state['flow_updates']
        # Checkpoints from before the adaptive temperature used the fixed config value.
        self.flow_tau = state.get('flow_tau', float(self.dual['flow_tau']))
        self.displacement_ema = state.get('displacement_ema')
