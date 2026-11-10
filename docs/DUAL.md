# Dual RL: steer the noise and reshape the flow

Two learners share one action critic `Q(s, a)` and both move the frozen-BC policy
toward high return:

| Learner | Changes | Signal | Schedule |
|---|---|---|---|
| Noise RL (DSRL, unchanged) | `pi_omega(z \| s)`: which noise is fed to the flow | Noise critic `Q(s, z)` | Every actor update |
| Distribution RL (new) | Bounded flow correction `v = v_BC + r*tanh(g_theta)` | Action critic `Q(s, a)` | Every `flow_every` actor updates |

The executed arm command is `tanh(ODE_v(s, z))` with `z ~ pi_omega(.|s)`; the gripper
stays on the frozen BC head. The BC checkpoint is never modified: `ResidualFlow`
wraps it and trains only `g_theta` (zero-initialized, so training starts exactly at BC).

## Flow update (Option A)

1. For replay states, draw `M = flow_samples` noises `z ~ N(0, I)` and decode them
   through the **frozen BC flow** (`ResidualFlow.base_latent`).
2. Score each candidate with the action critic (twin minimum, gripper marginalized).
3. Weight candidates **within each state** by `softmax(Q / flow_tau)`.
4. Flow matching on the weighted candidates (pre-tanh targets), plus
   `demo_weight` times unweighted flow matching on BC training demonstrations.

The target is `pi_BC * exp(Q / flow_tau)`: one tilt anchored to BC. Candidates come
from the BC flow, not from the current corrected flow or from `pi_omega`, so repeated
flow updates do not compound and steering is not counted twice. Steering then adds
its own tilt (`kl_beta`) on top. `residual_bound` caps the velocity correction, so for
the same noise every action stays within a bounded distance of BC's.

## Configuration

Add a `dual` block inside `dsrl` (see `configs/can_dual_small.json`); without it the
runner is plain DSRL, unchanged.

| Key | Meaning | Example |
|---|---|---|
| `residual_bound` | Cap `r` on each velocity-correction component (pre-tanh units) | 0.5 |
| `flow_every` | Actor updates per flow update | 5 |
| `flow_lr` | Adam learning rate of `g_theta` | 1e-4 |
| `flow_tau` | Tilt temperature, in return units (reward is -1 per step) | 1.0 |
| `flow_samples` | Candidates per state, at least 2 | 8 |
| `demo_weight` | Weight of the demonstration anchor; 0 disables it | 1.0 |
| `demo_batch` | Demonstration batch size | 128 |

Optional, shared with DSRL: `actor_std_min` (std floor of `pi_omega`, default `e^-5`).

## Outputs

- Checkpoint schema `flowq.dsrl.v3` stores the correction and its optimizer with the
  learner; `load_checkpoint` rebuilds a `DualLearner`.
- `updates.jsonl` adds `flow_tilt_loss`, `flow_demo_loss`, `flow_weight_ess`
  (1 = one candidate dominates, `M` = uniform), `flow_q_spread`, and
  `flow_displacement_mean/max` (corrected vs BC action for the same noise).
- Each evaluation pairs three arms on the same seeds and noise: `bc` (original flow,
  prior noise), `reshaped` (corrected flow, prior noise), `dsrl` (corrected flow,
  learned noise).

## Running

```bash
$PY -m flowq.dsrl_runner train --config configs/can_dual_small.json \
  --flow runs/verify_flow_12036366/can/bc/flow.pt --out runs/can_dual_small_seed0
```

Use the same flow, seed and budget as a `can_dsrl_small.json` run for a matched
DSRL-only comparison. Tests: `tests/test_dual.py`.
