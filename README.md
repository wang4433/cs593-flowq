# FlowQ: frozen Flow-Matching BC for Robomimic

Research prototype extending Q2RL-style value estimation and action gating to
frozen flow-matching imitation policies. Only the original Robomimic **Lift,
Can and Square** tasks and PH demonstrations are used.

The completed work is **stage 1: train and evaluate Flow BC**. Value estimation,
likelihood/entropy and A/B/C runner code is implemented, but a trained-policy
Q-gating pilot has not been run. No RL improvement is claimed.

| Task | Closed-loop success | Evaluation horizon | Evaluation interactions | Record |
|---|---:|---:|---:|---|
| Lift | **12/20 (60%)** | 300 | 3,007 | [JSON](results/lift_bc_eval.json) |
| Can | **12/20 (60%)** | 300 | 3,758 | [JSON](results/can_bc_eval.json) |
| Square | **3/20 (15%)** | 500 | 8,971 | [JSON](results/square_bc_eval.json) |

These are actual simulator results from one training seed (0), 10,000 BC updates
per task, and evaluation reset seeds 30000–30019. Models were frozen throughout
evaluation. Square is still weak, and these small evaluations are not evidence
of a statistically significant improvement. See [report](docs/REPORT.md),
[all-task metrics](results/bc_stage1_all_tasks.json), and
[reference audit](docs/REFERENCE_AUDIT.md).

This Git repository contains source, configurations, dependency pins and compact
evaluation records. Datasets, pretrained weights, videos, raw trajectories,
local environments and machine-specific setup history are not stored in Git.
Checkpoint hashes in `results/` identify the locally evaluated models. The
commands below train new local checkpoints from the original demonstrations.

## Setup

Validated on Linux, Python 3.10.22, torch 2.5.1+cpu, numpy 1.26.4,
robosuite 1.4.1 and MuJoCo 3.2.6. No GPU is required. Q2RL pins MuJoCo 3.1.6;
this explicit simulator difference remains, so this is not a bit-for-bit
reference reproduction. The prototype reads HDF5 and stock robosuite directly,
without installing another Robomimic wrapper version.

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.lock.txt
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export NUMBA_CACHE_DIR="$PWD/.cache/numba"
export MUJOCO_GL=egl
```

Alternatively, use `conda env create -f environment.yml`, then
`source scripts/activate.sh`. The activation script locates Conda and supports
both a project-local environment and a named `cs593` environment.

## Original demonstration data

```bash
python scripts/prepare_data.py --tasks lift can square --checkpoints
for task in lift can square; do
  python scripts/audit_reference_checkpoint.py data/reference/$task/gmm.pth \
    --out artifacts/reference_$task.json
  python -m flowq.runner audit-data --config configs/${task}_bc_stage1.json \
    --out artifacts/${task}_dataset.json
done
```

Downloads use the official mirror `theaiinstitute/q2rl_robomimic_datasets`,
revision recorded for the experiments:
`ba3b72ac9f372634a1b7293f140bc3bd0f3acf89`. Files are
`{lift,can,square}/ph/low_dim_v141.hdf5`. The downloader records the resolved
revision, URL, SHA256 and LFS checksum. `--checkpoints` downloads the official
GMM files for mask auditing; those are not the flow checkpoints.

| Task | Training / validation masks | Demonstrations | Transitions |
|---|---|---:|---:|
| Lift | `20_percent_train` / `20_percent_valid` | 36 / 4 | 1,769 / 223 |
| Can | `train` / `valid` | 180 / 20 | 20,883 / 2,324 |
| Square | `train` / `valid` | 180 / 20 | 27,165 / 2,989 |

Masks were verified from the released GMM checkpoint configurations. Dataset
metadata must report robosuite 1.4.1; no automatic version conversion is allowed.
Do not substitute today's v1.5.1-converted downloads. No newly collected or
synthetic demonstrations are used. Evaluation records never enter BC training.

## Train and evaluate Flow BC

Run from the repository root with the environment above active:

```bash
for task in lift can square; do
  python -m flowq.runner train-bc --config configs/${task}_bc_stage1.json \
    --out runs/${task}_bc_stage1/bc
  python scripts/rollout_bc.py --config configs/${task}_bc_stage1.json \
    --flow runs/${task}_bc_stage1/bc/flow.pt \
    --episodes 20 --seed 30000 --out runs/${task}_bc_stage1/eval
done
```

Use new output directories for repeat runs. Each training run writes `flow.pt`
and `bc.json` (losses, exact demonstration IDs and provenance). Each evaluation
writes `summary.json`, `episodes.json` and `episode_*.npz`. Evaluation loads and
freezes the policy, repeatedly observes state, samples one action and steps the
simulator until success or timeout. It requires no value model or RL critic.

The physical state concatenates `robot0_eef_pos`, `robot0_eef_quat` (xyzw),
`robot0_gripper_qpos`, and `object` (live name `object-state`). Its dimension is
19 for Lift and 23 for Can/Square. Mean and sample std are fitted on training
demonstrations only, with a 0.02 std floor. No time-to-go, action history, images
or action chunks enter the BC policy.

The model uses a six-dimensional Gaussian-base arm flow and a separate
three-class gripper head over `{-1,0,+1}`, matching PandaGripper's sign operation.
The two-hidden-layer velocity MLP has width 128 and smooth bounded output
`6*tanh(raw_velocity/6)`. Sampling integrates 16 RK4 steps then applies tanh.
During BC only, arm targets use `atanh(0.999*action)`. There is no inference
clipping. This is a hybrid flow/categorical policy; any action density must use
Lebesgue measure for the arm and counting measure for the gripper.

```python
import json
import torch
from flowq.runner import load_flow

cfg = json.load(open("configs/can_bc_stage1.json"))
policy, checkpoint = load_flow("runs/can_bc_stage1/bc/flow.pt", cfg)
# state23: unnormalized current physical observation, ordered as documented above
s = torch.as_tensor(state23, dtype=torch.float64)[None]
action7 = policy.sample(s, steps=16)[0].numpy()
assert not policy.training
assert all(not parameter.requires_grad for parameter in policy.parameters())
```

## Checks and Lift video rendering

```bash
python -m unittest discover -s tests -v
python -m pip install -r requirements.render.txt
python scripts/render_bc_results.py --episodes 0 2 7 1 \
  --out runs/lift_bc_stage1/rendered
```

Four tests cover analytic Gaussian likelihood/entropy, integration convergence,
trace variance, action transforms, gradients/replay, and OOD action saturation.
The rendering command requires the Lift evaluation outputs above. It produces
MP4s using EGL, replays saved actions and checks every state/reward/termination
against the recorded episode. Its default examples are three successes and one
timeout in the recorded local result; a new training run can differ.

## Implemented stage 2 (not yet run with the trained checkpoints)

`flowq/flow.py` exposes sampling, continuous change-of-variables log likelihood
with exact or Hutchinson divergence, and Monte Carlo entropy. `qest` collects
complete frozen-flow episodes and fits a value model. The approximation is
`Q_FM = V_FM + q_alpha * (log_prob + entropy)` under a Boltzmann-policy assumption;
it is not guaranteed true-Q recovery. `q_alpha` is separate from `sac_alpha`.

One runner supports A (frozen flow only), B (a shared learned critic scores both
proposals), and C (frozen flow-derived score versus trainable RL critic, original
hard comparison). Executed transitions from both policies enter replay. B/C
must use the same task-specific flow checkpoint and warmup artifact. Initial
rollout collection counts toward the interaction budget.

Example commands for the prepared bounded Square configuration, when that stage
is explicitly resumed; these experiments have **not** been completed:

```bash
python -m flowq.runner train-bc --config configs/square_pilot.json --out runs/square_pilot/bc
python -m flowq.runner qest --config configs/square_pilot.json \
  --flow runs/square_pilot/bc/flow.pt --out runs/square_pilot/qest
for method in A B C; do
  python -m flowq.runner run --config configs/square_pilot.json \
    --flow runs/square_pilot/bc/flow.pt --warmup runs/square_pilot/qest/warmup.pt \
    --method $method --out runs/square_pilot/$method
done
python -m flowq.runner eval --config configs/square_pilot.json \
  --flow runs/square_pilot/bc/flow.pt --warmup runs/square_pilot/qest/warmup.pt \
  --online runs/square_pilot/C/online.pt --method C --out runs/square_pilot/C/eval.json
```

The finite-horizon reward convention, hybrid SAC-style actor, and other explicit
deviations from upstream are documented in the reference audit. Full-budget
Lift/Can/Square configurations are prepared but remain unlaunched.

## References and attribution

- [Q2RL paper](https://arxiv.org/html/2605.05172v3)
- [Official Q2RL implementation](https://github.com/rai-opensource/q2rl), audited at
  `d565ab09f5ae88546f2dec43445fe8701ee3de65`
- [Original Robomimic datasets](https://robomimic.github.io/docs/datasets/robomimic_v0.1.html)
- [FFJORD / continuous-flow density estimation](https://arxiv.org/abs/1810.01367)

Selected upstream source is preserved under `reference/q2rl/` for audit; this
is not a complete executable upstream checkout. Its original copyright and
[license](reference/q2rl/LICENSE) are preserved. See [third-party notice](THIRD_PARTY_NOTICES.md).
