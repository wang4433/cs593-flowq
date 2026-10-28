> Recorded local experiment report. Paths under `runs/`, `data/`, `artifacts/` and `deliverables/` describe generated local outputs, not files included in Git. Compact evaluation records are committed in `results/`; see the repository README for clean-clone commands.

# Stage 1 continuation — Can and Square Flow BC, 2026-10-27

The newly requested Can and Square BC training and closed-loop evaluation are
complete. Both used the existing bounded-velocity Flow BC implementation without
changing the algorithm. The original Lift checkpoint and evaluation are intact.

| Task | Closed-loop success | Horizon | Evaluation steps | Training / evaluation time |
|---|---:|---:|---:|---:|
| Lift | **12/20 (60%)** | 300 | 3,007 | 27.07 s / 27.85 s |
| Can | **12/20 (60%)** | 300 | 3,758 | 26.72 s / 40.55 s |
| Square | **3/20 (15%)** | 500 | 8,971 | 26.76 s / 101.68 s |

The models use two 128-wide hidden layers, six continuous flow arm dimensions
plus a three-class gripper head, std floor 0.02, max velocity 6, 16 RK4 steps,
batch 256, Adam 0.0003 and training seed 0. Both physical state vectors have 23
values: eef position 3, eef quaternion 4, gripper qpos 2, object features 14.
Normalization is fitted only on each task's training split. Actions are single
controller steps. Each final checkpoint was fixed at update 10,000 before the
simulator evaluation; no test episodes or checkpoints were filtered for success.
The earlier Square diagnostic was not resumed or overwritten.

## New artifacts

For each `task` in `can`, `square`:

- `runs/{task}_bc_stage1/bc/flow.pt`: loadable checkpoint, frozen by `load_flow`.
- `runs/{task}_bc_stage1/bc/bc.json`: training log, full demo ID and provenance manifest.
- `runs/{task}_bc_stage1/eval/summary.json`: closed-loop success and freeze checks.
- `runs/{task}_bc_stage1/eval/episode_000.npz` through `episode_019.npz`: states/actions/rewards.
- `runs/{task}_bc_stage1/train.log` and `eval.log`: stdout from the actual jobs.
- `configs/{task}_bc_stage1.json`: exact configuration.

All-task machine-readable summary: `artifacts/bc_stage1_all_tasks.json`.
Portable archive: `deliverables/all_tasks_flow_bc_stage1.zip`.

| Task | Checkpoint SHA256 |
|---|---|
| Can | `3b1623045be3796ad491518745d2d93f26e254d45023bc17edf297b739023b58` |
| Square | `31463990754db4b1ade27133ceed4ef1095faf0b694305cc320986e728352f7f` |

## Verified data and bounded checks

Official PH v141 `train`/`valid` masks were checked against the released GMM
checkpoint configs and the original dataset hashes before training. Can uses
20,883 training / 2,324 validation transitions; Square uses 27,165 / 2,989.
Each has 180 / 20 demonstrations and no train/validation overlap. Observation
and action preprocessing match the previously tested Lift implementation.

Every episode runs the real stock robosuite task until success or its specified
horizon. Seeds are 30000–30019; action-noise seeds are 130000–130019. All 40 new
episodes completed normally. All saved states/actions/rewards are finite, arm
actions are strictly inside (-1,1), and gripper actions are in {-1,0,1}.
Checkpoint reload, state/action dimensions, all trajectory lengths, reward sums,
checkpoint hashes and model fingerprints were independently checked. Both
policies remained frozen and evaluation performed no optimizer updates.
The existing numerical/gradient unit-test code is unchanged; it was not rerun
for this configuration-only extension. Actual new simulator evaluations provide
the task-specific integration check.

| Metric | Can | Square |
|---|---:|---:|
| Successes | 12/20 | 3/20 |
| Timeouts | 8/20 | 17/20 |
| Wilson 95% interval | 38.7%–78.1% | 5.2%–36.0% |
| Final training loss | 0.5102 | 0.3767 |
| Final sampled validation loss | 0.4805 | 0.4566 |
| Mean decision latency | 3.18 ms | 3.22 ms |
| Evaluation peak process RSS | 869.2 MiB | 858.7 MiB |

Can successes: seeds 30000, 30001, 30004, 30005, 30006, 30007, 30009, 30011,
30012, 30013, 30014 and 30015. Square successes: 30008, 30011 and 30013.
Total new evaluation interactions: **12,729**. No online training interactions,
V fitting or Q-gated RL were run. Timings are session wall times with independent
Can/Square jobs running concurrently, not a controlled hardware benchmark.

## Interpretation and next step

Can completes the task in 12/20 evaluations; Square remains weak at 3/20 despite
its decreased training/validation loss. Loss reduction is not evidence of
reliable closed-loop control. Failure-stage diagnosis has not yet been done,
so no unverified grasping or insertion failure mechanism is claimed.
These are one training seed and only 20 episodes per task. The same seed numbers
across tasks specify task-specific resets, not identical physical states.

Before using Square to test Q-estimation, inspect its recorded failure episodes
and improve/validate BC with validation-based checkpoint selection and additional
held-out evaluation seeds. Keep this fixed 10,000-update result as a baseline.
Any future B/C comparison must share the same frozen checkpoint within each task.
Larger BC sweeps, GMM online reference reproduction, V fitting and RL remain
unlaunched. The MuJoCo 3.2.6 versus upstream 3.1.6 deviation is unchanged.

## Exact commands

```bash
source scripts/activate.sh
python -m flowq.runner train-bc --config configs/can_bc_stage1.json \
  --out runs/can_bc_stage1_retrained/bc
python -m flowq.runner train-bc --config configs/square_bc_stage1.json \
  --out runs/square_bc_stage1_retrained/bc
python scripts/rollout_bc.py --config configs/can_bc_stage1.json \
  --flow runs/can_bc_stage1/bc/flow.pt --episodes 20 --seed 30000 \
  --out runs/can_bc_stage1/eval_repeat
python scripts/rollout_bc.py --config configs/square_bc_stage1.json \
  --flow runs/square_bc_stage1/bc/flow.pt --episodes 20 --seed 30000 \
  --out runs/square_bc_stage1/eval_repeat
```

The environment and original Lift implementation details follow.

# Lift BC details (earlier completed evaluation)

A loadable state-conditioned single-step Flow-Matching BC checkpoint achieved
**12 successes in 20 closed-loop Lift simulator episodes (60%)**. Can and Square were subsequently trained and evaluated as detailed above.
No V fitting, Q-estimation experiment or online learning has been launched
for these delivered checkpoints.

## Artifacts and commands

| Artifact | Path |
|---|---|
| BC checkpoint, frozen by loader | `runs/lift_bc_stage1/bc/flow.pt` |
| Training configuration | `configs/lift_bc_stage1.json` |
| Training log and complete data manifest | `runs/lift_bc_stage1/bc/bc.json` |
| Standalone rollout script | `scripts/rollout_bc.py` |
| Actual evaluation results | `runs/lift_bc_stage1/eval/summary.json` |
| State/action/reward evaluation records | `runs/lift_bc_stage1/eval/episode_000.npz` through `episode_019.npz` |
| Dependency closure | `requirements.lock.txt` |
| Test log | `artifacts/checks/stage1_unit.log` |
| Portable archive | `deliverables/lift_flow_bc_stage1.zip` |

Checkpoint SHA256:
`c1248fd894245be436f38f565462f0183dcaa14d004eaaa2e55820dbfcd57453`.
Model fingerprint before and after evaluation:
`50f666b733493982c5692f6f55afb0062fceba472d88ded915eb0d6902451e51`.

```bash
source scripts/activate.sh
python scripts/prepare_data.py --tasks lift --checkpoints
python scripts/audit_reference_checkpoint.py data/reference/lift/gmm.pth \
  --out artifacts/reference_lift.json
python -m flowq.runner train-bc --config configs/lift_bc_stage1.json \
  --out runs/lift_bc_stage1_retrained/bc
python scripts/rollout_bc.py --config configs/lift_bc_stage1.json \
  --flow runs/lift_bc_stage1/bc/flow.pt \
  --episodes 20 --seed 30000 --out runs/lift_bc_stage1/eval_repeat
```

Use new output directories. Evaluating the delivered checkpoint requires no
dataset download. `load_flow` restores the architecture and normalization,
calls `eval()` and disables every parameter gradient. The file contains model
state, not optimizer state for exact training continuation.

## Verified data and reference

Only original Robomimic demonstrations were used. No NIST/custom environment,
new demonstration, synthetic trajectory or evaluation trajectory entered BC
training. The official implementation audit is pinned to
[commit d565ab09f5ae88546f2dec43445fe8701ee3de65](https://github.com/rai-opensource/q2rl/commit/d565ab09f5ae88546f2dec43445fe8701ee3de65).
`reference/q2rl/` is an audit snapshot, not a complete executable clone. This
workspace has no usable Git repository; source hashes provide local identity.

Networking now works. Official Lift/Can/Square PH v1.4.1 HDF5s and released GMM
checkpoints were downloaded and checked against HF LFS SHA256. The prior missing
assets/network blocker is resolved. Data repository:
`theaiinstitute/q2rl_robomimic_datasets`, revision
`ba3b72ac9f372634a1b7293f140bc3bd0f3acf89`.

| Task | Dataset path | Verified BC masks | Train / validation demos | Train / validation steps |
|---|---|---|---:|---:|
| Lift | `lift/ph/low_dim_v141.hdf5` | `20_percent_train` / `20_percent_valid` | 36 / 4 | 1,769 / 223 |
| Can | `can/ph/low_dim_v141.hdf5` | `train` / `valid` | 180 / 20 | 20,883 / 2,324 |
| Square | `square/ph/low_dim_v141.hdf5` | `train` / `valid` | 180 / 20 | 27,165 / 2,989 |

Masks are verified from checkpoint-embedded training configs, not inferred from
replay masks. Audits: `artifacts/reference_{task}.json`. Each file contains 200
demonstrations; only the selected masks are loaded. Exact demo memberships and
observation shapes are recorded in the trained checkpoint's manifest.

Lift GMM revision: `7c31582498d6211c00a1fa2cccefc50b73f6ed71`;
GMM SHA256: `0f3b8a87777c9ce25b36c41e9bbf3faf1567d81c7e5ee7493c06a0657da1293d`.
Lift HDF5 SHA256:
`e11f940bf6e70573e371699aa0c615094b6aeeaaf3ffd60f8777a4c68573b50f`.

The current [Robomimic dataset documentation](https://robomimic.github.io/docs/datasets/robomimic_v0.1.html)
distinguishes its v1.5.1 conversion from the earlier v1.4.1 datasets. We kept
Q2RL's v1.4.1 files with no migration. Train and validation masks are disjoint.

## Model and simulator conventions

State has 19 values: `robot0_eef_pos` (3), `robot0_eef_quat` (4, xyzw),
`robot0_gripper_qpos` (2), `object` (10), in that order. Live `object-state`
corresponds to dataset `object`. Mean and sample std are fitted only to training
transitions; std floor is 0.02. Although the environment adapter appends time
remaining for future RL use, Flow BC slices the physical 19 values and does not
condition on time. No images, temporal history or action chunks are used.

Six-dimensional arm flow matching uses independent standard Gaussian noise,
linear interpolation targets, and a two-layer 128-wide SiLU velocity network.
Its smooth output is `6*tanh(raw/6)`. RK4 integrates t=0 to t=1 in 16 steps;
tanh then maps to normalized controller commands. Training targets use
`atanh(0.999*action)` to handle demonstrated boundary commands. No sampled action
is clipped or replaced.

A separate categorical gripper head has support `{-1,0,+1}`, matching
PandaGripper's sign operation. This is a **hybrid flow policy**: arm flow and
gripper are conditionally independent given state. Future log density must
include both the tanh Jacobian and gripper probability mass, not claim a
seven-dimensional continuous density after gripper discretization.

Original metadata supplies Panda OSC_POSE, 20 Hz, relative pose control,
translation scale ±0.05 m and rotation scale ±0.5 rad for normalized ±1 commands.
Each policy output drives one control step. Evaluation terminates at actual
`env._check_success()` or 300 steps and records sparse raw reward. There is no
Q gate, action chunk or external controller in the evaluated BC.

Validated environment: Python 3.10.22, torch 2.5.1+cpu, numpy 1.26.4,
robosuite 1.4.1, MuJoCo 3.2.6, CPU only. Original HDF5 is read directly; no Robomimic
wrapper is installed or mixed. Q2RL instead pins MuJoCo 3.1.6 and vendors
Robomimic 0.3.0. This simulator difference remains explicit; no bit-for-bit
reference reproduction is claimed.

## Actual training and closed-loop results

Training seed 0; all 36 reference Lift training-mask demos; batch 256; Adam 0.0003;
10,000 updates; final checkpoint fixed by this update budget before evaluation.
CPU training time was **27.07 s**. Evaluation reset seeds were 30000–30019;
noise generators used `environment_seed + 100000`. Raw state/action records
and initial observation hashes are saved per episode.

| Metric | Actual result |
|---|---:|
| Success | **12/20 = 60%** |
| Approximate Wilson 95% interval | 38.7%–78.1% |
| Normal timeouts | 8/20 |
| Simulator interactions | **3,007** |
| Mean successful episode length | 50.58 steps |
| Mean action generation latency | 3.29 ms CPU |
| Evaluation elapsed time | 27.85 s |
| Evaluation process peak RSS | 875.9 MiB |
| Parameter changes / optimizer updates in evaluation | **0 / 0** |

Success seeds: 30000, 30002, 30003, 30005, 30007, 30009, 30010, 30012, 30013,
30014, 30015 and 30017. All 20 episodes completed with finite observations,
actions and rewards, without action saturation. Four unit tests passed in
4.14 s, including an OOD action-saturation regression.

Final training loss was 0.2027; sampled validation FM loss was 0.9549. Validation
loss was lower earlier (0.3288 at update 3001), indicating overfitting. Reported
success is measured in simulation, not inferred from training loss. No
checkpoint or evaluation seed was discarded to inflate this result. One
training seed and 20 episodes are insufficient for a robust benchmark conclusion,
and establish no Q-estimation improvement.

## Diagnostic history and stop boundary

Before scope changed, a ten-demo Lift debug BC and all-train-demo Square BC
received 1,000 updates each. The Lift debug policy drifted OOD and hit tanh
floating-point saturation at step 174 of its first attempted initial rollout.
That attempt did not create warmup.pt or fit V. The delivered Lift model uses
all 36 reference training demos, a smooth bounded velocity and a larger std
floor. These are BC implementation changes, not Q-estimation improvements.
The earlier debug artifacts are retained, not presented as successful training.

Square's earlier 1,000-update BC and density diagnostic remain separate from
the new evaluated Square checkpoint. Historical random-policy simulator checks
under `artifacts/prior_20261027/checks/` are plumbing diagnostics. Can and Square
now have the trained-policy BC evaluations summarized at the top of this report.

The earlier detailed Q2RL audit and stage-2 commands are preserved in
`artifacts/prior_20261027/REPORT.md` and its README. Their old network-blocker
status is superseded. The current work remains BC only. If the user resumes stage 2,
use this exact frozen flow for matched B/C and count initial rollout steps.
