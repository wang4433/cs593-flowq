# Dual RL small test plan

Staged checks for the dual-RL code (`flowq/dual.py`, see [DUAL.md](DUAL.md)) on Can,
from unit tests to one small online run. Each stage is short and gates the next one.
Nothing here has been run yet: the dual code is only syntax-checked.

Stage names are T0–T4 to avoid confusion with the report's condition letters and the
DSRL small-test stages.

## Before starting

- **Flow:** the existing DSRL small runs (`runs/dsrl_small_eideas3/`) used
  `runs/bc_local/can/bc/flow.pt` (sha256 `af73047…`). Use the same flow for every
  run below so dual and DSRL-only results are comparable.
- **Replay:** those runs trained from a 50-episode warmup replay (sha256 `0ae56a7…`)
  that is **not in the repo** (`B_warmup/replay.npz` is missing). Either ask for that
  file, or create a new one in T1 and use it for both the dual and DSRL-only runs.
- **Hardware:** the existing DSRL runs ran on an RTX 5080 (torch 2.8 + CUDA). The
  Gilbreth `cs593` env has CPU-only torch, so `--device auto` runs on CPU there. Results
  from different machines are not bit-identical; compare dual vs DSRL-only **on the same
  machine and the same replay**.
- **Never run training on a `gilbreth-fe*` login node.** Submit with `sbatch`. The
  `rpaleja` account requires `--gres=gpu:1` even for CPU-only jobs.

```bash
ROOT=/scratch/gilbreth/hmoondra/cs593_project/cs593-flowq
PY=$ROOT/.conda/envs/cs593/bin/python
FLOW=$ROOT/runs/bc_local/can/bc/flow.pt
PART=a100-40gb   # or a30
run_gpu () {  # usage: run_gpu <job-name> <hours> <command ...>
  local name=$1 hours=$2; shift 2
  sbatch -A rpaleja -p $PART --gres=gpu:1 --cpus-per-task=4 --mem=16G --time=${hours}:00:00 \
    --job-name=$name --output=$ROOT/logs/%x-%j.out --error=$ROOT/logs/%x-%j.err \
    --wrap="cd $ROOT && export PYTHONPATH=$ROOT NUMBA_CACHE_DIR=$ROOT/.cache/numba MUJOCO_GL=egl && $*"
}
```

## Stages

| Stage | What it tests | Approx. time (CPU) |
|---|---|---|
| T0. Unit tests | Code invariants, including the 4 new dual tests | 2 min |
| T1. Warmup + identity | Untrained dual learner = BC exactly on all three arms; writes a reusable replay | 10 min |
| T2. Offline dual | Flow and steering trained from replay only; first look at what each part adds | 10–15 min |
| T3. Online dual | 5,000 online steps; health of both learners; comparison with DSRL-only | 45–60 min |
| T4. (optional) ablations | Sensitivity to `residual_bound`, `flow_tau`, `demo_weight` | 1 run each |

Times are estimates: the flow update adds roughly a third to the cost of a plain DSRL
update, since it runs every 5 updates.

### T0. Unit tests

```bash
run_gpu dual_T0 1 "$PY -m unittest discover -s tests -v"
```

Pass: `Ran 19 tests` (15 existing + 4 dual) and `OK`.

`test_flow_update_tilts_prior_toward_rewarded_mode` prints the rewarded-mode fraction
before and after. Its thresholds (>0.7 and a gain >0.15) are untested guesses. If only
that test fails, look at the printed numbers before concluding anything: a clear upward
move with a lower final value means the threshold needs adjusting, not the method.

### T1. Warmup replay + identity check

50 BC warmup episodes, one critic update, no actor or flow updates, then evaluation.
The correction is still zero and the noise policy is still N(0, I), so **all three arms
must be identical** episode by episode.

```bash
run_gpu dual_T1 1 "$PY -m flowq.dsrl_runner train --config configs/can_dual_small.json \
  --flow $FLOW --out runs/dual_small/T1_warmup \
  --critic-warmup-updates 1 --pretrain-updates 0 --online-steps 0 --eval-episodes 5"
```

Check:

- `summary.json`: `flow_updates` = 0, `updates` = 1.
- `evaluations.json`: `bc`, `reshaped` and `dsrl` entries are equal.
  ```bash
  python3 -c "import json;e=json.load(open('runs/dual_small/T1_warmup/evaluations.json'))[0];print('IDENTICAL' if e['bc']==e['reshaped']==e['dsrl'] else 'DIFFERENT')"
  ```
- `warmup.json`: roughly 25–35 successes out of 50 for Can.
- `replay.npz` exists. T2–T4 reuse it.

### T2. Offline dual

Train from the T1 replay only: 2,000 critic-only and 2,000 joint updates, no new
environment steps. The flow updates during the 2,000 joint updates.

```bash
run_gpu dual_T2 1 "$PY -m flowq.dsrl_runner train --config configs/can_dual_small.json \
  --flow $FLOW --replay runs/dual_small/T1_warmup/replay.npz \
  --warmup-episodes 0 --online-steps 0 --out runs/dual_small/T2_offline"
```

Check: `flow_updates` = **400**, `updates` = 4,000. Then compare the three arms (see
"What to check").

### T3. Online dual

Same replay, then 5,000 online steps at 5 updates per step.

```bash
run_gpu dual_T3 2 "$PY -m flowq.dsrl_runner train --config configs/can_dual_small.json \
  --flow $FLOW --replay runs/dual_small/T1_warmup/replay.npz \
  --warmup-episodes 0 --out runs/dual_small/T3_online"
```

Check: `flow_updates` = **5,400**, `updates` = 29,000. Evaluations at 2,500 and 5,000
online steps.

**Matched DSRL-only run** (your decision, not run automatically). It is the
reference for "does reshaping help on top of steering". It uses the same flow, replay,
seed, budget and machine as T3:

```bash
run_gpu dsrl_T3_ref 2 "$PY -m flowq.dsrl_runner train --config configs/can_dsrl_small.json \
  --flow $FLOW --replay runs/dual_small/T1_warmup/replay.npz \
  --warmup-episodes 0 --out runs/dual_small/T3_dsrl_only"
```

Alternatively, if the original `B_warmup/replay.npz` is recovered, run T3 with that
replay on the same machine as the existing `D_online` run and compare with it directly.

### T4. (optional) Ablations

Change one value at a time from `configs/can_dual_small.json`. The dual keys have no
command-line overrides, so save each variant as its own config file, e.g.
`configs/can_dual_small_r025.json`:

| Variant | Change | Question |
|---|---|---|
| No demo anchor | `demo_weight: 0` | Does the anchor matter? |
| Tighter / looser bound | `residual_bound: 0.25` or `1.0` | How much reshaping is useful? |
| Sharper / softer tilt | `flow_tau: 0.5` or `2.0` | Over-sharpening vs no effect |

## What to check

### Correctness (every stage)

| Check | Where | Pass |
|---|---|---|
| BC flow unchanged | `summary.json` → `frozen_unchanged` | `true` (the run also asserts it) |
| Flow update count | `summary.json` → `flow_updates` | T1: 0, T2: 400, T3: 5,400 |
| Learner update count | `summary.json` → `updates` | T1: 1, T2: 4,000, T3: 29,000 |
| Three paired arms | `evaluations.json` | `bc`, `reshaped`, `dsrl` present; `bc` identical at every evaluation |
| Checkpoint reload | `python -m flowq.dsrl_runner eval --checkpoint …/dsrl.pt --flow $FLOW …` | Loads as dual (schema `flowq.dsrl.v3`) without error |
| No crashes | job `.err` | No `Traceback`, `FloatingPointError` or `nonfinite` |

### Flow learner health (`updates.jsonl`, T2–T4)

These ranges are heuristics for spotting problems, not hard limits.

| Field | Healthy | Red flag |
|---|---|---|
| `flow_weight_ess` (1 to 8) | Between, e.g. 2–6 | Pinned near 1: tilt too sharp, raise `flow_tau`. Pinned at 8: critic can't tell candidates apart, so no reshaping happens |
| `flow_q_spread` | Clearly above 0 | ≈ 0: nothing for the flow to learn yet |
| `flow_displacement_mean` / `_max` | Rises from 0, then levels off; small compared with the action range | Stays ≈ 0 (bound or `flow_lr` too small), or keeps growing (bound too loose) |
| `flow_tilt_loss`, `flow_demo_loss` | Stable | Steadily growing |

The steering learner's fields (`action_critic_loss`, `td_target_mean`,
`latent_mean_abs_max`, `latent_std_min`, `latent_kl`) have the same healthy ranges as in
[DSRL_SMALL_TEST.md](DSRL_SMALL_TEST.md).

### Outcomes (T2–T4)

Each evaluation runs the three arms on the same 20 seeds with the same base noise, so
compare them **per seed**:

| Comparison | Answers |
|---|---|
| `reshaped` vs `bc` | Does reshaping alone help? |
| `dsrl` vs `reshaped` | Does steering help on top of the reshaped flow? |
| T3 `dsrl` vs DSRL-only `dsrl` | Does the combination beat steering alone? |

```bash
python3 - <<'EOF'
import json
for ev in json.load(open('runs/dual_small/T3_online/evaluations.json')):
    bc = ev['bc']['episodes']
    for arm in ('reshaped', 'dsrl'):
        other = ev[arm]['episodes']
        gained = sum(o['success'] and not b['success'] for b, o in zip(bc, other))
        lost = sum(b['success'] and not o['success'] for b, o in zip(bc, other))
        print(ev['training_env_steps'], arm, ev[arm]['success_rate'], 'vs bc', ev['bc']['success_rate'],
              '| gained', gained, 'lost', lost)
EOF
```

With 20 episodes per arm, this shows direction, not statistical significance.

## Go / no-go for longer runs

Move to full budgets (Lift and Can, more seeds) only if:

1. T0 and T1 pass, including the three-arm identity check.
2. Counts and invariants match in T2 and T3, with no crashes.
3. Flow health shows reshaping actually happening: displacement above 0, weight ESS
   below the maximum, and no runaway losses.
4. In T3, `dsrl` is not worse than `bc`. Ideally `reshaped` ≥ `bc`, and T3 `dsrl` ≥ the
   DSRL-only reference.

## Known caveats

- The synthetic-test thresholds in T0 are untested.
- One seed and 20 evaluation episodes per arm.
- The gripper is not reshaped; it always comes from the frozen BC head.
- Training decodes use 8 RK4 steps, execution uses 16 (same as DSRL).
- Results from different machines are not bit-identical; keep each comparison on one machine.
