# DSRL small test plan

A staged, cheap way to check that the DSRL latent-steering code works end to end
and moves in the right direction **before** launching the full budget
(`configs/*_dsrl.json`: 100 warmup episodes, 100k online steps, UTD 20, about
2M learner updates, roughly 1–1.5 days per task).

Nothing here has been run yet. Each stage is short and gates the next one.

## What we already know

From the full run that was started and then cancelled after 16 minutes
(job 12117134, three tasks in parallel on one a30 node):

| Item | Value |
|---|---|
| Unit tests | 15/15 passed |
| Lift warmup | 48/100 successful BC episodes, 18,082 env steps, about 10.5 min |
| Can warmup | 55/100 successful BC episodes, 19,944 env steps, about 13 min |
| Square warmup | 46/100 episodes done after 16 min (not finished) |
| Learner update time | median 0.041 s (Lift and Can), effective about 0.05 s including env steps |

`runs/dsrl_12117134/` holds those partial logs. No `replay.npz` was written,
because replay is only saved when a run finishes.

Cost extrapolation for the full budget: 2,001,500 updates × about 0.05 s is
about 28 h per task, plus warmup and evaluations.

## Prerequisites

- Environment: `.conda/envs/cs593` (see README).
- Frozen BC flows: `runs/verify_flow_12036366/{lift,can,square}/bc/flow.pt`.
- **Never run training on a `gilbreth-fe*` login node.** Submit everything with
  `sbatch` on a30. The `rpaleja` account requires `--gres=gpu:1` even though the
  code is CPU-only.

Helper for the commands below (paste into your shell first):

```bash
ROOT=/scratch/gilbreth/hmoondra/cs593_project/cs593-flowq
PY=$ROOT/.conda/envs/cs593/bin/python
FLOW=$ROOT/runs/verify_flow_12036366/can/bc/flow.pt
run_a30 () {  # usage: run_a30 <job-name> <hours> <command ...>
  local name=$1 hours=$2; shift 2
  sbatch -A rpaleja -p a30 --gres=gpu:1 --cpus-per-task=4 --mem=16G --time=${hours}:00:00 \
    --job-name=$name --output=$ROOT/logs/%x-%j.out --error=$ROOT/logs/%x-%j.err \
    --wrap="cd $ROOT && export PYTHONPATH=$ROOT NUMBA_CACHE_DIR=$ROOT/.cache/numba MUJOCO_GL=egl CUDA_VISIBLE_DEVICES='' && $*"
}
```

## Small config

`eval_interval` and `log_stride` cannot be overridden from the command line, so
the small test needs its own config. Save this as `configs/can_dsrl_small.json`
(identical to `can_dsrl.json` except for the budget fields):

```json
{
  "bc_config": "can_bc_stage1.json",
  "dsrl": {
    "seed": 0,
    "eval_seed": 40000,
    "hidden": 128,
    "batch_size": 128,
    "lr": 0.0003,
    "tau": 0.005,
    "grad_clip": 10.0,
    "kl_beta": 3.0,
    "prior_fraction": 0.5,
    "flow_steps": 16,
    "train_flow_steps": 8,
    "actor_mean_bound": 2.0,
    "replay_capacity": 50000,
    "warmup_episodes": 50,
    "critic_warmup_updates": 2000,
    "pretrain_updates": 2000,
    "online_steps": 5000,
    "utd": 5,
    "eval_episodes": 20,
    "eval_interval": 2500,
    "log_stride": 50
  }
}
```

**Why Can:** BC succeeds about 55–60% of the time, so warmup data contains
plenty of +4 rewards, and there is room to improve or degrade. Lift's BC
overfits (36 demos). Square's 500-step horizon and 15% BC success make it too
slow and too sparse for a small test.

## Stages

| Stage | What it tests | Approx. time |
|---|---|---|
| A. Unit tests | Code invariants | 1 min |
| B. Warmup + identity | Pipeline wiring; untrained actor equals BC exactly; produces a reusable replay | 8–10 min |
| C. Offline only | Can the critics and actor improve on BC from BC rollouts alone? | 6–10 min |
| D. Small online | Does online steering help, and do the critics and actor stay healthy? | 30–40 min |
| E. (optional) mini KL sweep | Sensitivity to `kl_beta` on identical data | 35–45 min (3 runs in one job) |

Times are estimates from the measured warmup and update speeds above.

### A. Unit tests

```bash
run_a30 dsrl_tests 1 "$PY -m unittest discover -s tests -v"
```

Pass: the `.err` log ends with `Ran 15 tests` and `OK`.

### B. Warmup replay + identity check

Collect 50 BC episodes once, do one critic update (the runner requires at least
one update), never train the actor, and evaluate. Because the actor is still
exactly N(0, I) and evaluation shares noise between BC and DSRL, the two
evaluations must be **identical**.

```bash
run_a30 dsrl_B 1 "$PY -m flowq.dsrl_runner train --config configs/can_dsrl_small.json \
  --flow $FLOW --out runs/dsrl_small/B_warmup \
  --critic-warmup-updates 1 --pretrain-updates 0 --online-steps 0 --eval-episodes 5"
```

Check:

- `runs/dsrl_small/B_warmup/warmup.json`: about 25–30 successes out of 50 is
  expected. Fewer than 10 means the critic will see almost no positive reward.
- `evaluations.json`: the `bc` and `dsrl` entries are equal (same successes,
  steps and returns for every seed).
  ```bash
  python3 -c "import json;e=json.load(open('runs/dsrl_small/B_warmup/evaluations.json'))[0];print('IDENTICAL' if e['bc']==e['dsrl'] else 'DIFFERENT')"
  ```
- `replay.npz` exists. Stages C–E reuse it, so they share exactly the same
  warmup data.

### C. Offline-only improvement

Train both critics and the actor on the Stage B replay with no new environment
interaction, then evaluate. This is a one-step improvement over BC.

```bash
run_a30 dsrl_C 1 "$PY -m flowq.dsrl_runner train --config configs/can_dsrl_small.json \
  --flow $FLOW --replay runs/dsrl_small/B_warmup/replay.npz --out runs/dsrl_small/C_offline \
  --warmup-episodes 0 --online-steps 0"
```

The interesting result is whether DSRL is at least as good as BC here. A clear
drop below BC means the critic is being exploited off the data.

### D. Small online run

Start from the same replay, run the 4,000 offline updates, then 5,000 online
steps at UTD 5 (25,000 updates). Evaluations run at 2,500 and 5,000 online steps.

```bash
run_a30 dsrl_D 2 "$PY -m flowq.dsrl_runner train --config configs/can_dsrl_small.json \
  --flow $FLOW --replay runs/dsrl_small/B_warmup/replay.npz --out runs/dsrl_small/D_online \
  --warmup-episodes 0"
```

### E. (optional) Mini KL sweep

Same as D for `kl_beta` 1, 3 and 10, run in parallel inside one job:

```bash
run_a30 dsrl_E 2 "for b in 1 3 10; do $PY -m flowq.dsrl_runner train \
  --config configs/can_dsrl_small.json --flow $FLOW \
  --replay runs/dsrl_small/B_warmup/replay.npz --warmup-episodes 0 \
  --kl-beta \$b --out runs/dsrl_small/E_beta\$b > runs/dsrl_small/E_beta\$b.log 2>&1 & done; wait"
```

## What to check

### Correctness invariants (every stage)

| Check | Where | Pass |
|---|---|---|
| Frozen flow unchanged | `summary.json` → `frozen_unchanged` | `true` (the run also asserts this) |
| Update count | `summary.json` → `updates` | `critic_warmup + pretrain + online_steps × utd` |
| Replay size | `summary.json` → `retained_replay_transitions` | `imported + warmup steps + online steps` |
| No crashes | job `.err` / run log | No `Traceback`, `FloatingPointError` or `nonfinite` |
| Paired evaluation | `evaluations.json` | BC results identical at every evaluation point (BC is deterministic under fixed seeds) |

### Learner health (`updates.jsonl`, stages C–E)

These thresholds are heuristics for spotting problems, not hard limits.

| Field | Healthy | Red flag |
|---|---|---|
| `action_critic_loss` | Falls during critic warmup, then settles | Keeps growing, or spikes >10× |
| `td_target_mean` | Inside [−100, 4], drifting slowly | Outside that range, or trending steadily up or down |
| `teacher_mean` | Close to the action critic's value scale | Far from `td_target_mean` |
| `noise_critic_loss` | Well below `action_critic_loss` after warmup | Comparable or growing (distillation failing) |
| `latent_mean_abs_max` | Rises from 0, below the bound 2 | Pinned at 2.0 for long stretches (KL too weak, or the critic is being exploited) |
| `latent_std_min` / `latent_std_max` | Between about 0.1 and 1 | `std_min` near 0.007 (e⁻⁵) = collapse |
| `latent_kl` | 0 at start, grows to a moderate value | Large and still growing late in training |
| `update_seconds` | About 0.04 | Much slower; check node contention |

### Outcome (`evaluations.json`, stages C–E)

- Success rate and mean discounted return for `bc` vs `dsrl` at each evaluation.
- Because BC and DSRL use the **same reset seeds and the same base noise**,
  compare them **per seed**. Count the seeds where DSRL succeeds and BC fails
  (n₁₀) and the reverse (n₀₁).
  ```bash
  python3 - <<'EOF'
  import json
  for ev in json.load(open('runs/dsrl_small/D_online/evaluations.json')):
      bc, ds = ev['bc']['episodes'], ev['dsrl']['episodes']
      n10 = sum(d['success'] and not b['success'] for b, d in zip(bc, ds))
      n01 = sum(b['success'] and not d['success'] for b, d in zip(bc, ds))
      print(ev['training_env_steps'], 'BC', ev['bc']['success_rate'], 'DSRL', ev['dsrl']['success_rate'], 'n10', n10, 'n01', n01)
  EOF
  ```
  With 20 paired episodes, an exact sign test only reaches p < 0.05 at splits
  like 6–0, 8–1 or 10–2. The small test shows **direction**, not significance.
- Online episode success over training: `summary.json` → `episodes`, entries
  with `"stage": "online"`.

### Checks not implemented yet (worth adding)

- **Critic calibration:** compare `Q^A(s, a)` on the Stage B replay with the
  discounted Monte Carlo return computed from the same episodes (they are stored
  in order and delimited by `terminals`). A large systematic gap means the
  critic, and therefore the steering, is unreliable.
- **Distillation agreement:** on held-out states and fresh `z ~ N(0, I)`,
  compare `Q^W(s, z)` with `Q^A(s, decode(s, z))`.
- **Where the actor moved:** decode the actor's mean noise vs. prior samples at
  fixed states and see which BC mode it selects.

## Go / no-go for the full run

Launch the full budget (`scripts/submit_dsrl.sh`) only if:

1. Stages A and B pass, including the identity check.
2. No crashes, and the learner-health table shows no persistent red flags.
3. In C and D, DSRL is not worse than BC (n₀₁ not clearly larger than n₁₀), and
   preferably moves upward with training.
4. `mean_update_seconds` × 2,001,500 updates is an acceptable wall time.

If D looks flat, try Stage E before scaling up. A flat result at this scale is
not evidence the method fails; it is 5,000 online steps instead of 100,000.

## Known caveats

- One seed. Repeat promising settings with other `--seed` values.
- Critic targets decode with 8 RK4 steps while executed actions use 16
  (`--train-flow-steps 16` removes the mismatch).
- The std cap passes no gradient once the raw log-std exceeds 0, so the std can
  stay at 1 for some states. That is the safe direction (the BC prior).
- Imported replay transitions are real environment interactions collected in
  Stage B. Count them in the interaction budget when comparing methods, even
  though `training_env_steps` in later stages excludes them.
