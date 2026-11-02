# Frozen-flow steering with action-to-noise critic distillation

This is a DSRL-inspired research implementation in `flowq/`, based on
[DSRL-NA, Algorithm 1](https://arxiv.org/html/2506.15799v2#S4.SS2).
The existing A/B/C pipeline and trained BC checkpoints remain usable. No code
from the upstream DSRL repository was copied. This is not an exact reproduction
of its architecture, objectives, or published results.

## Flow review before RL

Reviewed the training interpolation/target, observation normalization, RK4,
reverse-time change of variables, tanh Jacobian, categorical gripper, and
interaction between the flow and the finite-horizon environment.

- The flow-matching target `x1-x0` matches the linear interpolation from Gaussian
  noise to `atanh((1-epsilon)*action)`. No sign error was found.
- The backward density accumulator and tanh Jacobian pass the analytic Gaussian
  density/entropy/convergence tests. Exact divergence means the trace is exact;
  finite-step ODE integration is still approximate.
- `rk4` deliberately detaches its output at each step. It is an inference
  integrator, not a differentiable actor-training integrator. This is compatible
  with DSRL-NA because actor gradients pass through a learned noise critic.
- Fixed missing finite-input and batch/action-shape validation in `Flow.loss`.
  Previously NaNs could bypass the `abs(action)>1` check and reach the loss.
- Corrected the velocity-bound comment: bounded drift reduces saturation risk
  but cannot rule it out for unbounded Gaussian/learned noise. Saturation still
  raises explicitly; no clipping or silent resampling was introduced.
- Added `decode(s,z)` and `encode(s,arm)` for continuous arm commands only.
  `action_from_noise` separately samples the categorical gripper. Existing
  `sample` preserves the original RNG draw order and seeded outputs.
- Training's inward boundary smoothing is a deliberate model convention, not an
  exact density on raw boundary demonstrations. DSRL never queries that density.
- BC uses physical observations only. RL additionally uses time remaining, so
  success and horizon are terminal for the existing finite-horizon objective.
  Reward remains `5*raw_reward-1`: -1 ordinarily, +4 on success.

The existing BC performance limitations (particularly Square) and final-iterate
checkpoint selection are not resolved by this change. No BC retraining or
validation-checkpoint sweep is part of this RL implementation.

## Learning objective

Let `x(s)` be normalized physical state plus remaining time, `f(s,z)` the frozen
six-dimensional arm decoder, and `p(g|s)` the frozen categorical gripper.

1. **Action critic:** twin networks `QA(x,a)` learn task returns from real
   `(s,a,r,s',done)` transitions. The next latent is sampled from the actor;
   the next arm is decoded, and all three gripper outcomes are averaged:

   `y = r + gamma*(1-done)*E_g[min_i QA_target_i(x', (f(s',z'),g))]`.

   Terminal states are not decoded at all. No entropy or KL term is included
   in this task-return backup.
2. **Noise critic:** draw fresh latent proposals at replay states, decode them,
   and regress both `QZ(x,z)` heads onto

   `teacher = E_g[min_i QA_i(x, (f(s,z),g))]`.

   The teacher and flow are detached. Proposals mix the Gaussian prior and the
   current actor; `prior_fraction=1` uses the prior-only sampling in Algorithm 1.
   The supplied configs use 0.5 to also supervise the region the actor visits.
3. **Noise actor:** a diagonal Gaussian initialized to exactly `N(0,I)` minimizes

   `E_z[-min_i QZ_i(x,z)] + kl_beta * KL(N(mu,sigma^2) || N(0,I))`.

   The KL is analytic. The critic parameters are frozen for this gradient step,
   but its derivatives with respect to z remain available. Only six arm-noise
   dimensions are steered. No actor gradients pass through the flow. This is
   regularized policy improvement using a reward critic, not a full soft-SAC
   Bellman objective. `kl_beta=0` removes the prior penalty.

   New actors use `mu = actor_mean_bound*tanh(raw_mu)` (default bound 2) and
   `sigma = exp(clamp(raw_log_std, -5, 0))`, capping std at 1. The Gaussian
   samples themselves are still unbounded; no sample clipping is introduced.
   These bounds complement KL regularization rather than replacing it.

Critic targets and noise-critic distillation use `train_flow_steps=8` RK4 steps;
executed actions in warmup, online interaction, and evaluation use
`flow_steps=16`. This halves the number of velocity evaluations in each training
decode, not necessarily total update time. It introduces a numerical mismatch
between the teacher's decoder and the executed decoder. Prior held-out checks
found roughly 1e-3 or smaller maximum action errors at 8 steps versus 64; that
is not a uniform guarantee over future learned noise or encountered states.
Use `--train-flow-steps 16` to remove the step-count mismatch in an ablation.

**Aliasing qualification:** an ideal well-posed ODE followed by tanh is injective
in the arm noise. Distinct latent points need not have exactly identical actions.
The action-to-noise distillation is still applicable and may share value
information between nearby behaviors; its sample-efficiency benefit here must
be measured, not assumed from exact many-to-one aliases. Steering also does not
guarantee that every decoded action is supported by demonstrations.

## Running with the existing checkpoints

From the repository root, use the existing environment or the README setup:

```bash
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export NUMBA_CACHE_DIR="$PWD/.cache/numba"
export MUJOCO_GL=egl
PY=.conda/envs/cs593/bin/python

$PY -m flowq.dsrl_runner train \
  --config configs/lift_dsrl.json \
  --flow runs/verify_flow_12036366/lift/bc/flow.pt \
  --out runs/lift_dsrl_seed0
```

Use `can_dsrl.json` / `square_dsrl.json` and their corresponding checkpoints for
the other tasks. Each config references its BC config relative to the config
file, preserving preprocessing and the original training masks. Run commands
from the repository root. DSRL settings are in a separate `dsrl` section;
old A/B/C settings do not control the new training loop.

Defaults are **experiment starting settings**, not tuned hyperparameters:
100 BC warmup episodes, 500 critic-only updates, 1,000 updates including the
latent actor, and 100,000 online steps. Each online step adds one executed
transition and performs **20 updates**, for 2,001,500 learner updates overall.
The actor mean bound is 2 and `kl_beta=3`. Replay capacity is 200,000, sufficient
to retain all new transitions for these defaults on all three tasks.

Warmup interactions count toward the reported training budget; evaluation
interactions are separate. Evaluation runs every 10,000 online steps, with
50 episodes for **each** of BC and DSRL. More warmup increases the opportunity
to observe successes, but does not guarantee any particular number. Inspect
`warmup.json` for the actual success count and positive-reward transitions.
Logs include actor mean/std ranges, KL, and update duration; `summary.json`
includes total and mean learner update time for estimating larger-run cost.

The old `total_steps` field belongs to the A/B/C runner; DSRL's maximum training
interaction count is `warmup_episodes*horizon + online_steps`.

Sweep KL while holding the training seed and other settings fixed (these are
full experiments, not smoke checks):

```bash
for beta in 1 3 10; do
  $PY -m flowq.dsrl_runner train \
    --config configs/square_dsrl.json \
    --flow runs/verify_flow_12036366/square/bc/flow.pt \
    --kl-beta "$beta" --seed 0 \
    --out "runs/square_dsrl_beta${beta}_seed0"
done
```

The resolved CLI settings are saved in each run's config and checkpoint.
Repeat comparisons with additional training seeds before drawing conclusions.
No full sweep or expanded-budget training was launched as part of this fix.

For a bounded implementation smoke check (not a performance experiment):

```bash
$PY -m flowq.dsrl_runner train \
  --config configs/lift_dsrl.json \
  --flow runs/verify_flow_12036366/lift/bc/flow.pt \
  --out runs/my_dsrl_smoke \
  --warmup-episodes 1 --online-steps 4 \
  --critic-warmup-updates 2 --pretrain-updates 2 --eval-episodes 1 --utd 1
```

Outputs include `config.json`, `updates.jsonl`, `evaluations.json`, `summary.json`,
`dsrl.pt`, and `replay.npz`. The checkpoint contains the actor, both critics,
target critic, optimizers, update count, configuration, and frozen BC identity.
New checkpoints use schema `flowq.dsrl.v2`. The loader also supports old v1
checkpoints, retaining their original unbounded mean and std cap of e rather
than silently changing their policy. Existing v1 results are historical results
for that old parameterization; retraining is required to assess the bounded actor.
It supports reload/evaluation; exact interrupted-run continuation (environment,
replay/RNG restoration) is not implemented. Intermediate checkpoints are saved
at evaluations. Existing output directories are rejected.

```bash
$PY -m flowq.dsrl_runner eval \
  --checkpoint runs/lift_dsrl_seed0/dsrl.pt \
  --flow runs/verify_flow_12036366/lift/bc/flow.pt \
  --episodes 50 --seed 50000 --out runs/lift_dsrl_seed0/heldout_eval.json
```

Both methods use matched reset seeds and separate sampling generators. The
default evaluates the **stochastic learned Gaussian**, so an unchanged actor
recovers BC. `--mean-noise` is a separately labeled diagnostic; it is not BC in
the large-KL limit. Evaluation preserves training RNGs/model state, runs in
separate environments, and never appends data to replay. Use fresh evaluation
seeds for final reporting rather than selecting a model on those outcomes.

## Action-only replay and offline-to-online use

The new learner does not need an inverse flow or known latent labels. Warmup and
online training export raw state/action transitions; optionally recorded z is
diagnostic only. Train from a previous **training** replay with:

```bash
$PY -m flowq.dsrl_runner train \
  --config configs/lift_dsrl.json \
  --flow runs/verify_flow_12036366/lift/bc/flow.pt \
  --replay runs/lift_dsrl_seed0/replay.npz \
  --warmup-episodes 0 --online-steps 0 --pretrain-updates 1000 \
  --out runs/lift_dsrl_offline
```

Set `online_steps` above zero to follow replay pretraining with online learning.
Replay sampling is uniform across retained transitions; it does not maintain a
fixed demonstration/online ratio. The archive schema is:

- `states`, `next_states`: `[N, physical_obs_dim+1]`, **unnormalized** physical
  observations plus `1-t/horizon`.
- `actions`: `[N,7]`, six executed arm commands in `[-1,1]`, gripper in `{-1,0,1}`.
- `rewards`, `terminals`: `[N]`, rewards already transformed using the environment
  convention; terminal at success, environment termination, or finite horizon.
- `metadata`: JSON with schema `flowq.dsrl.replay.v1`, purpose `training`, and
  the exact dictionary produced by `replay_contract(cfg, flow)`.
- `noise: [N,6]` and `noise_known: [N]` are optional and ignored by learner updates.

`ActionReplay.add` / `.save` provide the programmatic construction interface.
The loader checks task, observation order, simulator versions, reward convention,
and horizon. Old evaluation archives are intentionally not accepted as training
replay. Direct original HDF5-to-transition conversion is not implemented; do not
invent terminal/next-state conventions by reusing the BC-only data loader.

## Tests and experimental scope

```bash
$PY -m unittest discover -s tests -v
```

The regression suite covers the original analytic density tests; legacy sampler
equivalence; inverse integration; BC input validation; initial prior/KL;
unbounded latent storage; gripper expectation; terminal/nonterminal targets;
flow freezing; actor/critic updates; checkpoint round trips; action-only offline
replay; paired evaluation and RNG isolation; and complete runner accounting.

A controlled **analytic two-mode flow bandit** gives reward to opposite modes
in two different states. The RL learner sees action-only transitions and must
learn state-conditioned latent steering. This test establishes that the update
path can learn the desired behavior; it is not a robot success-rate claim and
does not measure whether the real BC checkpoints learned sufficient modes.

Implementation validation on 2026-11-01: all 12 tests passed. The synthetic
rewarded-mode fraction increased from approximately 0.506 to 1.000. A real Lift
smoke run collected 44 warmup and 4 online steps, performed 8 updates, retained
48 replay transitions, and evaluated BC and DSRL for one episode each (600
evaluation steps). Both evaluation episodes timed out: no robot improvement is
claimed. The BC fingerprint remained unchanged. Explicit encode/decode checks
on 8 recorded states per task converged as integration increased from 16 to 64
steps; the largest 64-step round-trip error was below 2e-7. See the compact
[check record](../results/dsrl_implementation_checks.json) and local artifacts
under `runs/dsrl_smoke_20261101/`.

After the bounded-actor/budget update, all **15 tests** passed, including actor
bounds/gradients, analytic KL, separate training/execution integration counts,
multi-update accounting, and legacy checkpoint behavior. The existing synthetic
regression test (its small KL weight is 0.03, not the new experiment default)
scored 0.506 for BC and 0.989 for the bounded actor. A local CPU microbenchmark
on 128 Lift replay samples measured median update times of 61.5 ms at 16 steps
and 34.7 ms at 8 steps (1.77x); maximum action difference was 0.00108 for sampled
prior noise. These timings are not a controlled hardware benchmark or an error
bound. See [stability checks](../results/dsrl_stability_checks.json). The larger
robotics budgets and KL sweep have not yet been run.

## Follow-on: learned context and multimodal noise

Keep this Gaussian actor as the baseline. A state-only deterministic context
concatenated to its inputs still produces one Gaussian per state; it does not
by itself create a multimodal latent distribution.

A concrete next experiment is `p(c|s)` selecting a categorical context, followed
by `p(z|s,c)`. Each component can steer a different useful mode through the same
frozen flow. The existing action critic and action-to-noise distillation can be
reused, while actor updates enumerate a small number of contexts. The actor
boundary is isolated in `LatentActor`, making this extension local.

To initialize contexts from the trained flow, collect frozen decoded actions or
fixed-probe velocity/hidden features on **training** states, learn a compact
embedding, and initialize diverse context heads from that embedding. Features
from the velocity network depend on state, noise, and integration time: pooling
must use specified fixed probes/times, not future rewards or evaluation data.
Do not append a context to the frozen velocity network as a new input; it was
not trained to interpret it. Initially use context to transform/select z.

Evaluate against the Gaussian actor with matched interaction budgets. Measure
return and success, context occupancy/entropy, action diversity at fixed states,
and sensitivity to seeds. Track collapse: reward optimization can select one
mode, so more contexts alone cannot guarantee multimodal outputs. Componentwise
KL averaged over contexts is not the exact mixture KL; specify the intended
regularizer before making claims about recovering the BC prior.

Context learning is a proposed next experiment, not part of the current code.
