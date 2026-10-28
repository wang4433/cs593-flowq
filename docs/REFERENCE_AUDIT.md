# Q2RL reference audit

This audit predates the completed Flow BC runs. The earlier missing-dataset and
mask-verification blockers are resolved; current data and results are in the README.

## Verified setup and reference audit

The initial workspace had no implementation, demonstrations, or policy weights;
the user's proposal PDF was preserved. No ancestor `AGENTS.md` applied. `.git`
is a protected placeholder, so this workspace has no usable local commit. Source
checksums are recorded in `artifacts/source_manifest.json` instead.

Upstream was inspected at commit
[`d565ab09f5ae88546f2dec43445fe8701ee3de65`](https://github.com/rai-opensource/q2rl/commit/d565ab09f5ae88546f2dec43445fe8701ee3de65).
Selected original files, their license, and upstream requirements are preserved
under `reference/q2rl/`. That directory is an audit snapshot, not a complete
executable clone. Upstream vendors modified Robomimic **0.3.0** and pins
robosuite **1.4.1**, MuJoCo **3.1.6**, numpy **1.26.1**, JAX **0.4.33**, and
torch **2.9.0**. The state launch scripts establish:

| Task | Dataset path under official dataset repo | Replay mask | Horizon | Initial rollouts | Q-estimation updates |
|---|---|---|---:|---:|---:|
| Lift | `lift/ph/low_dim_v141.hdf5` | `20_percent_train` | 300 | 100 | 20,000 |
| Can | `can/ph/low_dim_v141.hdf5` | `train` | 300 | 100 | 20,000 |
| Square | `square/ph/low_dim_v141.hdf5` | `train` | 500 | 100 | 50,000 |

Thus **PH is verified**, not assumed from benchmark convention. The GMM
checkpoint-embedded BC data paths, train masks, validation masks, and actual
demonstration ID memberships are **not yet verified**. Configuration defaults
for BC are provisional and visibly marked. No claim is made that replay masks
necessarily equal BC masks. The audit loader records exact HDF5 masks/IDs and
hashes when data are available; it refuses missing masks or mismatched versions.

The current [Robomimic dataset documentation](https://robomimic.github.io/docs/datasets/robomimic_v0.1.html)
warns that its current downloads use a newer robosuite conversion. The prototype
selects the official Q2RL `low_dim_v141` assets rather than that conversion.

Reference implementation findings, from the saved state runner, agent, wrapper,
and replay code:

- Frozen BC generates complete initial trajectories. A separate buffer supplies
  Monte Carlo targets. V fits returns; critic initialization fits return + log
  probability + entropy with implicit temperature 1; RL actor initialization is
  likelihood fitting to initial BC actions.
- Online selection compares min ensemble Q for the deterministic RL proposal
  against V + log probability + entropy for the sampled BC proposal. RL is used
  iff its score is strictly greater. V is excluded from online update sets.
- Executed transitions enter online replay regardless of their source. Optional
  original demonstrations seed that buffer. Warmup and online buffers are separate;
  samples become available after episode completion.
- Reward is `5*r - 1`, discount .99, batch 256, UTD 4. The critic ensemble has ten
  members with two subsampled for targets. Networks use 512-wide layers and layer
  normalization. The actor uses a BC auxiliary likelihood loss weighted .3 for
  Lift/Can and .2 for Square. Its online loss is not the conventional SAC
  entropy-regularized objective, despite temperature-related machinery.
- Critic entropy backup defaults off. The wrapper clips commands and labels
  success as `truncated`, time limit as `terminated`. Current main fixes the TD
  mask to stop at success, while bootstrapping at the time limit. In
  `calc_return_to_go`, an all-negative sparse episode gets `r_neg/(1-gamma)` at
  every state; other returns are computed backward. This matters for value scale.
- Training uses sparse rewards; the reference evaluation environment enables
  reward shaping. Logged warmup offsets use horizon × rollout count, rather than
  actual episode lengths. The new prototype records actual steps.
- Current main fixes recurrent GMM query consistency and uses an entropy upper
  bound in nats. Older code's entropy used log2/mean variance and is not the same
  estimator. The pinned commit is later than the paper version; bit-for-bit paper
  reproduction has not been established.

The [Q2RL paper](https://arxiv.org/html/2605.05172v3) motivates recovering an action
advantage from a Boltzmann policy. Here the construction is explicitly an
**approximation**: if the frozen flow is not Boltzmann with respect to the actual
task Q, `V + alpha*(log pi + H)` need not recover that Q. A zero-mean log-density
advantage does not by itself establish reward calibration or correct rankings.

## Prototype semantics and deviations

The validated local conda environment is `.conda/envs/cs593`; named activation
works using `scripts/activate.sh`. Python 3.10.22, torch 2.5.1+cpu, numpy 1.26.4,
h5py 3.16.0, robosuite 1.4.1, MuJoCo 3.2.6 were verified. GPU access is blocked.
The 3.2.6 simulator deviation is explicit and guarded by version assertions;
equivalence to upstream 3.1.6 is unverified. No Robomimic Python package is used
by the implementation; original HDF5 and its environment metadata are consumed
directly. Inherited unrelated packages are not used by the prototype.

Observation order is `robot0_eef_pos`, `robot0_eef_quat`,
`robot0_gripper_qpos`, `object`. In live robosuite observations the last key is
`object-state`. Physical observations are normalized using training-mask data
only. Critic and V additionally receive time remaining; the imitation policy
uses only physical observations. Original OSC_POSE controller configuration and
control frequency are loaded from dataset metadata. Each transition is one
controller action, not a chunk. Headless diagnostics use the stock Panda
OSC_POSE setup (20 Hz), with default task environments and original horizons.

PandaGripper applies sign to its command, with internal finger actuator state and
saturation. The prototype therefore models six tanh-transformed continuous arm
commands and a categorical gripper with support `{-1,0,1}`. Its mixed-measure log
probability includes the categorical probability and tanh Jacobian. It is a
density over controller inputs, not physical torques or post-saturation finger
positions. Gripper and arm are conditionally independent given state: a deliberate
modeling restriction. Demonstration arm endpoints are smoothed inward by .001
for atanh training. No inference clipping or unreported continuous density for a
discretized gripper is used.

Likelihood integrates `dx/dt=v` and divergence backward from action latent to a
Gaussian base. Entropy is an independent sample mean of negative joint log
probability. Exact Jacobian trace and fixed-probe Hutchinson trace are supported;
all logs use nats. Fixed RK4 step count controls numerical accuracy. The approach
follows the continuous change-of-variables framework described by
[FFJORD](https://arxiv.org/abs/1810.01367); finite-step integration error remains.

Both candidate scores target finite-horizon discounted `5*r-1`, with terminal
success and terminal horizon, gamma .99, and no entropy in critic backups. V is
fit to complete-rollout finite returns, including failure episodes, rather than
upstream's negative-reward infinite-return shortcut. Time-to-go in V/critics
supports that finite-horizon convention. The actor uses a SAC-style hybrid loss
with exact expectation over three gripper choices; `sac_alpha=.01` is independent
of `q_alpha=1`. The prototype uses two 128-wide critics, SiLU, UTD 1, and no
auxiliary BC loss online. These are simplifications, not upstream reproductions.

Warmup transitions seed online replay. No offline demonstration replay is used.
B and C share a single flow file and warmup file containing initial actor, critic,
V, and transitions. Both initialize their critic by fitting frozen reference
estimates, so initialization is held constant and their online gates isolate the
score-source change. Both use the same deterministic RL proposal and stochastic
flow proposal scheme. A has no online updates. Frozen module hashes, source-file
hashes, semantic config checks, reset seeds, evaluation observation hashes, and
separate proposal/density RNG streams support matched comparisons.
