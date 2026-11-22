# Dual RL on a frozen flow-matching policy: idea and references

## In one paragraph

We start from a flow-matching behavior-cloning (BC) policy trained on Robomimic
demonstrations and never modify its weights. Two RL learners then improve it,
sharing one critic and one reward. **RL #1 (noise steering, DSRL)** learns which input
noise to feed the flow in each state. **RL #2 (flow re-weighting)** learns a small,
bounded correction to the flow's velocity field that shifts its default action
distribution toward actions the critic rates higher. Every executed action is still
produced by the (corrected) flow, and both learners are kept close to BC.

## The pieces

| Piece | What it is | Trained? |
|---|---|---|
| BC flow `v_BC(t, y, s)` | Conditional flow matching from N(0, I) to `atanh(0.999·a)` (6-D arm) + categorical gripper | Frozen |
| Correction `g_θ` | `v_θ = v_BC + r·tanh(g_θ(ŝ, y, t))`, last layer zero-initialized, so it starts at BC | RL #2 |
| Noise policy `π_ω(z \| x)` | Gaussian over the flow's 6-D input noise | RL #1 |
| Action critic `Q^A(x, a)` | Twin critics trained by TD on real transitions: **the only critic that sees the reward** | Yes |
| Noise critic `Q^Z(x, z)` | Copy of `Q^A` seen through the flow, so π_ω gets gradients without backprop through the ODE | Yes (distilled) |

Executed action: `z ~ π_ω(·|x)`, integrate `dy/dt = v_θ(t, y, s)` from `y(0) = z` (16 RK4
steps), `a_arm = tanh(y(1))`, gripper from the frozen BC head. `x` is the normalized
state plus time remaining.

## The objective

Both learners serve one KL-regularized objective:

```
J(ω, θ) = E_s E_{z~π_ω} [ Q(s, f_θ(s, z)) ]  −  β · KL(π_ω ‖ N(0, I))  −  τ · KL(p_θ ‖ π_BC)
```

where `p_θ` is what the corrected flow produces from plain N(0, I) noise.

- **RL #1** is J's exact ω-step (DSRL). Its optimum re-weights the flow by `exp(Q/β)`.
- **RL #2** approximates J's θ-step. Its target is the BC distribution re-weighted once,
  `π_BC · exp(Q/τ)`. Candidates come from the **frozen BC flow with prior noise**
  ("Option A"), so repeated updates do not compound and steering is not counted twice.

## Update equations

```
Reward:        r = 5·success − 1   (−1 per step, +4 on success), episode ends at success or horizon, γ = 0.99
Action critic: y = r + γ(1−done) · E_g[ min_i Q̄^A_i(x', (tanh Φ_θ(s', z'), g)) ],   z' ~ π_ω(·|x')
Noise critic:  Q^Z(x, z) ← E_g[ min_i Q^A_i(x, (tanh Φ_θ(s, z), g)) ],   z half from N(0,I), half from π_ω
Noise policy:  min_ω  E[ −min_i Q^Z_i(x, z) ] + β · KL(π_ω ‖ N(0, I))
Flow update (every K actor updates):
  candidates   y_ji = Φ_BC(s_j, z_ji),  z_ji ~ N(0, I),  M per state
  weights      w_ji = softmax_i( Q̂(s_j, tanh y_ji) / τ )            (within each state)
  tilt loss    L_tilt = Σ w_ji ‖ v_θ(t, y_t, s_j) − (y_ji − x₀) ‖² / Σ w,   y_t = (1−t)x₀ + t·y_ji
  demo anchor  L_demo = ‖ v_θ(t, y_t, s_d) − (atanh(0.999·a_d) − x₀) ‖²   on BC training demos
  total        L = L_tilt + λ · L_demo     (only g_θ is updated)
```

`Φ` denotes integrating the flow from noise to action; `Q̂` is the twin minimum averaged
over the frozen gripper probabilities.

## What keeps it close to BC

| Mechanism | Limits |
|---|---|
| Noise mean bound (±2) and std range `[actor_std_min, 1]` | How far steering moves the noise |
| KL weight β | Steering's distance from the noise prior (equal to the action-space distance, since the flow is invertible) |
| Correction cap r (`residual_bound`) | How far each action can move for the same noise |
| Demo anchor λ (`demo_weight`) | How far the reshaped flow can drift from the demonstrations |

## Where it lives in the code

| File | Contents |
|---|---|
| `flowq/flow.py` | BC flow (configurable depth via `flow_layers`), sampling, likelihood |
| `flowq/dsrl.py` | Noise policy, critics, replay, DSRL updates (RL #1) |
| `flowq/dual.py` | `ResidualFlow` (bounded correction) and `DualLearner` (RL #2) |
| `flowq/dsrl_runner.py` | Training and paired evaluation (BC / reshaped / steered arms) |
| `configs/*_dsrl*.json`, `configs/*_dual*.json` | Settings; `_v2` = 3×256 networks with LayerNorm |
| `docs/DUAL.md`, `docs/DUAL_SMALL_TEST.md` | Configuration reference and staged test plan |

## Status and open questions (2026-11-22)

- DSRL alone improves Lift and Can on two seeds; Square shows no gain. The Square BC
  policy itself is weak, which limits what steering can choose from.
- Dual RL passed its identity check; a small Can test cannot yet separate it from DSRL alone.
- The larger `_v2` networks and separately sized correction are implemented but untested.
- Open: action chunking (no longer blocked now that gating is dropped), a stronger
  Square BC policy, demonstrations in replay, clipping TD targets to the time-dependent
  return range, and a fully joint variant (θ trained on π_ω's noise) as an ablation.

## References

Only the entries marked **verified** were confirmed against arXiv or our bibliography;
arXiv rate-limited the other lookups. Check the rest before citing.

| Work | Relevance |
|---|---|
| Wagenmaker et al. 2025, *Steering Your Diffusion Policy with Latent Space Reinforcement Learning* (DSRL), arXiv 2506.15799 — **verified** | RL #1 |
| Dodeja et al. 2026, *When Life Gives You BC, Make Q-functions* (Q2RL), arXiv 2605.05172 — **verified (bibliography)** | Original gating idea; Boltzmann Q from BC |
| Abdolmaleki et al. 2018, *Maximum a Posteriori Policy Optimisation* (MPO) | Same structure as RL #2: re-weight samples by exp(Q/η) under a KL bound, then weighted fitting |
| Peters et al. 2010, *Relative Entropy Policy Search* (REPS) | KL-bounded improvement gives `π_old · exp(Q/η)` |
| Haarnoja et al. 2017, *Reinforcement Learning with Deep Energy-Based Policies* | Maximum-entropy optimal policy ∝ exp(Q/α) |
| Peng et al. 2019, *Advantage-Weighted Regression* (AWR) | Advantage-weighted policy fitting |
| Zhang et al. 2025, *Energy-Weighted Flow Matching for Offline RL* (QIPO) | Closest flow-matching analogue of RL #2 (offline) |
| Ding et al. 2024, *Q-weighted Variational Policy Optimization* (QVPO) | Online Q-weighted diffusion loss |
| Nakamoto et al. 2024, *Steering Your Generalists* (V-GPS) | Value-based re-ranking of a frozen policy's samples |
| Chen et al. 2022, *Offline RL via High-Fidelity Generative Behavior Modeling* (SfBC); Hansen-Estruch et al. 2023, *IDQL* | Critic-based selection among generative-policy samples |
| Wang et al. 2022, *Diffusion-QL*; Psenka et al. 2023, *Q-Score Matching*; Park et al. 2025, *Flow Q-Learning*; Frans et al. 2025, *CFGRL* | Other ways to improve a diffusion or flow policy with a critic |
| Ren et al. 2024, *DPPO*; Zhang et al. 2025, *ReinFlow*, arXiv 2505.22094 — **verified (bibliography)**; McAllister et al. 2025, *Flow Matching Policy Gradients* | Fine-tuning the generator itself (alternative to the bounded correction) |
| Ankile et al. 2024, *From Imitation to Refinement* (ResiP); Yuan et al. 2024, *Policy Decorator* | Residual RL on a frozen BC policy |
| Lipman et al. 2023, *Flow Matching for Generative Modeling* — **verified (bibliography)**; Grathwohl et al. 2019, *FFJORD* — **verified (bibliography)** | Flow matching and continuous-flow likelihoods |
| Mandlekar et al. 2021, *robomimic* — **verified (bibliography)**; Chi et al. 2023, *Diffusion Policy*, arXiv 2303.04137 — **verified** | Benchmark and the standard visuomotor/chunking recipe |
