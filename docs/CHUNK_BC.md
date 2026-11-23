# Action-chunked Flow BC on Square

Question: the Square BC policy succeeds in under 10% of episodes, which caps what DSRL
steering can do. Is that because the network is small and the policy predicts one
command at a time, and if so, which chunk horizon helps?

Short answer: the best configuration found is a 4x512 network predicting 4 commands
per decision, about **74%** success (147-150 of 200 episodes on held-out seeds),
against 4% for the stage-1 recipe under the same evaluation. Training length and
network width matter at least as much as chunking.

All numbers are one training seed per configuration.

## Method

`flowq/chunk.py` keeps the stage-1 recipe (data, masks, state normalization, atanh arm
targets, bounded velocity, categorical gripper) and widens the output from one command
to `H`. The policy samples `H` commands, executes them open loop, then replans. Chunk
targets never cross a demonstration boundary; the tail of each demonstration repeats
its last command. With `H = 1` the model and loss equal the stage-1 `Flow`
(`tests/test_chunk.py`).

Optional training settings, all off by default: network size overrides, a separately
sized gripper head, AdamW weight decay, cosine learning-rate decay, and an EMA copy of
the weights.

These checkpoints do not load into the DSRL / dual code, which assumes a single 6-D
arm command.

## Round 1: horizon x network x training length

100 episodes per cell, seeds 30000-30099, whole chunk executed before replanning.
The 95% interval is roughly +/- 8 to 10 points.

| H | 2x128, 10k | 2x128, 50k | 4x256, 10k | 4x256, 50k |
|---:|---:|---:|---:|---:|
| 1 | 4% | 24% | 10% | 33% |
| 2 | 5% | 19% | 12% | 49% |
| 4 | 9% | 13% | 14% | 55% |
| 8 | 1% | 6% | 5% | 21% |
| 16 | 2% | 0% | 1% | 4% |

- The stage-1 budget of 10k updates is far too short: 4% to 24% from training length alone.
- Chunking helps only with the larger network, best at H = 2 to 4.
- Replanning after half a chunk gave no consistent gain (4x256, 50k: H=4 55% to 43%; H=16 4% to 10%).
- On 4x256 the gripper head overfits: validation cross-entropy rises from about 0.25 at
  10k updates to 0.9-2.6 at 50k while training cross-entropy goes to about 0.01.

## Round 2: variants of 4 layers, H = 4

All with EMA 0.999. 100 episodes, same seeds as round 1.

| Run | Change | 50k EMA | 100k EMA | 200k EMA | Final raw |
|---|---|---:|---:|---:|---:|
| `h4_base` | 4x256 | 61% | 61% | 60% | 55% |
| `h2_base` | H = 2 | 65% | 48% | 44% | 39% |
| `h8_base` | H = 8 | 29% | 39% | 59% | 57% |
| `h4_wd` | weight decay 0.01 | 59% | 65% | 66% | 58% |
| `h4_cos` | cosine decay | n/a | 59% | 65% | 68% |
| `h4_grip` | 2x128 gripper head | 52% | 62% | 63% | 54% |
| `h4_wide` | 4x512, 100k updates | 70% | 68% | n/a | 54% |
| `h4_all` | 4x512 + weight decay + cosine + small gripper head, 100k | 67% | 70% | n/a | 69% |

- Width is the clearest gain: both 4x512 runs reach 67-70%.
- EMA weights are usually a few points above the raw weights and much more stable
  (`h4_wide` at 100k: 68% vs 54%).
- Longer training does not help 4x256 at H = 4, and hurts at H = 2. H = 8 needs 200k
  updates to catch up with H = 4.
- Weight decay, cosine decay and the small gripper head each move results by 0 to 5
  points, inside the noise.

## Held-out confirmation

The three best round-2 results were picked from 29 evaluations, which biases them
upward, so they were re-evaluated on fresh seeds (60000-60199, 200 episodes).

| Checkpoint | Weights | Success | 95% interval |
|---|---|---:|---|
| `h4_wide`, 50k | EMA | 150/200 | 69-80% |
| `h4_all`, 100k | raw | 147/200 | 67-79% |
| `h4_all`, 100k | EMA | 140/200 | 63-76% |

The three intervals overlap; they are not distinguishable.

## Reproduce

```bash
python -m flowq.chunk train --config configs/square_bc_deep.json --chunk 4 --hidden 512 \
    --updates 50000 --ema 0.999 --out runs/chunk/h4_wide
python -m flowq.chunk eval --config configs/square_bc_deep.json \
    --flow runs/chunk/h4_wide/flow_50000.pt --ema --episodes 200 --seed 60000 \
    --out runs/chunk/h4_wide/confirm
```

`scripts/submit_chunk_sweep.sh` and `scripts/submit_chunk_sweep2.sh` are the two Slurm
jobs. Every evaluation is listed in `results/chunk_bc_square.json`.

## Open

- One training seed per configuration.
- DSRL and dual RL need changes to steer a chunked policy (critic and noise policy
  over `6H` dimensions, rewards summed over a chunk).
- Not tried: lower sampling noise, deterministic gripper, actions modeled without the
  atanh transform, observation history.
