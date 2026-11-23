#!/bin/bash
# Square chunked Flow BC, round 2: push success above the round-1 best (4x256, H=4, 50k: 55/100).
# Eight variants train in parallel (one core each), all on configs/square_bc_deep.json with an
# EMA copy of the weights. Each worker evaluates its own snapshots (EMA weights) and the final
# raw weights for 100 episodes on the round-1 seeds. The three best results are then
# re-evaluated for 200 episodes on fresh seeds, since picking a maximum is biased upward.
# CPU-only workload; rpaleja still requires a GPU request.
#SBATCH -A rpaleja
#SBATCH -p a30
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --job-name=chunk_sweep2
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -uo pipefail
ROOT=${ROOT:-/scratch/gilbreth/$USER/cs593-flowq}
cd "$ROOT"
export PY="$ROOT/.venv/bin/python" CFG=configs/square_bc_deep.json
export PYTHONPATH="$ROOT" NUMBA_CACHE_DIR="$ROOT/.cache/numba" MUJOCO_GL=egl CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1
export OUT=runs/chunk_sweep2_$SLURM_JOB_ID
mkdir -p "$OUT"
stamp () { echo "== $(date '+%F %T') $*"; }

worker () {  # name updates "snapshots" train-args...
  local name=$1 updates=$2 snaps=$3; shift 3
  $PY -m flowq.chunk train --config $CFG --updates $updates --snapshots $snaps --ema 0.999 "$@" \
    --out $OUT/$name > $OUT/$name.train.log 2>&1 || { echo "train failed: $name"; return; }
  for step in $snaps $updates; do
    $PY -m flowq.chunk eval --config $CFG --flow $OUT/$name/flow_$step.pt --ema --episodes 100 \
      --out $OUT/$name/eval_${step}_ema > $OUT/$name.eval_${step}_ema.log 2>&1 || echo "eval failed: $name $step ema"
  done
  $PY -m flowq.chunk eval --config $CFG --flow $OUT/$name/flow_$updates.pt --episodes 100 \
    --out $OUT/$name/eval_${updates}_raw > $OUT/$name.eval_${updates}_raw.log 2>&1 || echo "eval failed: $name raw"
  echo "$(date '+%T') finished $name"
}

table () {
$PY - "$1" <<'EOF'
import glob, json, os, sys
rows = []
for p in glob.glob(os.environ['OUT']+'/*/'+sys.argv[1]+'/summary.json'):
    r = json.load(open(p)); r['name'] = p.split('/')[-3]; rows.append(r)
print(f"{'run':>9} {'net':>6} {'H':>2} {'updates':>7} {'weights':>7} {'success':>8} {'95% CI':>13} {'steps':>5}")
for r in sorted(rows, key=lambda r: (r['name'], r['updates'], r['ema'])):
    lo, hi = r['wilson_95_interval']
    print(f"{r['name']:>9} {r['layers']}x{r['hidden']:<4} {r['chunk']:>2} {r['updates']:>7} "
          f"{'ema' if r['ema'] else 'raw':>7} {r['successes']:>4}/{r['episodes_count']:<3} "
          f"[{lo:.2f}, {hi:.2f}] {r['mean_steps']:>5.0f}")
EOF
}

stamp "train + evaluate (8 workers)"
worker h4_base 200000 "50000 100000" --chunk 4 &
worker h2_base 200000 "50000 100000" --chunk 2 &
worker h8_base 200000 "50000 100000" --chunk 8 &
worker h4_wd   200000 "50000 100000" --chunk 4 --weight-decay 0.01 &
worker h4_cos  200000 "100000"       --chunk 4 --cosine &
worker h4_grip 200000 "50000 100000" --chunk 4 --gripper-hidden 128 --gripper-layers 2 &
worker h4_wide 100000 "50000"        --chunk 4 --hidden 512 &
worker h4_all  100000 "50000"        --chunk 4 --hidden 512 --weight-decay 0.01 --cosine \
                                     --gripper-hidden 128 --gripper-layers 2 &
wait

stamp "selection results (seeds 30000-30099)"
table 'eval_*'

stamp "confirm top 3 on fresh seeds (60000-60199)"
$PY - <<'EOF' | xargs -P 3 -L 1 bash -c '$PY -m flowq.chunk eval --config $CFG --flow $0 $1 --episodes 200 --seed 60000 --out $2 > $2.log 2>&1 || echo "confirm failed: $0"'
import glob, json, os
rows = [json.load(open(p)) for p in glob.glob(os.environ['OUT']+'/*/eval_*/summary.json')]
for r in sorted(rows, key=lambda r: -r['successes'])[:3]:
    name = r['checkpoint'].split('/')[-2]
    tag = 'ema' if r['ema'] else 'raw'
    flag = '--ema' if r['ema'] else '--seed=60000'   # placeholder keeps the argument count fixed
    print(r['checkpoint'], flag, f"{os.environ['OUT']}/{name}/confirm_{r['updates']}_{tag}")
EOF
table 'confirm_*'
stamp "done: $OUT"
