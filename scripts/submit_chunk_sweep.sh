#!/bin/bash
# Square Flow BC with action chunking: which prediction horizon helps?
# Grid: network {stage1 2x128, deep 4x256} x horizon {1,2,4,8,16}. Each model trains for
# 50k updates with a snapshot at 10k (the stage-1 budget). Every snapshot is evaluated
# closed loop for 100 episodes executing the whole chunk; the 50k models with H>=4 are
# also evaluated executing half the chunk before replanning.
# CPU-only workload; rpaleja still requires a GPU request.
#SBATCH -A rpaleja
#SBATCH -p a30
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --job-name=chunk_sweep
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -uo pipefail
ROOT=${ROOT:-/scratch/gilbreth/$USER/cs593-flowq}
cd "$ROOT"
export PY="$ROOT/.venv/bin/python"
export PYTHONPATH="$ROOT" NUMBA_CACHE_DIR="$ROOT/.cache/numba" MUJOCO_GL=egl CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1
export OUT=runs/chunk_sweep_$SLURM_JOB_ID
HORIZONS=${HORIZONS:-"1 2 4 8 16"}
NETS=${NETS:-"stage1 deep"}
EPISODES=${EPISODES:-100}
mkdir -p "$OUT"
stamp () { echo "== $(date '+%F %T') $*"; }

# The reference audit records an absolute checkpoint path, so build it on this machine.
[ -f artifacts/reference_square.json ] || \
  $PY scripts/audit_reference_checkpoint.py data/reference/square/gmm.pth --out artifacts/reference_square.json

stamp "train"
for net in $NETS; do for h in $HORIZONS; do echo "$net $h"; done; done | xargs -P 8 -L 1 bash -c '
  $PY -m flowq.chunk train --config configs/square_bc_$0.json --chunk $1 --updates 50000 \
    --snapshots 10000 --out $OUT/$0_h$1 > $OUT/$0_h$1.train.log 2>&1 || echo "train failed: $0 h$1"'

stamp "evaluate"
for net in $NETS; do for h in $HORIZONS; do
  echo "$net $h 10000 $h"; echo "$net $h 50000 $h"
  [ "$h" -ge 4 ] && echo "$net $h 50000 $((h/2))"
done; done | EPISODES=$EPISODES xargs -P 8 -L 1 bash -c '
  $PY -m flowq.chunk eval --config configs/square_bc_$0.json --flow $OUT/$0_h$1/flow_$2.pt \
    --episodes $EPISODES --execute $3 --out $OUT/$0_h$1/eval_$2_e$3 > $OUT/$0_h$1.eval_$2_e$3.log 2>&1 \
    || echo "eval failed: $0 h$1 $2 e$3"'

stamp "results"
$PY - <<'EOF'
import glob, json, os
rows = [json.load(open(p)) for p in glob.glob(os.environ['OUT']+'/*/eval_*/summary.json')]
print(f"{'net':>7} {'H':>3} {'exec':>4} {'updates':>7} {'success':>8} {'95% CI':>15} {'mean steps':>10}")
for r in sorted(rows, key=lambda r: (r['layers'], r['chunk'], r['updates'], -r['execute'])):
    lo, hi = r['wilson_95_interval']
    print(f"{r['layers']}x{r['hidden']:<4} {r['chunk']:>3} {r['execute']:>4} {r['updates']:>7} "
          f"{r['successes']:>4}/{r['episodes_count']:<3} [{lo:.2f}, {hi:.2f}] {r['mean_steps']:>10.0f}")
EOF
stamp "done: $OUT"
