#!/bin/bash
# Render saved Frozen Flow BC evaluation episodes (recorded-action replay, verified
# against the recording) for Lift, Can and Square. EGL offscreen rendering needs the GPU.
#SBATCH -A rpaleja
#SBATCH -p a100-40gb
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --job-name=render_bc
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail
ROOT=/scratch/gilbreth/hmoondra/cs593_project/cs593-flowq
cd "$ROOT"
PY="$ROOT/.conda/envs/cs593/bin/python"
export PYTHONPATH="$ROOT" NUMBA_CACHE_DIR="$ROOT/.cache/numba" MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
RUNS=${RUNS:-runs/verify_flow_12036366}
declare -A EPISODES=([lift]="0 2 1" [can]="0 1 2" [square]="8 11 0")  # two successes, then one timeout

for task in lift can square; do
  echo "== $task: episodes ${EPISODES[$task]}"
  $PY scripts/render_bc_results.py --config configs/${task}_bc_stage1.json \
    --flow "$RUNS/$task/bc/flow.pt" --evaluation "$RUNS/$task/eval" \
    --episodes ${EPISODES[$task]} --tolerance 1e-3 --out "$RUNS/$task/rendered"
done
echo "== done"
