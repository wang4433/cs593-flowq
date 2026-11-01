#!/bin/bash
# Flow-matching verification: unit tests, synthetic oracle, fresh stage-1 BC per task,
# held-out demo diagnostics, and a 20-episode closed-loop reproduction.
# CPU-only workload; the rpaleja account still requires a GPU request.
#SBATCH -A rpaleja
#SBATCH -p a30
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --job-name=verify_flow
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail
ROOT=/scratch/gilbreth/hmoondra/cs593_project/cs593-flowq
cd "$ROOT"
PY="$ROOT/.conda/envs/cs593/bin/python"
export PYTHONPATH="$ROOT" NUMBA_CACHE_DIR="$ROOT/.cache/numba" MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=""
OUT=runs/verify_flow_$SLURM_JOB_ID
mkdir -p "$OUT"

echo "== unit tests"
$PY -m unittest discover -s tests -v

echo "== reference masks and data audit"
for task in lift can square; do
  [ -f artifacts/reference_$task.json ] || \
    $PY scripts/audit_reference_checkpoint.py data/reference/$task/gmm.pth --out artifacts/reference_$task.json
  $PY -m flowq.runner audit-data --config configs/${task}_bc_stage1.json --out "$OUT/${task}_dataset.json"
done

echo "== synthetic oracle + stage-1 BC training (parallel)"
$PY scripts/verify_flow.py synthetic --out "$OUT/synthetic.json" > "$OUT/synthetic.log" 2>&1 &
for task in lift can square; do
  $PY -m flowq.runner train-bc --config configs/${task}_bc_stage1.json --out "$OUT/$task/bc" > "$OUT/${task}_train.log" 2>&1 &
done
wait

echo "== held-out demo diagnostics + closed-loop rollouts (parallel)"
for task in lift can square; do
  $PY scripts/verify_flow.py demos --config configs/${task}_bc_stage1.json --flow "$OUT/$task/bc/flow.pt" \
    --out "$OUT/${task}_demos.json" > "$OUT/${task}_demos.log" 2>&1 &
  $PY scripts/rollout_bc.py --config configs/${task}_bc_stage1.json --flow "$OUT/$task/bc/flow.pt" \
    --episodes 20 --seed 30000 --out "$OUT/$task/eval" > "$OUT/${task}_eval.log" 2>&1 &
done
wait
echo "== done: $OUT"
