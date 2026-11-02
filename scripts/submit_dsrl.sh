#!/bin/bash
# DSRL-inspired latent steering on the frozen stage-1 flows: full test suite, then
# Lift/Can/Square trained in parallel with the shipped configs (one process each,
# single-threaded). Override with TASKS="can" SEED=1 KL_BETA=10 sbatch ...
# CPU-only workload; the rpaleja account still requires a GPU request.
#SBATCH -A rpaleja
#SBATCH -p a30
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=3-00:00:00
#SBATCH --job-name=dsrl
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail
ROOT=/scratch/gilbreth/hmoondra/cs593_project/cs593-flowq
cd "$ROOT"
PY="$ROOT/.conda/envs/cs593/bin/python"
export PYTHONPATH="$ROOT" NUMBA_CACHE_DIR="$ROOT/.cache/numba" MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=""
FLOWS=${FLOWS:-runs/verify_flow_12036366}
TASKS=${TASKS:-"lift can square"}
SEED=${SEED:-0}
EXTRA=()
[ -n "${KL_BETA:-}" ] && EXTRA+=(--kl-beta "$KL_BETA")
OUT=runs/dsrl_$SLURM_JOB_ID
mkdir -p "$OUT"

echo "== unit tests"
$PY -m unittest discover -s tests -v

echo "== training: tasks=[$TASKS] seed=$SEED ${EXTRA[*]:-shipped kl_beta}"
pids=()
for task in $TASKS; do
  $PY -m flowq.dsrl_runner train --config configs/${task}_dsrl.json --flow "$FLOWS/$task/bc/flow.pt" \
    --seed "$SEED" "${EXTRA[@]}" --out "$OUT/$task" > "$OUT/${task}.log" 2>&1 &
  pids+=($!)
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
echo "== done: $OUT (status $status)"
exit $status
