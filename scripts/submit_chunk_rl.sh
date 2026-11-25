#!/bin/bash
# Dual RL vs DSRL on the chunked Square BC policy (h4_wide, 50k updates, EMA weights).
# 1) full unit-test suite (stop on failure); 2) 3 seeds x {dsrl, dual} in parallel, identical
# except for the dual block, sharing warmup per seed; 3) held-out paired evaluation on seeds
# 60000-60199 (200 episodes, same seeds as the chunked-BC confirmation); 4) comparison table.
# CPU-only workload; the rpaleja account still requires a GPU request.
#SBATCH -A rpaleja
#SBATCH -p a30
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=2-00:00:00
#SBATCH --job-name=chunk_rl
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -uo pipefail
ROOT=/scratch/gilbreth/hmoondra/cs593_project/cs593-flowq
cd "$ROOT"
export PY="$ROOT/.conda/envs/cs593/bin/python" FLOW=runs/chunk_bc/h4_wide/flow_50000.pt
export PYTHONPATH="$ROOT" NUMBA_CACHE_DIR="$ROOT/.cache/numba" MUJOCO_GL=egl CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1
export OUT=runs/chunk_rl_$SLURM_JOB_ID
SEEDS=${SEEDS:-"0 1 2"}
mkdir -p "$OUT"
stamp () { echo "== $(date '+%F %T') $*"; }

stamp "unit tests"
$PY -m unittest discover -s tests -v > "$OUT/tests.log" 2>&1 || { tail -30 "$OUT/tests.log"; echo "TESTS FAILED"; exit 1; }
tail -3 "$OUT/tests.log"

stamp "train: seeds [$SEEDS] x {dsrl, dual}"
for seed in $SEEDS; do for method in dsrl dual; do echo "$method $seed"; done; done | xargs -P 6 -L 1 bash -c '
  $PY -m flowq.chunk_rl train --config configs/square_chunk_$0_small.json --flow $FLOW --seed $1 \
    --device cpu --out $OUT/$0_s$1 > $OUT/$0_s$1.log 2>&1 || echo "train failed: $0 s$1"'

stamp "held-out paired evaluation (seeds 60000-60199)"
for seed in $SEEDS; do for method in dsrl dual; do echo "$method $seed"; done; done | xargs -P 6 -L 1 bash -c '
  [ -f $OUT/$0_s$1/chunk_rl.pt ] && $PY -m flowq.chunk_rl eval --checkpoint $OUT/$0_s$1/chunk_rl.pt --flow $FLOW \
    --episodes 200 --seed 60000 --device cpu --out $OUT/$0_s$1/heldout.json > $OUT/$0_s$1.heldout.log 2>&1 \
    || echo "heldout failed: $0 s$1"'

stamp "comparison"
$PY - <<'EOF'
import glob, json, os
out = os.environ['OUT']
def arms(ev):
    return {a: ev[a] for a in ('bc', 'reshaped', 'dsrl') if a in ev}
print('Per-run evaluations (training seeds 30000+, 100 episodes; known chunked BC = 70/100):')
for path in sorted(glob.glob(f'{out}/*_s*/summary.json')):
    s = json.load(open(path)); name = path.split('/')[-2]
    for ev in s['evaluations']:
        row = ' '.join(f"{a}={sum(e['success'] for e in v['episodes'])}/{len(v['episodes'])}" for a, v in arms(ev).items())
        print(f"  {name:8} online={ev['online_env_steps']:>6}  {row}  flow_updates={s['flow_updates']}  "
              f"mean_update_s={s['mean_update_seconds']:.3f}")
print('\nHeld-out (seeds 60000-60199, 200 episodes; known chunked BC = 150/200), pooled over seeds:')
pooled = {}
for path in sorted(glob.glob(f'{out}/*_s*/heldout.json')):
    method = path.split('/')[-2].split('_s')[0]; ev = json.load(open(path))
    for arm, v in arms(ev).items():
        p = pooled.setdefault((method, arm), [0, 0]); p[0] += sum(e['success'] for e in v['episodes']); p[1] += len(v['episodes'])
    bc, steered = ev['bc']['episodes'], ev['dsrl']['episodes']
    gained = sum(x['success'] and not b['success'] for b, x in zip(bc, steered))
    lost = sum(b['success'] and not x['success'] for b, x in zip(bc, steered))
    print(f"  {path.split('/')[-2]:8} " + ' '.join(f"{a}={sum(e['success'] for e in v['episodes'])}/{len(v['episodes'])}"
          for a, v in arms(ev).items()) + f"  steered vs bc: gained {gained}, lost {lost}")
for (method, arm), (k, n) in sorted(pooled.items()):
    print(f"  pooled {method:5} {arm:9} {k}/{n} = {k/n:.3f}")
print('\nDual vs DSRL, steered arms on the same held-out starts (paired within each training seed):')
total = [0, 0]
for p in sorted(glob.glob(f'{out}/dual_s*/heldout.json')):
    q = p.replace('/dual_s', '/dsrl_s')
    if not os.path.exists(q):
        continue
    dual, dsrl = json.load(open(p))['dsrl']['episodes'], json.load(open(q))['dsrl']['episodes']
    gained = sum(a['success'] and not b['success'] for a, b in zip(dual, dsrl))
    lost = sum(b['success'] and not a['success'] for a, b in zip(dual, dsrl))
    total[0] += gained; total[1] += lost
    print(f"  {p.split('/')[-2]}: dual-only successes {gained}, dsrl-only successes {lost}")
print(f"  all seeds: dual-only {total[0]}, dsrl-only {total[1]}")
EOF
stamp "done: $OUT"
