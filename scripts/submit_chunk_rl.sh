#!/bin/bash
# Dual RL vs DSRL on the chunked Square BC policy (h4_wide, 50k updates, EMA weights).
# 1) full unit-test suite (stop on failure); 2) dsrl and dual runs in parallel, sharing warmup
# per seed; 3) held-out paired evaluation on seeds 60000-60199 (200 episodes, same seeds as the
# chunked-BC confirmation) of every run that has none yet; 4) comparison table.
# CPU-only workload; the rpaleja account still requires a GPU request.
#
# Which runs (environment variables, all optional):
#   TAG             configs/square_chunk_{dsrl,dual}_$TAG.json (default small)
#   DSRL_TAG/DUAL_TAG  per-method config tag (default $TAG)
#   SEEDS           training seeds (default "0 1 2"); DSRL_SEEDS/DUAL_SEEDS override per method
#                   (an empty string trains no run of that method)
#   REUSE_DSRL      an earlier job's output dir; its finished dsrl_s<seed> runs for REUSE_DSRL_SEEDS
#                   (training + held-out) are copied instead of retrained. Runs reproduce exactly on
#                   these nodes; the copy is refused unless the DSRL config (apart from the seed), the
#                   BC flow file and the held-out protocol all match.
# Examples:
#   TAG=40k sbatch scripts/submit_chunk_rl.sh                       # 3 seeds x {dsrl, dual}
#   DSRL_TAG=40k DUAL_TAG=m32 DUAL_SEEDS="0 1 2 3 4" DSRL_SEEDS="3 4" \
#     REUSE_DSRL=runs/chunk_rl_12541061 REUSE_DSRL_SEEDS="0 1 2" sbatch scripts/submit_chunk_rl.sh
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
TAG=${TAG:-small}
DSRL_TAG=${DSRL_TAG:-$TAG} DUAL_TAG=${DUAL_TAG:-$TAG}
DSRL_SEEDS=${DSRL_SEEDS-$SEEDS} DUAL_SEEDS=${DUAL_SEEDS-$SEEDS}
REUSE_DSRL=${REUSE_DSRL:-} REUSE_DSRL_SEEDS=${REUSE_DSRL_SEEDS:-}
P=${SLURM_CPUS_PER_TASK:-8}   # one single-threaded process per CPU
for f in configs/square_chunk_dsrl_$DSRL_TAG.json configs/square_chunk_dual_$DUAL_TAG.json; do
  [ -f "$f" ] || { echo "missing config $f"; exit 1; }
done
runs=$(for s in $DSRL_SEEDS; do echo "dsrl $s $DSRL_TAG"; done; for s in $DUAL_SEEDS; do echo "dual $s $DUAL_TAG"; done)
[ "$(echo "$runs" | grep -c .)" -le "$P" ] || { echo "more training runs than CPUs ($P)"; exit 1; }
[ -z "$REUSE_DSRL_SEEDS" ] || [ -n "$REUSE_DSRL" ] || { echo "REUSE_DSRL_SEEDS needs REUSE_DSRL"; exit 1; }
mkdir -p "$OUT"
stamp () { echo "== $(date '+%F %T') $*"; }

for seed in $REUSE_DSRL_SEEDS; do
  src=$REUSE_DSRL/dsrl_s$seed
  case " $DSRL_SEEDS " in *" $seed "*) echo "DSRL seed $seed is both reused and trained"; exit 1;; esac
  for f in config.json summary.json chunk_rl.pt heldout.json; do
    [ -f "$src/$f" ] || { echo "reuse: missing $src/$f"; exit 1; }
  done
  $PY - "$src" "configs/square_chunk_dsrl_$DSRL_TAG.json" <<'EOF' || exit 1
import hashlib, json, os, sys
src, path = sys.argv[1:]
strip = lambda d: {k: v for k, v in d.items() if k != 'seed'}
if strip(json.load(open(f'{src}/config.json'))['dsrl']) != strip(json.load(open(path))['dsrl']):
    sys.exit(f'reuse: {src} was trained with a different DSRL config than {path}')
if json.load(open(f'{src}/summary.json'))['flow_sha256'] != hashlib.sha256(open(os.environ['FLOW'], 'rb').read()).hexdigest():
    sys.exit(f'reuse: {src} used a different BC flow than {os.environ["FLOW"]}')
bc = json.load(open(f'{src}/heldout.json'))['bc']['episodes']
if len(bc) != 200 or bc[0]['seed'] != 60000:
    sys.exit(f'reuse: {src}/heldout.json is not the 200-episode evaluation on seeds 60000+')
EOF
  cp -r "$src" "$OUT/" && cp "$REUSE_DSRL/dsrl_s$seed.log" "$OUT/" || { echo "reuse: copy failed for $src"; exit 1; }
  stamp "reused $src (trained and held-out evaluated in an earlier job)"
done

stamp "unit tests"
$PY -m unittest discover -s tests -v > "$OUT/tests.log" 2>&1 || { tail -30 "$OUT/tests.log"; echo "TESTS FAILED"; exit 1; }
tail -3 "$OUT/tests.log"

stamp "train: dsrl seeds [$DSRL_SEEDS] (square_chunk_dsrl_$DSRL_TAG.json), dual seeds [$DUAL_SEEDS] (square_chunk_dual_$DUAL_TAG.json)"
echo "$runs" | grep . | xargs -r -P "$P" -L 1 bash -c '
  $PY -m flowq.chunk_rl train --config configs/square_chunk_$0_$2.json --flow $FLOW --seed $1 \
    --device cpu --out $OUT/$0_s$1 > $OUT/$0_s$1.log 2>&1 || echo "train failed: $0 s$1"'

stamp "held-out paired evaluation (seeds 60000-60199) of runs without one"
for d in "$OUT"/dsrl_s* "$OUT"/dual_s*; do [ -d "$d" ] && [ ! -f "$d/heldout.json" ] && basename "$d"; done \
  | xargs -r -P "$P" -L 1 bash -c '
  [ -f $OUT/$0/chunk_rl.pt ] && $PY -m flowq.chunk_rl eval --checkpoint $OUT/$0/chunk_rl.pt --flow $FLOW \
    --episodes 200 --seed 60000 --device cpu --out $OUT/$0/heldout.json > $OUT/$0.heldout.log 2>&1 \
    || echo "heldout failed: $0"'

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
print('\nFinal flow correction (last logged update of each dual run):')
for p in sorted(glob.glob(f'{out}/dual_s*/updates.jsonl')):
    u = json.loads(open(p).readlines()[-1])
    print(f"  {p.split('/')[-2]}: " + ' '.join(f"{k}={u[k]:.4g}" for k in ('flow_tau', 'flow_displacement_mean',
          'flow_displacement_ema', 'flow_displacement_max', 'flow_weight_ess') if u.get(k) is not None))
EOF
stamp "done: $OUT"
