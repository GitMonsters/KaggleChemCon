#!/bin/bash
# R10 version: assemble on Rustad's npz -> offset-only lambda opt -> rounds x (joint refit -> column gauge -> offset opt) -> one-body refresh
# -> circuits -> official verifier.   Usage: full_pipeline3.sh <state npz> <workdir> [rounds] [qcap]
source ~/vqc/bin/activate
state=$1; work=$2; rounds=${3:-3}; qcap=${4:-0.0199985}
D=/mnt/c/Users/omen/Desktop/quantumAI/paradigm/chem
BASE=${BASE:-~/vqcwork/r10/rustad/dfthcsos/submission.npz}
REC=~/vqcwork/m1575/ds/source/chemistry_research/scripts/v5/recovered
mkdir -p $work && cd $work
python $D/assemble.py $BASE $state cand0.npz || exit 1
OMP_NUM_THREADS=2 python $D/offset_opt.py cand0.npz cand0_o.npz 2000 > oo0.log 2>&1 || { echo "offset0 failed"; tail -3 oo0.log; exit 1; }
echo "cand0_o: $(tail -1 oo0.log)"
prev=cand0_o.npz
if [ "$(python3 -c "import numpy as np; print(np.load('cand0.npz')['W'].shape[2])")" != "$(python3 -c "import numpy as np; print(np.load('$BASE')['W'].shape[2])")" ]; then
  # columns changed: also try offsets matched to the base's one-body correction, keep the lower lambda_eff
  OMP_NUM_THREADS=2 python $D/offset_match.py cand0.npz $BASE cand0_m.npz > om0.log 2>&1 &&   OMP_NUM_THREADS=2 python $D/offset_opt.py cand0_m.npz cand0_mo.npz 2000 > oom0.log 2>&1 && echo "cand0_mo: $(tail -1 oom0.log)"
  prev=$(python3 - <<PY
import json
def lam(f):
    try:
        return json.loads(open(f).read().strip().splitlines()[-1].split('official ', 1)[1])['lambda_eff']
    except Exception:
        return float('inf')
print('cand0_mo.npz' if lam('oom0.log') < lam('oo0.log') else 'cand0_o.npz')
PY
)
  echo "offset start: $prev"
fi
for i in $(seq 1 $rounds); do
  OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 JR_BROT=8 JR_CAP=500 JR_QCAP=$qcap JR_ACCEPT_GAP=3.91505 \
    JR_STAGES="[(1e-4,60),(1e-5,100),(1e-6,100),(1e-7,60)]" \
    python ~/vqcwork/b303/joint_candidate.py $prev cand$i.npz > joint$i.log 2>&1 || { echo "joint $i failed"; tail -3 joint$i.log; exit 1; }
  python -I /mnt/c/Users/omen/Desktop/quantumAI/chem_gauge/gauge_opt.py cand$i.npz cand${i}_g.npz - 3000 > gauge$i.log 2>&1 || { echo "gauge $i failed"; tail -3 gauge$i.log; exit 1; }
  OMP_NUM_THREADS=2 python $D/offset_opt.py cand${i}_g.npz cand${i}_o.npz 2000 > oo$i.log 2>&1 || { echo "offset $i failed"; tail -3 oo$i.log; exit 1; }
  echo "round $i: $(tail -1 oo$i.log)"
  prev=cand${i}_o.npz
done
python3 -c "import numpy as np; z=np.load('$prev'); np.save('last_words.npy', z['angle_words'])"
rm -rf ob
( cd $REC && python refresh_one_body8.py --submission $work/$prev --angle-words $work/last_words.npy \
    --verifier-dir ~/vqcwork/comp/verifiers/dfthcsos --out $work/ob --brot 8 --sweeps 4 --seconds 300 --threads 3 ) > refresh.log 2>&1 || { echo "refresh failed"; tail -5 refresh.log; exit 1; }
OMP_NUM_THREADS=3 python $D/metrics.py ob/submission.npz
rm -rf circuits
OMP_NUM_THREADS=1 python $D/emit_r10.py ob/submission.npz circuits 5 16 | cut -c1-160
OMP_NUM_THREADS=3 python /mnt/c/Users/omen/Desktop/quantumAI/chem_gauge/verify_chem.py ob/submission.npz circuits ~/vqcwork/comp $work/verify 2>&1 | tail -2
