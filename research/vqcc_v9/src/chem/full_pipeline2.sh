#!/bin/bash
# Like full_pipeline.sh but refreshes the one-body words right after assembly (needed when the dictionary
# shape changes: stale one-body rows make E_SOS/E_gap meaningless for the joint refit), then again at the end.
# Usage: full_pipeline2.sh <state npz> <workdir> [rounds] [qcap]
source ~/vqc/bin/activate
state=$1; work=$2; rounds=${3:-2}; qcap=${4:-0.0199985}
D=/mnt/c/Users/omen/Desktop/quantumAI/paradigm/chem
BASE=~/vqcwork/chem8/base/dfthcsos/submission.npz
REC=~/vqcwork/m1575/ds/source/chemistry_research/scripts/v5/recovered
refresh() {  # refresh <in npz> <out dir>
  python3 -c "import numpy as np; z=np.load('$1'); np.save('$2.words.npy', z['angle_words'])"
  rm -rf $2
  ( cd $REC && python refresh_one_body8.py --submission $1 --angle-words $2.words.npy \
      --verifier-dir ~/vqcwork/comp/verifiers/dfthcsos --out $2 --brot 8 --sweeps 4 --seconds 300 --threads 3 ) > $2.log 2>&1
}
mkdir -p $work && cd $work
python $D/assemble.py $BASE $state cand0.npz || exit 1
refresh $work/cand0.npz $work/r0 || { echo "refresh0 failed"; tail -5 r0.log; exit 1; }
OMP_NUM_THREADS=3 python $D/metrics.py r0/submission.npz
prev=r0/submission.npz
for i in $(seq 1 $rounds); do
  OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 JR_BROT=8 JR_CAP=500 JR_QCAP=$qcap JR_ACCEPT_GAP=3.91505 \
    JR_STAGES="[(1e-4,60),(1e-5,100),(1e-6,100),(1e-7,60)]" \
    python ~/vqcwork/b303/joint_candidate.py $prev cand$i.npz > joint$i.log 2>&1 || { echo "joint $i failed"; tail -3 joint$i.log; exit 1; }
  python -I /mnt/c/Users/omen/Desktop/quantumAI/chem_gauge/gauge_opt.py cand$i.npz cand${i}_g.npz - 3000 > gauge$i.log 2>&1 || { echo "gauge $i failed"; tail -3 gauge$i.log; exit 1; }
  OMP_NUM_THREADS=3 python $D/metrics.py cand${i}_g.npz
  prev=cand${i}_g.npz
done
refresh $work/$prev $work/ob || { echo "refresh failed"; tail -5 ob.log; exit 1; }
OMP_NUM_THREADS=3 python $D/metrics.py ob/submission.npz
rm -rf circuits
OMP_NUM_THREADS=1 python $D/emit_family.py ob/submission.npz circuits 5 | cut -c1-120
OMP_NUM_THREADS=3 python /mnt/c/Users/omen/Desktop/quantumAI/chem_gauge/verify_chem.py ob/submission.npz circuits ~/vqcwork/comp $work/verify 2>&1 | tail -2
