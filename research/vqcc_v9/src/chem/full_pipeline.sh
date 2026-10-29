#!/bin/bash
# state (K, W) -> assembled npz -> joint refit + gauge rounds -> 8-bit one-body refresh -> circuits -> official verify
# Usage: full_pipeline.sh <state npz> <workdir> [rounds] [qcap]
source ~/vqc/bin/activate
state=$1; work=$2; rounds=${3:-2}; qcap=${4:-0.0199985}
D=/mnt/c/Users/omen/Desktop/quantumAI/paradigm/chem
BASE=~/vqcwork/chem8/base/dfthcsos/submission.npz
bash $D/lam_pipeline.sh $BASE $state $work $rounds $qcap || exit 1
cd $work
last=$(ls -1 cand*_g.npz | sort -V | tail -1)
echo "lambda stage done: $last"
python3 -c "import numpy as np; z=np.load('$last'); np.save('last_words.npy', z['angle_words'])"
rm -rf ob
( cd ~/vqcwork/m1575/ds/source/chemistry_research/scripts/v5/recovered && \
  python refresh_one_body8.py --submission $work/$last --angle-words $work/last_words.npy \
    --verifier-dir ~/vqcwork/comp/verifiers/dfthcsos --out $work/ob --brot 8 --sweeps 4 --seconds 300 --threads 4 > $work/refresh.log 2>&1 ) || { echo refresh failed; tail -5 refresh.log; exit 1; }
OMP_NUM_THREADS=3 python $D/metrics.py ob/submission.npz
rm -rf circuits
OMP_NUM_THREADS=1 python $D/emit_family.py ob/submission.npz circuits 5 | cut -c1-120
OMP_NUM_THREADS=3 python /mnt/c/Users/omen/Desktop/quantumAI/chem_gauge/verify_chem.py ob/submission.npz circuits ~/vqcwork/comp $work/verify 2>&1 | tail -2
