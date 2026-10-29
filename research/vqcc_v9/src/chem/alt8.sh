#!/bin/bash
# Alternate fixed-word joint refit and Adam column gauge on an 8-bit factorization; verify each candidate.
# Usage: alt8.sh <workdir> <start.npz> <angle dir> <first> <last> [qcap]
source ~/vqc/bin/activate
work=$1; prev=$2; angles=$3; qcap=${6:-0.0199985}
mkdir -p $work && cd $work
for i in $(seq $4 $5); do
  OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 JR_BROT=8 JR_CAP=500 JR_QCAP=$qcap JR_ACCEPT_GAP=3.91505 \
    JR_STAGES="[(1e-4,60),(1e-5,100),(1e-6,100),(1e-7,60)]" \
    python ~/vqcwork/b303/joint_candidate.py $prev cand$i.npz > joint$i.log 2>&1 || { echo "joint $i failed"; tail -5 joint$i.log; exit 1; }
  python -I /mnt/c/Users/omen/Desktop/quantumAI/chem_gauge/gauge_opt.py cand$i.npz cand${i}_g.npz - 3000 > gauge$i.log 2>&1 || { echo "gauge $i failed"; tail -5 gauge$i.log; exit 1; }
  OMP_NUM_THREADS=3 python /mnt/c/Users/omen/Desktop/quantumAI/chem_gauge/verify_chem.py cand${i}_g.npz $angles ~/vqcwork/comp $work/v_c${i}g > verify$i.log 2>&1
  echo "round $i: $(tail -1 gauge$i.log) | $(head -1 verify$i.log | cut -c1-200)"
  prev=cand${i}_g.npz
done
