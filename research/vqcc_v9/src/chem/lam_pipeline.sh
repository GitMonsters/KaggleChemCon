#!/bin/bash
# lambda polish of a two-body state at fixed words: assemble -> joint refit -> column gauge, report metrics.
# Usage: lam_pipeline.sh <base npz> <state npz> <workdir> [rounds] [qcap]
source ~/vqc/bin/activate
base=$1; state=$2; work=$3; rounds=${4:-2}; qcap=${5:-0.0199985}
D=/mnt/c/Users/omen/Desktop/quantumAI/paradigm/chem
mkdir -p $work && cd $work
python $D/assemble.py $base $state cand0.npz || exit 1
OMP_NUM_THREADS=3 python $D/metrics.py cand0.npz
prev=cand0.npz
for i in $(seq 1 $rounds); do
  OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 JR_BROT=8 JR_CAP=500 JR_QCAP=$qcap JR_ACCEPT_GAP=3.91505 \
    JR_STAGES="[(1e-4,60),(1e-5,100),(1e-6,100),(1e-7,60)]" \
    python ~/vqcwork/b303/joint_candidate.py $prev cand$i.npz > joint$i.log 2>&1 || { echo "joint $i failed"; tail -3 joint$i.log; exit 1; }
  python -I /mnt/c/Users/omen/Desktop/quantumAI/chem_gauge/gauge_opt.py cand$i.npz cand${i}_g.npz - 3000 > gauge$i.log 2>&1 || { echo "gauge $i failed"; tail -3 gauge$i.log; exit 1; }
  OMP_NUM_THREADS=3 python $D/metrics.py cand${i}_g.npz
  prev=cand${i}_g.npz
done
