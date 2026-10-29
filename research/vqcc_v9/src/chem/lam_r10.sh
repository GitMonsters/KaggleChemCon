#!/bin/bash
# Words-fixed lambda loop for R10 packages: rounds x (joint refit -> column gauge -> offset opt); angle words and
# therefore circuits unchanged. Verifies each round with the given circuit dir.
# Usage: lam_r10.sh <start npz> <circuit dir> <workdir> <rounds> [qcap]
source ~/vqc/bin/activate
start=$1; circ=$2; work=$3; rounds=$4; qcap=${5:-0.0199985}
D=/mnt/c/Users/omen/Desktop/quantumAI/paradigm/chem
mkdir -p $work && cd $work
cp $start s0.npz
prev=s0.npz
for i in $(seq 1 $rounds); do
  OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 JR_BROT=8 JR_CAP=500 JR_QCAP=$qcap JR_ACCEPT_GAP=3.91505 \
    JR_STAGES="[(1e-4,60),(1e-5,100),(1e-6,100),(1e-7,60)]" \
    python ~/vqcwork/b303/joint_candidate.py $prev j$i.npz > joint$i.log 2>&1 || { echo "joint $i failed"; tail -3 joint$i.log; exit 1; }
  python -I /mnt/c/Users/omen/Desktop/quantumAI/chem_gauge/gauge_opt.py j$i.npz g$i.npz - 3000 > gauge$i.log 2>&1 || { echo "gauge $i failed"; tail -3 gauge$i.log; exit 1; }
  OMP_NUM_THREADS=2 python $D/offset_opt.py g$i.npz o$i.npz 2000 > oo$i.log 2>&1 || { echo "offset $i failed"; tail -3 oo$i.log; exit 1; }
  echo "round $i: $(tail -1 oo$i.log)"
  prev=o$i.npz
done
OMP_NUM_THREADS=3 python /mnt/c/Users/omen/Desktop/quantumAI/chem_gauge/verify_chem.py $prev $circ ~/vqcwork/comp $work/verify 2>&1 | tail -2
