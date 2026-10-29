#!/bin/bash
# R10: eigen-truncate C (once) -> long resumable continuous fit.  Usage: r10_cchain.sh <C1> <tag> <steps>
source ~/vqc/bin/activate
cd ~/vqcwork/r10
D=/mnt/c/Users/omen/Desktop/quantumAI/paradigm/chem
C1=$1; tag=$2; steps=$3
[ -f ${tag}_pruned.npz ] || python $D/prune_shape.py lamA/o3.npz c46_cont.npz 129 $C1 ${tag}_pruned.npz 2e-5 || exit 1
python $D/run_cont_ckpt.py lamA/o3.npz ${tag}_pruned.npz ${tag}c $steps 2e-5 5e-5 3e-5
