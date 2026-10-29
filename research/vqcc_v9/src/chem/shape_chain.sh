#!/bin/bash
# prune (once) -> long resumable continuous fit.  Usage: shape_chain.sh <src npz> <B1> <C1> <tag> <steps>
source ~/vqc/bin/activate
cd ~/vqcwork/chem8
D=/mnt/c/Users/omen/Desktop/quantumAI/paradigm/chem
src=$1; B1=$2; C1=$3; tag=$4; steps=$5
[ -f ${tag}_pruned.npz ] || python $D/prune_shape.py base/dfthcsos/submission.npz $src $B1 $C1 ${tag}_pruned.npz 3e-5 || exit 1
python $D/run_cont_ckpt.py base/dfthcsos/submission.npz ${tag}_pruned.npz ${tag}c $steps 3e-5 5e-5 3e-5
