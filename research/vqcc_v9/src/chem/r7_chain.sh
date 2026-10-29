#!/bin/bash
# drop groups to R=7 (once) -> long resumable continuous fit.  Usage: r7_chain.sh <src npz> <tag> <steps>
source ~/vqc/bin/activate
cd ~/vqcwork/r10
D=/mnt/c/Users/omen/Desktop/quantumAI/paradigm/chem
src=$1; tag=$2; steps=$3
[ -f ${tag}_r7.npz ] || python $D/drop_groups.py lamA/o3.npz $src 7 ${tag}_r7.npz 6000 2e-5 || exit 1
python $D/run_cont_ckpt.py lamA/o3.npz ${tag}_r7.npz ${tag}c $steps 2e-5 5e-5 3e-5
