#!/bin/bash
source ~/vqc/bin/activate
cd ~/vqcwork/chem8
setsid nohup python /mnt/c/Users/omen/Desktop/quantumAI/paradigm/chem/run_cont_ckpt.py base/dfthcsos/submission.npz s239x128_pruned.npz s239x128c 120000 3e-5 5e-5 3e-5 >> s239x128.log 2>&1 < /dev/null &
disown
sleep 15; tail -2 s239x128.log
