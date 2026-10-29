#!/bin/bash
source ~/vqc/bin/activate
cd ~/vqcwork/chem8
for p in $(pgrep -f "[s]hape_chain.sh src255_c.npz 223 128 s223x128"); do kill $p; done
for p in $(pgrep -f "[r]un_cont_ckpt.py base/dfthcsos/submission.npz s223x128_pruned.npz"); do kill $p; done
sleep 2
setsid nohup python /mnt/c/Users/omen/Desktop/quantumAI/paradigm/chem/run_cont_ckpt.py base/dfthcsos/submission.npz s223x128_pruned.npz s223x128c 150000 3e-5 5e-5 3e-5 >> s223x128.log 2>&1 < /dev/null &
disown
sleep 15; tail -3 s223x128.log
