#!/bin/bash
source ~/vqc/bin/activate
cd ~/vqcwork/chem8
for t in s223x112 s191x128; do
  for p in $(pgrep -f "[s]hape_chain.sh src255_.\.npz [0-9]* [0-9]* $t "); do kill $p; done
  for p in $(pgrep -f "[r]un_cont_ckpt.py base/dfthcsos/submission.npz ${t}_pruned.npz"); do kill $p; done
done
sleep 1
setsid nohup python /mnt/c/Users/omen/Desktop/quantumAI/paradigm/chem/run_sweeps.py base/dfthcsos/submission.npz pqH_q.npz pqH_sw.npz 12 1 0 20 3e-5 > pqH_sw.log 2>&1 < /dev/null &
disown
sleep 2
pgrep -fa "[r]un_|[f]ull_pipeline" | cut -c60-200
