#!/bin/bash
cd ~/vqcwork/chem8
cp c255k_best.npz src255_a.npz
for s in "223 112 s223x112" "255 96 s255x96"; do
  set -- $s
  setsid nohup bash /mnt/c/Users/omen/Desktop/quantumAI/paradigm/chem/shape_chain.sh src255_a.npz $1 $2 $3 80000 > $3.log 2>&1 < /dev/null &
  disown
done
sleep 2
pgrep -fa "[s]hape_chain" | cut -c1-120
