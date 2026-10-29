#!/bin/bash
cd ~/vqcwork/chem8
cp c255k_best.npz src255_b.npz
setsid nohup bash /mnt/c/Users/omen/Desktop/quantumAI/paradigm/chem/shape_chain.sh src255_b.npz 191 128 s191x128 80000 > s191x128.log 2>&1 < /dev/null &
disown
sleep 2
pgrep -fa "[s]hape_chain" | cut -c1-120
