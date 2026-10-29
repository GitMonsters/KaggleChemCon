#!/bin/bash
cd ~/vqcwork/chem8
setsid nohup bash /mnt/c/Users/omen/Desktop/quantumAI/paradigm/chem/shape_chain.sh src255_c.npz 255 112 s255x112 100000 > s255x112.log 2>&1 < /dev/null &
disown
sleep 40; cat s255x112.log; grep -E "step +[0-9]+000 " s223x128.log | tail -1; grep -E "step +[0-9]+000 " s239x128.log | tail -1; tail -2 fpH2v3.log
