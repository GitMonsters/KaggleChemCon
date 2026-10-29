#!/bin/bash
source ~/vqc/bin/activate
cd ~/vqcwork/chem8
cp -n s255x96c_best.npz s255x96_snap.npz
export PQ_LRTH=1e-4 PQ_LRW=5e-5 PQ_EARLY=4000 PQ_MID=3000 PQ_LATE=1200 PQ_SPLIT=0.05 PQ_SPLIT2=0.3
export PQ_FRACS=0.0005,0.001,0.0015,0.002,0.003,0.004,0.005,0.0075,0.01,0.015,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1,0.12,0.14,0.17,0.2,0.25,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0
setsid nohup python /mnt/c/Users/omen/Desktop/quantumAI/paradigm/chem/run_pq_ckpt.py base/dfthcsos/submission.npz s255x96_snap.npz pq96 >> pq96.log 2>&1 < /dev/null &
disown
sleep 20; cat pq96.log
