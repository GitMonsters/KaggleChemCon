#!/bin/bash
source ~/vqc/bin/activate
cd ~/vqcwork/r10
until grep -q "done:" c46c.log; do sleep 10; done
cp c46c_best.npz c46_cont.npz
export PQ_LRTH=1e-4 PQ_LRW=5e-5 PQ_KAPPA=2e-5 PQ_SPLIT=0.05 PQ_SPLIT2=0.3 PQ_LATE=1500 PQ_FINAL=4000
export PQ_FRACS=0.0005,0.001,0.0015,0.002,0.003,0.004,0.005,0.0075,0.01,0.015,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1,0.11,0.12,0.135,0.15,0.17,0.19,0.21,0.24,0.27,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0
PQ_EARLY=4000 PQ_MID=4000 setsid nohup python /mnt/c/Users/omen/Desktop/quantumAI/paradigm/chem/run_pq_ckpt.py lamA/o3.npz c46_cont.npz pq46a >> pq46a.log 2>&1 < /dev/null &
disown
PQ_EARLY=8000 PQ_MID=6000 setsid nohup python /mnt/c/Users/omen/Desktop/quantumAI/paradigm/chem/run_pq_ckpt.py lamA/o3.npz c46_cont.npz pq46b >> pq46b.log 2>&1 < /dev/null &
disown
sleep 30; tail -2 c46c.log; head -2 pq46a.log pq46b.log
