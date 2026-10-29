#!/bin/bash
source ~/vqc/bin/activate
cd ~/vqcwork/chem8
# stop all continuous fits
for p in $(pgrep -f "[s]hape_chain.sh"); do kill $p; done
for p in $(pgrep -f "[r]un_cont_ckpt.py"); do kill $p; done
sleep 2
cp s255x112c_best.npz s255x112_snap.npz
cp s239x128c_best.npz s239x128_snap.npz
export PQ_LRTH=1e-4 PQ_LRW=5e-5 PQ_EARLY=4000 PQ_MID=4000 PQ_LATE=1500 PQ_FINAL=4000 PQ_SPLIT=0.05 PQ_SPLIT2=0.3 PQ_KAPPA=3e-5
export PQ_FRACS=0.0005,0.001,0.0015,0.002,0.003,0.004,0.005,0.0075,0.01,0.015,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1,0.11,0.12,0.135,0.15,0.17,0.19,0.21,0.24,0.27,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0
for t in s255x112 s239x128; do
  setsid nohup python /mnt/c/Users/omen/Desktop/quantumAI/paradigm/chem/run_pq_ckpt.py base/dfthcsos/submission.npz ${t}_snap.npz pq_$t >> pq_$t.log 2>&1 < /dev/null &
  disown
done
bash /mnt/c/Users/omen/Desktop/quantumAI/paradigm/chem/package.sh ~/vqcwork/chem8/fpH2v3 best_255b
sleep 20; tail -1 pq_s255x112.log; tail -1 pq_s239x128.log
