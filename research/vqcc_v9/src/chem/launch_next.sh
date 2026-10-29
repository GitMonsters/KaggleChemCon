#!/bin/bash
source ~/vqc/bin/activate
cd ~/vqcwork/chem8
for p in $(pgrep -f "[r]un_pq_ckpt.py base/dfthcsos/submission.npz s255x96_snap.npz pq96"); do kill $p; done
setsid nohup bash /mnt/c/Users/omen/Desktop/quantumAI/paradigm/chem/full_pipeline3.sh ~/vqcwork/chem8/pqH2_q.npz ~/vqcwork/chem8/fpH2v3 3 > fpH2v3.log 2>&1 < /dev/null &
disown
cp -n c255k_best.npz src255_c.npz
for s in "239 128 s239x128" "223 128 s223x128"; do
  set -- $s
  setsid nohup bash /mnt/c/Users/omen/Desktop/quantumAI/paradigm/chem/shape_chain.sh src255_c.npz $1 $2 $3 60000 > $3.log 2>&1 < /dev/null &
  disown
done
sleep 3; pgrep -fa "[r]un_|[f]ull_pipeline|[p]rune_shape" | cut -c60-200
