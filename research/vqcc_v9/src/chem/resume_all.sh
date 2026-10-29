#!/bin/bash
# Resume every checkpointed chemistry GPU job (safe to re-run: each script resumes from its .pt / skips done steps).
source ~/vqc/bin/activate
cd ~/vqcwork/chem8
D=/mnt/c/Users/omen/Desktop/quantumAI/paradigm/chem
start() {  # start <log> <cmd...> unless an identical job is already running
  log=$1; shift
  if pgrep -f "[${1:0:1}]${1:1} $2 $3 $4" > /dev/null; then echo "running: $*"; return; fi
  setsid nohup "$@" >> $log 2>&1 < /dev/null &
  disown
  echo "started: $*"
}
[ -f pqH_q.npz ] || PQ_LRTH=1e-4 PQ_LRW=5e-5 start pqH.log python $D/run_pq_ckpt.py base/dfthcsos/submission.npz fr255b_cont.npz pqH
for s in "255 96 s255x96 src255_a.npz" "223 112 s223x112 src255_a.npz" "191 128 s191x128 src255_b.npz"; do
  set -- $s
  start $3.log bash $D/shape_chain.sh $4 $1 $2 $3 80000
done
sleep 3
pgrep -fa "[r]un_|[p]rune_shape" | cut -c1-160
