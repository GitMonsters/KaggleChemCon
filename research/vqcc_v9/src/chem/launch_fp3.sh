#!/bin/bash
cd ~/vqcwork/chem8
for p in $(pgrep -f "[f]ull_pipeline.sh /home/omen/vqcwork/chem8/pqH_q.npz") $(pgrep -f "[f]ull_pipeline2.sh /home/omen/vqcwork/chem8/pqH_q.npz"); do pkill -P $p; kill $p; done
sleep 1
setsid nohup bash /mnt/c/Users/omen/Desktop/quantumAI/paradigm/chem/full_pipeline3.sh ~/vqcwork/chem8/pqH_q.npz ~/vqcwork/chem8/fpH3 3 > fpH3.log 2>&1 < /dev/null &
disown
sleep 2; pgrep -fa "[f]ull_pipeline|[j]oint_candidate" | cut -c1-140
