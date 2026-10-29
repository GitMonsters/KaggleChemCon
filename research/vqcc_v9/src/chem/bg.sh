#!/bin/bash
# Run a python job in the background inside the venv: bg.sh <log> <script> [args...]
source ~/vqc/bin/activate
cd ~/vqcwork/chem8
log=$1; shift
setsid nohup python "$@" > "$log" 2>&1 < /dev/null &
disown
sleep 3
echo "started $!"
