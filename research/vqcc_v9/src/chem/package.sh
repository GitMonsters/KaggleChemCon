#!/bin/bash
# Copy a verified full_pipeline result into paradigm/chem/<name>/ (npz + all angle_k_*.kmb + verifier JSON).
# Usage: package.sh <workdir> <name>
work=$1; name=$2
dst=/mnt/c/Users/omen/Desktop/quantumAI/paradigm/chem/$name
grep -q '"status": "PASS"' $work/verify/chem_verify.json || { echo "not PASS"; exit 1; }
rm -rf $dst && mkdir -p $dst/dfthcsos
cp $work/ob/submission.npz $dst/dfthcsos/
cp $work/circuits/angle_k_*.kmb $dst/dfthcsos/
cp $work/verify/chem_verify.json $dst/
cp $work/circuits/plans.json $dst/ 2>/dev/null
( cd $dst && sha256sum dfthcsos/submission.npz dfthcsos/angle_k_*.kmb > SHA256SUMS )
echo "packaged $dst: $(ls $dst/dfthcsos | wc -l) files, npz sha $(sha256sum $dst/dfthcsos/submission.npz | cut -c1-16)"
python3 -c "import json;d=json.load(open('$dst/chem_verify.json'));m=d['metrics'];print({k:d[k] for k in ('status','lambda_eff','toffoli_count','num_qubits','score','ratio')}, {k:m[k] for k in ('error_1','error_2','E_gap','Lambda')})"
