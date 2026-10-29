#!/bin/bash
# package_r10.sh <npz> <circuit dir> <verify json> <name> : copy into paradigm/chem/<name>/ after protected-field check
npz=$1; circ=$2; vj=$3; name=$4
dst=/mnt/c/Users/omen/Desktop/quantumAI/paradigm/chem/$name
source ~/vqc/bin/activate
python /mnt/c/Users/omen/Desktop/quantumAI/paradigm/chem/protect_check.py ~/vqcwork/r10/rustad/dfthcsos/submission.npz $npz || exit 1
grep -q '"status": "PASS"' $vj || { echo "not PASS"; exit 1; }
rm -rf $dst && mkdir -p $dst/dfthcsos
cp $npz $dst/dfthcsos/submission.npz
cp $circ/angle_k_*.kmb $dst/dfthcsos/
cp $vj $dst/chem_verify.json
( cd $dst && sha256sum dfthcsos/* > SHA256SUMS )
echo "packaged $dst: $(ls $dst/dfthcsos | wc -l) files, npz sha $(sha256sum $dst/dfthcsos/submission.npz | cut -c1-16)"
python3 -c "import json;d=json.load(open('$dst/chem_verify.json'));m=d['metrics'];print({k:d[k] for k in ('status','lambda_eff','toffoli_count','num_qubits','score','ratio')}, {k:m[k] for k in ('error_1','error_2','E_gap','Lambda')})"
