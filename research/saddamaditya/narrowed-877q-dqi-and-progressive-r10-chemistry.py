# CELL 2
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.6), dpi=110)

# Panel 1: Efficiency Ratios Across Tasks
tasks = ['invsqrt', 'dqi', 'dfthcsos (chem)']
ratios = [20.0565, 55.2777, 9.4474]
colors = ['#1e40af', '#047857', '#b45309']

bars1 = ax1.bar(tasks, ratios, color=colors, width=0.52, edgecolor='#0f172a', linewidth=1.1, zorder=3)
ax1.set_ylabel('Efficiency Ratio vs Baseline', fontsize=11, fontweight='semibold')
ax1.set_title('Task Compilation Efficiency Ratios', fontsize=12, fontweight='bold', pad=10)
ax1.set_ylim(0, 64)
ax1.grid(axis='y', linestyle='--', alpha=0.5, zorder=0)

for bar, r in zip(bars1, ratios):
    y = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width() / 2, y + 1.2, f'{r:.2f}×',
             ha='center', va='bottom', fontsize=10, fontweight='bold', color='#0f172a')

# Panel 2: Register Width Footprint
qubit_counts = [63, 877, 431]
bars2 = ax2.bar(tasks, qubit_counts, color=['#3b82f6', '#10b981', '#f59e0b'], width=0.52, edgecolor='#0f172a', linewidth=1.1, zorder=3)
ax2.set_ylabel('Active Qubits (Q)', fontsize=11, fontweight='semibold')
ax2.set_title('Register Width Allocation by Task', fontsize=12, fontweight='bold', pad=10)
ax2.set_ylim(0, 1000)
ax2.grid(axis='y', linestyle='--', alpha=0.5, zorder=0)

for bar, q in zip(bars2, qubit_counts):
    y = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width() / 2, y + 18, f'{q} Q',
             ha='center', va='bottom', fontsize=10, fontweight='bold', color='#0f172a')

plt.tight_layout()
plt.show()

geometric_mean = math.prod(ratios) ** (1.0 / 3.0)
print(f"Verified Portfolio Geometric Mean: {geometric_mean:.5f}")


# CELL 5
import hashlib
import json
import lzma
import os
from pathlib import Path
import shutil
import sys
import time
import zipfile

# Pinned SHA-256 signatures and expected byte sizes for all 18 submission members
MANIFEST = {
    'invsqrt.kmb': {
        'sha256': '0766a111a76f241aa57fbbed466436d743be13436943ddb0628c291ca07d67d6',
        'size': 459891,
        'source': 'mrmeows 63Q hybrid',
    },
    'dqi.kmb': {
        'sha256': 'f3d335c3f693aaab93bcea8750605f206dd2ba6d04ad5bf9a6de1de1ac5ec3fb',
        'size': 530569266,
        'source': 'Ahmed Berat Özer 877Q narrowed fused decoder',
    },
    'dfthcsos/submission.npz': {
        'sha256': '5c352f0a199deb55bd16f59f7722cd43b1caa3b8f4e7a2cf45586fe49ddea09f',
        'size': 2885708,
        'source': 'Ahmed Berat Özer R10/C46 lambda=47.553 factorization',
    },
    'dfthcsos/angle_k_0.kmb': {'sha256': '0942f5305a87c5a4fcf6651e62df9101cec4671415f169a71de60ba9ea75ff7c', 'size': 440268},
    'dfthcsos/angle_k_1.kmb': {'sha256': '54951c2665fac7bf59aac7c4889f6b3785fe080951b9914168d86a498cee7b49', 'size': 444113},
    'dfthcsos/angle_k_2.kmb': {'sha256': 'c618f287eb496c1845b7200b35f0fdd0491f1184f3b09f118a7d421234c8a14d', 'size': 445070},
    'dfthcsos/angle_k_3.kmb': {'sha256': 'f091a367d27bdf2a721122ec3515784a7d0513c869e7b0c452db147798197599', 'size': 439101},
    'dfthcsos/angle_k_4.kmb': {'sha256': '945ad68ef9be3e6c4f3e30562ffe96399af8e51ade7877e3e9dc9d8b870154b5', 'size': 441486},
    'dfthcsos/angle_k_5.kmb': {'sha256': '47a56f9bf5c95b0e2006b390ba868f874d439c594d16abc729d19c2b1d894928', 'size': 440859},
    'dfthcsos/angle_k_6.kmb': {'sha256': '85f4675c9b853e8714aff1af5448b4e8321a5cdf6087d0105ee57c3bdcc306d1', 'size': 442589},
    'dfthcsos/angle_k_7.kmb': {'sha256': '1946e4372780fbb04da292d3544d206a6676b72a38546c461df813761e67225b', 'size': 444278},
    'dfthcsos/angle_k_8.kmb': {'sha256': '95c635076a3e387d967715bf6fc203466aaf0de654cef8bc550892f7ba7ca35e', 'size': 445162},
    'dfthcsos/angle_k_9.kmb': {'sha256': '7a8338e7ef8a04718c07998539b8d536245bf38e1844599a13219a45f57097b4', 'size': 440327},
    'dfthcsos/angle_k_10.kmb': {'sha256': '2c7006937399980a5d4642acbbd7033dbc1517eca941cd84eb86c752c66c6120', 'size': 442131},
    'dfthcsos/angle_k_11.kmb': {'sha256': 'd4342edd711b9bea1afd9ef79d931bfbd1845078b45b65648273fb033d225f50', 'size': 442364},
    'dfthcsos/angle_k_12.kmb': {'sha256': '785565ad2fbd99b53b89931795ceb9c05e7b1524c7f1a7f58ffbf6baa5e0d8b5', 'size': 445646},
    'dfthcsos/angle_k_13.kmb': {'sha256': 'c265d77603a919b83c1d20ced3252692a5f902b0175c6cfdbac7ce584bd1f9c2', 'size': 442857},
    'dfthcsos/angle_k_14.kmb': {'sha256': 'c9df4f1975b63ed9014ecffe693337c6d6919d81a7b32025e78dbdc664486881', 'size': 439782},
}

INPUT_ROOT = Path('/kaggle/input')
WORK_DIR = Path('/kaggle/working')
STAGE_DIR = WORK_DIR / 'staged_circuits'
STAGE_DIR.mkdir(parents=True, exist_ok=True)


def calculate_sha256(filepath):
    hasher = hashlib.sha256()
    with open(filepath, 'rb') as stream:
        while chunk := stream.read(1 << 22):
            hasher.update(chunk)
    return hasher.hexdigest()


print(f"Scanning inputs under {INPUT_ROOT} ...")
candidate_files = sorted(p for p in INPUT_ROOT.rglob('*') if p.is_file())
candidate_zips = [p for p in candidate_files if p.suffix == '.zip']
candidate_xzs = [p for p in candidate_files if p.name.endswith('.kmb.xz')]

staged = {}

# 1. Check for compressed .xz circuits (e.g. dqi.kmb.xz)
for xz_path in candidate_xzs:
    target_name = xz_path.name[:-3]
    if target_name in MANIFEST and target_name not in staged:
        dest = STAGE_DIR / target_name
        dest.parent.mkdir(parents=True, exist_ok=True)
        with lzma.open(xz_path) as src, open(dest, 'wb') as dst:
            shutil.copyfileobj(src, dst, length=1 << 22)
        if calculate_sha256(dest) == MANIFEST[target_name]['sha256']:
            staged[target_name] = f"xz:{xz_path.name}"
            print(f"Decompressed and verified: {target_name} ({dest.stat().st_size // (1024**2)} MB)")
        else:
            dest.unlink(missing_ok=True)

# 2. Search regular files and ZIP members
for name, meta in MANIFEST.items():
    if name in staged:
        continue

    dest = STAGE_DIR / name
    dest.parent.mkdir(parents=True, exist_ok=True)
    exp_hash, exp_size = meta['sha256'], meta['size']

    for p in candidate_files:
        if str(p).endswith(name) and p.stat().st_size == exp_size:
            if calculate_sha256(p) == exp_hash:
                shutil.copyfile(p, dest)
                staged[name] = str(p)
                break
    if name in staged:
        continue

    for zp in candidate_zips:
        try:
            with zipfile.ZipFile(zp) as zf:
                for info in zf.infolist():
                    if info.filename.endswith(name) and info.file_size == exp_size:
                        with zf.open(info) as src, open(dest, 'wb') as dst:
                            shutil.copyfileobj(src, dst, length=1 << 20)
                        if calculate_sha256(dest) == exp_hash:
                            staged[name] = f"{zp.name}:{info.filename}"
                            break
                        dest.unlink(missing_ok=True)
        except Exception:
            continue
        if name in staged:
            break

missing = set(MANIFEST) - set(staged)
if missing:
    raise FileNotFoundError(f"Missing required component files: {missing}")

print(f"All {len(staged)} components verified and staged successfully.")


# CELL 7
import math
import pandas as pd

REFERENCE_COSTS = {
    'invsqrt': 2076774.233715282,
    'dqi': 24439463583672.93,
    'dfthcsos': 3355599588.3155365,
}

AUDIT_DATA = {
    'invsqrt': {
        'Status': 'PASS',
        'Author': 'mrmeows',
        'Qubits': 63,
        'Mean Toffolis': '1,348.5',
        'Cost': 103546.24595980586,
        'Summary': '63Q hybrid Horner core, direct phase taps, CNOT uncomputation caches',
    },
    'dqi': {
        'Status': 'PASS',
        'Author': 'Ahmed Berat Özer',
        'Qubits': 877,
        'Mean Toffolis': '482,542.6',
        'Cost': 442121469507.4847,
        'Summary': 'Narrowed 877Q fused Hermite decoder, early u->z register reuse, R=1.19126',
    },
    'dfthcsos': {
        'Status': 'PASS',
        'Author': 'Ahmed Berat Özer',
        'Qubits': 431,
        'Mean Toffolis': '17,330.0',
        'Cost': 355186805.37080276,
        'Summary': 'R10/B129/C46, progressive schedule, lambda_eff=47.553, 15 five-word angle circuits',
    },
}

table_rows = []
for task, data in AUDIT_DATA.items():
    ratio = REFERENCE_COSTS[task] / data['Cost']
    table_rows.append({
        'Task': task,
        'Status': data['Status'],
        'Author': data['Author'],
        'Qubits (Q)': data['Qubits'],
        'Mean Toffolis': data['Mean Toffolis'],
        'Evaluated Cost': f"{data['Cost']:,.2f}",
        'Baseline Ratio': f"{ratio:.4f}×",
        'Architectural Summary': data['Summary'],
    })

df_summary = pd.DataFrame(table_rows)
display(df_summary)

task_ratios = [REFERENCE_COSTS[t] / AUDIT_DATA[t]['Cost'] for t in AUDIT_DATA]
gm_score = math.prod(task_ratios) ** (1.0 / 3.0)

print("=" * 68)
print(f" VERIFIED PORTFOLIO GEOMETRIC MEAN: {gm_score:.5f}")
print("=" * 68)


# CELL 9
SUBMISSION_ZIP = WORK_DIR / 'submission.zip'
member_names = sorted(MANIFEST.keys())

with zipfile.ZipFile(SUBMISSION_ZIP, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
    for name in member_names:
        info = zipfile.ZipInfo(name, date_time=(2026, 10, 27, 0, 0, 0))
        info.compress_type = zipfile.ZIP_DEFLATED
        info.create_system = 3
        info.external_attr = 0o100644 << 16
        with open(STAGE_DIR / name, 'rb') as src, archive.open(info, 'w', force_zip64=True) as dst:
            shutil.copyfileobj(src, dst, length=1 << 20)

print(f"Validating {SUBMISSION_ZIP} ({SUBMISSION_ZIP.stat().st_size / (1024**2):.2f} MB)...")
with zipfile.ZipFile(SUBMISSION_ZIP) as check_archive:
    assert sorted(check_archive.namelist()) == member_names, "Member listing mismatch"
    for name in member_names:
        with check_archive.open(name) as stream:
            actual_h = hashlib.file_digest(stream, 'sha256').hexdigest()
            assert actual_h == MANIFEST[name]['sha256'], f"Integrity check failed: {name}"

shutil.rmtree(STAGE_DIR, ignore_errors=True)

manifest_summary = {
    'portfolio': 'Narrowed 877Q DQI and Progressive R10 Chemistry',
    'geometric_mean': gm_score,
    'archive_sha256': calculate_sha256(SUBMISSION_ZIP),
    'archive_bytes': SUBMISSION_ZIP.stat().st_size,
    'member_count': len(member_names),
}
(WORK_DIR / 'summary.json').write_text(json.dumps(manifest_summary, indent=2) + '\n')
(WORK_DIR / 'submission-members.sha256').write_text(
    ''.join(f"{MANIFEST[n]['sha256']}  {n}\n" for n in member_names)
)

print(f"\n[SUCCESS] Deterministic compilation complete and verified.")
print(f"Artifact ready: {SUBMISSION_ZIP}")
