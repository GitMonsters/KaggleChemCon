# CELL 1
import json, math, os, pathlib, sys
os.environ.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
import pandas as pd
from IPython.display import display

INPUT = pathlib.Path(os.environ.get('QCC_INPUT', '/kaggle/input'))
WORK = pathlib.Path(os.environ.get('QCC_WORK', '/kaggle/working'))
SUB = pathlib.Path(os.environ.get('QCC_SUB', '/tmp/qcc_submission'))
SLUG = 'google-verifiable-quantum-circuit-compilation'
COMP = next(p for p in [INPUT / 'competitions' / SLUG, INPUT / SLUG] if (p / 'kickmix').exists())
# The chemistry numeric input is located by the build cell.
WORKERS = int(os.environ.get('QCC_WORKERS', os.cpu_count() or 4))
for d in (SUB, WORK / 'src' / 'gen_invsqrt_r2', WORK / 'src' / 'gen_dqi_v6' / 'ref'):
    d.mkdir(parents=True, exist_ok=True)
os.chdir(WORK)
REFERENCE = {'invsqrt': 2076774.233715282, 'dqi': 24439474177519.598, 'dfthcsos': 3355599588.3155365}

for _d in []:
    (WORK / _d).mkdir(parents=True, exist_ok=True)

# CELL 2
import hashlib, shutil
def sha(path):
    with open(path, 'rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()
OWN = next(p.parent for p in INPUT.rglob('femoco_thc_r5.npz'))
shutil.copytree(next(OWN.rglob('public_v11.py')).parent, WORK / 'src', dirs_exist_ok=True)
OPEN = OWN / 'open_circuits'
PINS = {
    'invsqrt.kmb': '7bde83f48462a245b9307fe0d9db985d9b2e9b5b1901a48b21df96eca2106985',
    'dfthcsos/angle_k_0.kmb': '8b0771aea947c282c0da204b59366b01bf8c8ac8e20bb6e36bb9cc645af6d04f',
    'dfthcsos/angle_k_1.kmb': 'dac3cbc0b560c6e1d4d2787751baf2a372d1d49161ecd7101e3011c02c753d37',
    'dfthcsos/angle_k_10.kmb': 'a027d89d3f522ff373270196c1ed56975b97cc820490f4e061545e24ea990eb4',
    'dfthcsos/angle_k_11.kmb': 'a07301189a00a4c85c43ea6a3e42242bce229d65059912823ecf70459ab73b4d',
    'dfthcsos/angle_k_12.kmb': 'dc66d18e5c84624b37b2a4a1f5388054b1c395299848fed6291f9b5dfc4aa6c0',
    'dfthcsos/angle_k_13.kmb': '2ba7002934812b0e80f927cbfb47f03bd5b8ea86042b744dfb3b349f508de274',
    'dfthcsos/angle_k_14.kmb': '102bf40e6e1b8dca8e640cca669f530f73cf185c79a1979b60147503003f5ae5',
    'dfthcsos/angle_k_2.kmb': '48338c0ed7047be27f526fc4c8ee10f117428cd239a0ce8db6c1a817244d6ed9',
    'dfthcsos/angle_k_3.kmb': '62d53a0c8e2f64b9336c118b75adf6d77c569d0b6dde77166a87681e54cf1305',
    'dfthcsos/angle_k_4.kmb': 'bdebd22ed26449aceacd44ec262a47555ef6d064f945c3d4e3859126c1df704b',
    'dfthcsos/angle_k_5.kmb': 'e3de4485c276b71b50878cf2af1c9a93b78f487670b6a539219bd609939047a0',
    'dfthcsos/angle_k_6.kmb': '2d8de690f8b09b906c450cd7aad325fc135e33fdbc63263956a48126d4175445',
    'dfthcsos/angle_k_7.kmb': 'a784b6b89f8a6a3b6813f92e18f49037cda630bbc9d73c2cfb621fb0b0586deb',
    'dfthcsos/angle_k_8.kmb': '0de383e2cadf1b4465193e61a3aab43210eb5207e2fe0c92bdbde490653ba635',
    'dfthcsos/angle_k_9.kmb': '88505bbef8cd3b49e5d96d9264ebf254bc78d678523b6d08709f0c5a103fce7a',
    'dfthcsos/submission.npz': '29bebc6e3aab18e758985b30c5e0ca8a1b77c4b0dfd55caddc2f9ee2046fa79e',
}
sys.path.insert(0, str(WORK / 'src'))
import qcc_pipeline as P
P.setup(COMP)
import public_v11 as V
V.build_dqi(SUB, WORK / 'build' / 'dqi', COMP)
for name, pin in PINS.items():
    assert sha(OPEN / name) == pin, name
    (SUB / name).parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(OPEN / name, SUB / name)
REFIT = OWN / 'femoco_r4_b303_refit.npz'
assert sha(REFIT) == '35bda9555d76f74430ef074f08f2dcdc24686305a630a489fff77ba9a283a55b'
shutil.copyfile(REFIT, SUB / 'dfthcsos' / 'submission.npz')
RESULTS = P.verify_all(SUB, 'notebook-verification', workers=WORKERS)
assert all(r['status'] == 'PASS' for r in RESULTS.values())
(WORK / 'verification.json').write_text(json.dumps(RESULTS, indent=1, default=str))
P.write_zip(SUB, WORK / 'submission.zip')

# CELL 7
r = RESULTS
rows = [
    ('Inverse square root', r['invsqrt']['num_qubits'], f"{r['invsqrt']['toffoli_count']:,.0f}",
     f"C = {r['invsqrt']['toffoli_equivalent']:,.0f} ({r['invsqrt']['rotation_count']:.0f} rotations)", 'invsqrt'),
    ('DQI decoder', r['dqi']['num_qubits'], f"{r['dqi']['toffoli_count']:,.0f}",
     f"R = {r['dqi']['repetition_factor']:.3f} ({r['dqi']['value_errors'] + r['dqi']['phase_errors']} failed of {r['dqi']['verified_shots']:,})", 'dqi'),
    ('FeMoCo lookups', r['dfthcsos']['num_qubits'], f"{r['dfthcsos']['toffoli_count']:,.0f}",
     f"λ_eff = {r['dfthcsos']['lambda_eff']:.2f}", 'dfthcsos'),
]
table = pd.DataFrame([{'Task': name, 'Qubits': q, 'Toffoli': t, 'Other': other,
                       'Cost': f"{r[k]['score']:.4g}", 'Reference cost': f'{REFERENCE[k]:.4g}',
                       'Reference / cost': f"{REFERENCE[k] / r[k]['score']:.2f}"} for name, q, t, other, k in rows])
score = math.prod(REFERENCE[k] / r[k]['score'] for k in REFERENCE) ** (1 / 3)
display(table.style.hide(axis='index'))
print(f'Score (geometric mean of reference / cost): {score:.3f}')

# CELL 9
P.write_zip(SUB, WORK / 'submission.zip')