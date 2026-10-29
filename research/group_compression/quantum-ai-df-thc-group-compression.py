# CELL 1
import hashlib, json, math, os, pathlib, sys
os.environ.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
import pandas as pd
from IPython.display import display

INPUT = pathlib.Path(os.environ.get('QCT_INPUT', '/kaggle/input'))
WORK = pathlib.Path(os.environ.get('QCT_WORK', '/kaggle/working'))
SUB = pathlib.Path(os.environ.get('QCT_SUB', '/tmp/v8_submission'))
SLUG = 'google-verifiable-quantum-circuit-compilation'
COMP = next(p for p in [INPUT / 'competitions' / SLUG, INPUT / SLUG] if (p / 'kickmix').exists())
DATA = next(INPUT.rglob('vqcc_v7.py')).parent
ROOTS = [p for p in INPUT.iterdir() if p.name not in (SLUG, 'competitions')]   # everything but the competition data
import lzma, shutil
XZ = pathlib.Path(os.environ.get('QCT_XZ', '/tmp/v8_xz'))
for i, p in enumerate(sorted(q for r in ROOTS for q in r.rglob('*.kmb.xz') if q.is_file())):   # xz-packed circuits: unpack once
    dst = XZ / str(i) / p.name[:-3]
    dst.parent.mkdir(parents=True, exist_ok=True)
    with lzma.open(p) as f, open(dst, 'wb') as g:
        shutil.copyfileobj(f, g, 1 << 22)
    print('unpacked', p.name, '->', dst, dst.stat().st_size // 2**20, 'MiB')
if XZ.exists():
    ROOTS.append(XZ)
WORK.mkdir(parents=True, exist_ok=True)
SUB.mkdir(parents=True, exist_ok=True)
WORKERS = int(os.environ.get('QCT_WORKERS', os.cpu_count() or 4))
REFERENCE = {'invsqrt': 2076774.233715282, 'dqi': 24439474177519.598, 'dfthcsos': 3355599588.3155365}
print('competition data:', COMP, '| v7 folder:', DATA, '| other inputs:', [p.name for p in ROOTS], '| workers:', WORKERS)


# CELL 2
sys.path.insert(0, str(DATA))
import vqcc_v7 as V
V.PINS = {   # output file -> SHA-256; the notebook pins every file itself
    'dfthcsos/angle_k_0.kmb': '616e4eeafa9af786d9517b0970b73d3980a6d3647c19e8e75763be1637fcc7a6',
    'dfthcsos/angle_k_1.kmb': 'e1cc1150ce2ab2c7556d12526ea14141d1ce11d416aa7bb748500436f48162e3',
    'dfthcsos/angle_k_10.kmb': 'd859aa8fc36af3828d895cbed53adda4f9511e35bc07c8e7db1c2f4f30acce4a',
    'dfthcsos/angle_k_11.kmb': '55b1660c5b9af49b375381779e496f4b0c39a7491d6f008bbae15ab77161a459',
    'dfthcsos/angle_k_12.kmb': 'a888bb96ddd14f8384822db1466a2abda9e22aae37c1dc03be81100138aaec2b',
    'dfthcsos/angle_k_13.kmb': 'add600eb09f4f6c0ea41072d4975cff0ec4b63cf74d46e02e0541b2bab7f6b21',
    'dfthcsos/angle_k_14.kmb': 'd79c80a46fbd6f3d333b11b681ee927892c49193da4e21266e1f082b3ead1379',
    'dfthcsos/angle_k_15.kmb': '34b150e91bdad43b15887028f91df66bad62047e4f011e70fb3905f132347eeb',
    'dfthcsos/angle_k_16.kmb': '955013639d5d7fc8cc79f1e7167ca520659e03f07bcc11f45c3e6f9775f6ad39',
    'dfthcsos/angle_k_17.kmb': 'f47adcfdffda82f4efab82768d76a1edbf92533842f44ef4acbdc0635e97fea5',
    'dfthcsos/angle_k_18.kmb': '55204e127d6e529b0c7caa7906020ad644b0731b5fdee6df69f7f46ab008cbce',
    'dfthcsos/angle_k_2.kmb': 'ea12a2c1a5395a820449b089ea8e9cb61dc8a97540288e18730297442c3d1c65',
    'dfthcsos/angle_k_3.kmb': '86446eb4d2e8df4136132a5db84ebce86e93862664f073aa58d499e72c798e37',
    'dfthcsos/angle_k_4.kmb': 'c46dae74ff267ae87f95f2d5a15070fc1222bfb4f67fc7a581bbb25c983af504',
    'dfthcsos/angle_k_5.kmb': 'a667e050ab0bdb1363d1607135f9462796c92de33f47512fecc265e19a3042af',
    'dfthcsos/angle_k_6.kmb': 'a42d0d342a2c3d4fae2164020ae585f83f9eb5d1425286f2bdea7ebc58802fdd',
    'dfthcsos/angle_k_7.kmb': '667af311226e77f2d587cfa38026f876c827805663d18db2e71aedde172b9760',
    'dfthcsos/angle_k_8.kmb': '11f7f3bc690f13a612f881ac74c09f5e122c30e2a9cd12e0b5340260eb60e6cd',
    'dfthcsos/angle_k_9.kmb': '299f532a92bde483443bd857e1bd0e56f1dd4d4b627a1860ab05f82c3abecd75',
    'dfthcsos/submission.npz': 'a5eabd76c2b2869e39511a53130cd909034dec6f26fa3db61354a9d12ffe6413',
    'dqi.kmb': '569863ae0fafd5f94fe91b498ec67ed0a4d96b826ced31cb0ff57c7fe6c063cb',
    'invsqrt.kmb': '9ac50386192a544f353ab0f46604e5e072981b8bfc8521e8551c689a9445e3b9',
}
ORIGIN = V.build(SUB, ROOTS)
for out, src in ORIGIN.items():
    print(f'{out:28s} <- {src}')
import subprocess, time
VER = COMP / 'verifiers'
ENV = dict(os.environ, PYTHONPATH=os.pathsep.join([str(COMP / 'kickmix'), str(VER)]))
COMMANDS = {   # the official verifiers, unmodified, at their default shot counts
    'invsqrt': [VER / 'invsqrt' / 'verify_submission_invsqrt.py', SUB / 'invsqrt.kmb', '--max-workers', str(WORKERS)],
    'dqi': [VER / 'dqi' / 'verify_submission_dqi.py', SUB / 'dqi.kmb'],
    'dfthcsos': [VER / 'dfthcsos' / 'verify.py', SUB / 'dfthcsos'],
}
RESULTS = {}
for task, cmd in COMMANDS.items():
    t0 = time.time()
    run = subprocess.run([sys.executable, '-u'] + [str(c) for c in cmd] + ['--salt', 'notebook-verification'],
                         capture_output=True, text=True, env=ENV)
    RESULTS[task] = json.loads(run.stdout[run.stdout.index('{'):]) if '{' in run.stdout else {'status': 'NO_OUTPUT', 'stderr': run.stderr[-2000:]}
    print(f"{task}: {RESULTS[task].get('status')} cost {RESULTS[task].get('score')} ({time.time() - t0:.0f} s)")
assert all(r.get('status') == 'PASS' for r in RESULTS.values()), {k: r.get('status') for k, r in RESULTS.items()}
(WORK / 'verification.json').write_text(json.dumps(RESULTS, indent=1, default=str))
import zipfile
with zipfile.ZipFile(WORK / 'submission.zip', 'w', zipfile.ZIP_DEFLATED) as z:
    for p in sorted(SUB.rglob('*')):
        if p.is_file():
            z.write(p, p.relative_to(SUB).as_posix())
print('submission.zip written,', (WORK / 'submission.zip').stat().st_size // 2**20, 'MiB')


# CELL 4
r = RESULTS
rows = [('Inverse square root', 'invsqrt'), ('DQI decoder', 'dqi'), ('FeMoCo lookups', 'dfthcsos')]
table = pd.DataFrame([{'Task': name, 'Qubits': r[k].get('num_qubits'), 'Toffoli (mean)': f"{r[k].get('toffoli_count', 0):,.0f}",
                       'Cost': f"{r[k]['score']:.5g}", 'Reference cost': f'{REFERENCE[k]:.5g}',
                       'Reference / cost': f"{REFERENCE[k] / r[k]['score']:.3f}"} for name, k in rows])
score = math.prod(REFERENCE[k] / r[k]['score'] for k in REFERENCE) ** (1 / 3)
display(table.style.hide(axis='index'))
print(f'Score (geometric mean of reference / cost): {score:.4f}')
