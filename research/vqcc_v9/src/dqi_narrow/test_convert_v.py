"""Original monic-root conversion (with 12-qubit history word) vs the virtual-lead version.

B = E^2 + x O^2 from chosen Euclid quotient-degree sequences of (E, O) (half degree 17), plus random B.
Checks forward outputs equal, round trip restored, scratch clean, phase clean.
Run with cwd = SJ source root, PYTHONPATH = official kickmix.
"""
import sys, json
from pathlib import Path
import numpy as np
ROOT = Path.cwd(); sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'exp/decoder'))
import kickmix as km
from exp.decoder.common import write, read, mul, inv, pmul, add, trim
from exp.convert_short_hankel.phase34 import convert_safe as conv_orig
import exp.narrow.convert_v as CV

gf = km.GF2Field(12); rng = np.random.default_rng(int(sys.argv[2]) if len(sys.argv) > 2 else 11)
per = int(sys.argv[1]) if len(sys.argv) > 1 else 64
t = 17; s = 35


def rpoly(deg):
    p = [int(x) for x in rng.integers(0, 4096, deg + 1)]
    while p[-1] == 0: p[-1] = int(rng.integers(1, 4096))
    return p


def chart(degs):
    Rn, R = [], [int(rng.integers(1, 4096))]
    for d in reversed(degs):
        q = rpoly(d); new = add(pmul(q, R), Rn); Rn, R = R, trim(new)
    O = R; li = inv(O[-1]); O = [mul(c, li) for c in O]; R1 = [mul(c, li) for c in Rn]
    c0 = int(rng.integers(0, 4096)); E = add([mul(c0, c) for c in O], R1); E = E + [0] * (t + 1 - len(E))
    assert len(O) == t + 1 and O[-1] == 1 and len(E) == t + 1
    B = [0] * (s + 1)
    for i, v in enumerate(E): B[2 * i] = mul(v, v)
    for i, v in enumerate(O): B[2 * i + 1] = mul(v, v)
    return B[:s]


def degs_for(kind):
    if kind == 'gap0': return [1] * 17
    if kind == 'gap1': d = [1] * 15; d.insert(int(rng.integers(0, 16)), 2); return d
    if kind == 'gap2': d = [1] * 13; [d.insert(int(rng.integers(0, len(d) + 1)), 2) for _ in range(2)]; return d
    if kind == 'gap_root': return [2] + [1] * 15
    if kind == 'gap_root2': return [2, 2] + [1] * 13
    if kind == 'gap_child': return [1, 2] + [1] * 14
    if kind == 'deg3_root': return [3] + [1] * 14
    raise ValueError(kind)


kinds = ['random', 'gap0', 'gap1', 'gap2', 'gap_root', 'gap_root2', 'gap_child', 'deg3_root']
cases = []; labels = []
for k in kinds:
    for _ in range(per):
        B = [int(x) for x in rng.integers(0, 4096, s)] if k == 'random' else chart(degs_for(k))
        A = [int(x) for x in rng.integers(0, 4096, s)]
        cases.append((A, B)); labels.append(k)
n = len(cases)


def build(fn, with_hist):
    b = km.CircuitBuilder(); A = [b.alloc_qubits(12) for _ in range(s)]; B = [b.alloc_qubits(12) for _ in range(s)]
    hist = b.alloc_qubits(12) if with_hist else None
    kw = dict(history_scratch12=hist, monic_root=True) if with_hist else {}
    tok = fn(b, A, B, **kw)
    cf = b.finish_circuit()
    fn(b, A, B, True, phase_token=tok, **({'history_scratch12': hist} if with_hist else {}))
    if with_hist: b.free(hist)
    return A, B, cf, b.finish_circuit()


res = {}
for name, fn, wh in [('orig', conv_orig, True), ('virt', CV.convert_safe, False)]:
    A, B, cf, cr = build(fn, wh)
    sim = km.Simulator(batch_size=n, seed=21); sim.use_same_registers_as(cf)
    for i in range(s):
        write(sim, A[i], [c[0][i] for c in cases]); write(sim, B[i], [c[1][i] for c in cases])
    sim.do(cf)
    outA = np.array([read(sim, A[i]) for i in range(s)]); outB = np.array([read(sim, B[i]) for i in range(s)])
    sim2 = km.Simulator(batch_size=n, seed=22); sim2.use_same_registers_as(cr)
    for i in range(s):
        write(sim2, A[i], [c[0][i] for c in cases]); write(sim2, B[i], [c[1][i] for c in cases])
    sim2.do(cr)
    rest = np.ones(n, bool)
    for i in range(s):
        rest &= read(sim2, A[i]) == np.array([c[0][i] for c in cases]); rest &= read(sim2, B[i]) == np.array([c[1][i] for c in cases])
    data = {q for w in A + B for q in [x.id for x in w]}
    junk = km.array([km.q(i) for i in range(cr.num_qubits) if i not in data])
    res[name] = dict(outA=outA, outB=outB, Tf=cf.operation_statistics().num_toffoli_like, Tr=cr.operation_statistics().num_toffoli_like - cf.operation_statistics().num_toffoli_like,
                     Q=cr.num_qubits, restored=rest, junk=sim2.read_across_shots(junk).any(axis=0) if len(junk) else np.zeros(n, bool),
                     phase=sim2.read_phase_flipped_across_shots())
labels = np.array(labels)
same = (res['orig']['outA'] == res['virt']['outA']).all(axis=0) & (res['orig']['outB'] == res['virt']['outB']).all(axis=0)
out = dict(Tf=[res[k]['Tf'] for k in res], Tr=[res[k]['Tr'] for k in res], Q=[res[k]['Q'] for k in res])
for k in kinds:
    sel = labels == k
    out[k] = dict(same=int(same[sel].sum()), n=int(sel.sum()),
                  bad_roundtrip={nm: int((~res[nm]['restored'][sel] | res[nm]['junk'][sel] | res[nm]['phase'][sel]).sum()) for nm in res})
print(json.dumps(out), flush=True)
