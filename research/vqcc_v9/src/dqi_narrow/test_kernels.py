"""Native tests of the narrow jet pair and narrow outer division against the originals.

Run with cwd = SJ source root (blocks/dqi/team-fused/source), PYTHONPATH = official kickmix.
"""
import sys, json
from pathlib import Path
import numpy as np
ROOT = Path.cwd(); sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'exp/decoder'))
import kickmix as km
from exp.decoder.common import K, write, read
from exp.jet24.kernel import jet_forward as jf_old
from exp.narrow.kernel import jet_forward_new, jet_reverse_new, divide_zero_narrow
from exp.fold_cached_norm.kernel import divide_zero324

rng = np.random.default_rng(1027)
shots = int(sys.argv[1]) if len(sys.argv) > 1 else 512


def run_jets():
    uvals = rng.integers(0, 1 << 24, shots); vvals = rng.integers(0, 1 << 24, shots)
    uvals[:16] = 0; uvals[16:32] = 1; vvals[32:40] = 0
    out = {}
    for name, fwd in [('old', jf_old), ('new', jet_forward_new)]:
        b = km.CircuitBuilder(); u = b.alloc_qubits(24); v = b.alloc_qubits(24)
        tok = fwd(b, u, v, K)
        c = b.finish_circuit()
        sim = km.Simulator(batch_size=shots, seed=5); sim.use_same_registers_as(c)
        write(sim, u, uvals); write(sim, v, vvals); sim.do(c)
        out[name] = (read(sim, u), read(sim, v), c.num_qubits, c.operation_statistics().num_toffoli_like)
    same = bool((out['old'][0] == out['new'][0]).all() and (out['old'][1] == out['new'][1]).all())
    # full round trip with the new pair
    b = km.CircuitBuilder(); u = b.alloc_qubits(24); v = b.alloc_qubits(24)
    tok = jet_forward_new(b, u, v, K); c1 = b.finish_circuit(); T1 = c1.operation_statistics().num_toffoli_like; Q1 = c1.num_qubits
    jet_reverse_new(b, u, v, tok, K)
    c = b.finish_circuit()
    sim = km.Simulator(batch_size=shots, seed=7); sim.use_same_registers_as(c)
    write(sim, u, uvals); write(sim, v, vvals); sim.do(c)
    junk = km.array([km.q(i) for i in range(48, c.num_qubits)])
    r = dict(forward_equals_original=same, fwd_T_old=out['old'][3], fwd_T_new=T1, fwd_Q_old=out['old'][2], fwd_Q_new=Q1,
             pair_T=c.operation_statistics().num_toffoli_like, pair_Q=c.num_qubits,
             restored=bool((read(sim, u) == uvals).all() and (read(sim, v) == vvals).all()),
             scratch=bool(sim.read_across_shots(junk).any()) if len(junk) else False,
             phase=int(sim.read_phase_flipped_across_shots().sum()))
    print('jets', json.dumps(r), flush=True)
    return r


def run_divide():
    xvals = rng.integers(0, 1 << 24, shots); yvals = rng.integers(0, 1 << 24, shots); yvals[:24] = 0
    res = {}
    for name, fn in [('old', divide_zero324), ('new', divide_zero_narrow)]:
        b = km.CircuitBuilder(); x = b.alloc_qubits(24); y = b.alloc_qubits(24)
        fn(b, x, y, K=K)
        c = b.finish_circuit()
        sim = km.Simulator(batch_size=shots, seed=9); sim.use_same_registers_as(c)
        write(sim, x, xvals); write(sim, y, yvals); sim.do(c)
        junk = km.array([km.q(i) for i in range(48, c.num_qubits)])
        res[name] = dict(T=c.operation_statistics().num_toffoli_like, Q=c.num_qubits, x=read(sim, x), y_ok=bool((read(sim, y) == yvals).all()),
                         scratch=bool(sim.read_across_shots(junk).any()) if len(junk) else False, phase=int(sim.read_phase_flipped_across_shots().sum()))
    r = dict(same_x=bool((res['old']['x'] == res['new']['x']).all()), identity_at_zero=bool((res['new']['x'][:24] == xvals[:24]).all()),
             **{f'{k}_{n}': v for n in res for k, v in res[n].items() if k != 'x'})
    print('divide', json.dumps(r), flush=True)
    return r


run_jets(); run_divide()
