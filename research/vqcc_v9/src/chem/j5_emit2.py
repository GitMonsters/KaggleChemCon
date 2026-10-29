"""J=5 ANGLE family (15 circuits) for 1366-row tables: stage-respecting relation layout (j5layout) + three-phase
low cache with bit-7 scratch (j5_native); each circuit is checked natively on every address with sign-gauge
inputs (bit 7 == bit 6, as the verifier loads), forward and reverse.

Usage: python j5_emit.py <npz> <out dir> [jobs] [relations] [seeds]
"""
from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path[:0] = ['/mnt/c/Users/omen/Desktop/quantumAI/paradigm/chem', '/home/omen/vqcwork/comp/kickmix']


def check(circuit, delta, repeats=4, seed=1):
    import kickmix as km
    count, J = delta.shape
    n = (count - 1).bit_length()
    addresses = np.tile(np.arange(count), repeats)
    rng = np.random.default_rng(seed)
    init = rng.integers(0, 128, size=(len(addresses), J))
    init = init | ((init >> 6) & 1) << 7          # bit 7 = bit 6 (sign-gauge words)
    sim = km.Simulator(batch_size=len(addresses), seed=seed, counting_operations=True)
    sim.use_same_registers_as(circuit)
    sim.write_across_shots('selection', ((addresses[None, :] >> np.arange(n)[:, None]) & 1).astype(bool))
    for j in range(J):
        sim.write_across_shots(f'word_{j}', ((init[:, j][None, :] >> np.arange(8)[:, None]) & 1).astype(bool))
    powers = np.int64(1) << np.arange(8)
    for application in range(2):
        sim.do(circuit)
        got = np.stack([powers @ sim.read_across_shots(f'word_{j}').astype(np.int64) for j in range(J)], axis=1)
        expected = init ^ delta[addresses] if application == 0 else init
        assert np.array_equal(got, expected), 'wrong XOR'
        assert not np.any(sim.read_phase_flipped_across_shots()), 'phase'
        dec = (np.int64(1) << np.arange(n)) @ sim.read_across_shots('selection').astype(np.int64)
        assert np.array_equal(dec, addresses), 'selection changed'
        total = sim.read_across_shots(km.array([km.q(i) for i in range(circuit.num_qubits)]))
        assert not np.any(total[n + 8 * J:]), 'scratch not clean'
    return float(sim.operation_statistics().num_toffoli_like) / len(addresses) / 2


def build_one(args):
    k, words, out, nrel, seeds = args
    import kickmix as km
    import j5layout as JL
    import j5relsearch as RS
    from j5_native import build_j5
    core = JL.core
    t0 = time.time()
    delta = core.delta_batch(words, 5, k)
    ac = core.coefficients(delta, 8, 4)
    T = ac[1:, 1:, :]
    best = None
    for seed in range(seeds):
        st = RS.search(T, seed=seed, budget=600, block_mode='comp_addr_mix1', block_sizes=(3,), stall=2500)
        for _ in range(8):
            terms, rep = RS.pair_plan(st, seed=seed)
            if terms is not None:
                break
            rels = st.rels[:-1]
            st2 = RS.State(T)
            for r in rels:
                assert st2.try_add(r)
            st = st2
        if terms is not None and (best is None or rep['paired'] < best[1]['paired']):
            best = (terms, rep)
    if best is None:                          # fall back to the fixed cyclic layout
        live_cyc = [tuple(live[(i + d) % len(live)] for d in range(3)) for live in JL.STAGES for i in range(len(live))]
        for total in nrel:
            per, extra = divmod(total, len(live_cyc))
            wants = [per + (1 if i < extra else 0) for i in range(len(live_cyc))]
            terms, rep = JL.plan_layout(T, live_cyc, wants, seed=0, plane_kinds=('pair', 'triple'))
            if terms is not None:
                best = (terms, rep)
                break
    if best is None:
        return {'k': k, 'error': 'no plan'}
    terms, rep = best
    circuit, stats = build_j5(delta, ac, terms)
    path = Path(out) / f'angle_k_{k}.kmb'
    circuit.to_file(path, format='kmb')
    circuit = km.Circuit.from_file(path, format='kmb')
    native = check(circuit, delta, repeats=2, seed=7 + k)
    assert native == stats['T'], (native, stats['T'])
    return {'k': k, 'T': int(stats['T']), 'Q': int(circuit.num_qubits), **{x: rep[x] for x in ('relations', 'paired', 'per_stage')},
            'seconds': round(time.time() - t0, 1)}


def main():
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    jobs = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    nrel = [int(x) for x in sys.argv[4].split(',')] if len(sys.argv) > 4 else [49, 47, 45, 41]
    seeds = int(sys.argv[5]) if len(sys.argv) > 5 else 3
    words = np.asarray(np.load(src)['angle_words'], dtype=np.int64)
    out.mkdir(parents=True, exist_ok=True)
    nb = (words.shape[1] + 4) // 5
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        reports = list(pool.map(build_one, [(k, words, str(out), nrel, seeds) for k in range(nb)]))
    total = sum(r.get('T', 0) for r in reports)
    (out / 'plans.json').write_text(json.dumps({'rows': int(words.shape[0]), 'per_pass': total, 'circuits': reports}, indent=1))
    print(json.dumps({'per_pass': total, 'T_both': 2 * total, 'per_circuit': [r.get('T') for r in reports],
                      'Q': [r.get('Q') for r in reports], 'errors': [r for r in reports if 'error' in r]}))


if __name__ == '__main__':
    main()
