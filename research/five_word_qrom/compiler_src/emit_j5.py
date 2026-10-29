# Derived from Ahmed Berat Ozer v8 emit_family.py (Apache 2.0).
# New experiment: use five 8-bit outputs in freed QROM workspace;
# select five relations per low-address block so the paired dimension is even.
"""Derived J=5 / 8-bit ANGLE family for an arbitrary row count with mrmeows' exact paired-product lookup
(Apache-2.0, source/chemistry_6p69/reproduction), generalized: the number of kernel relations per three-vertex
block is a search parameter, so tables with fewer high-address rows (fewer vectors) use their larger kernel.

Usage: python emit_family.py <words .npy or submission .npz> <out dir> [jobs]
Writes angle_k_*.kmb and plans.json; every circuit is natively checked on all addresses (64 targets each).
"""
from __future__ import annotations

import json
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

REP=Path(__file__).resolve().parent
sys.path.insert(0,str(REP))
import rank_parity as core  # noqa: E402
import kernel_planes as binary  # noqa: E402
from permuted_kernel import quotient_selection  # noqa: E402


def plan_general(delta, rel_base=3, extra=15, seed=0, permutation=None):
    permutation = list(range(15)) if permutation is None else list(permutation)
    original_ac = core.coefficients(delta, 8, 4)
    ac = original_ac.copy()
    ac[:, 1:, :] = ac[:, 1:, :][:, permutation, :]
    tensor = ac[1:, 1:, :]
    H, L, m = tensor.shape
    assert L == 15
    rows = [binary.packed(row) for row in tensor.reshape(H, L * m)]
    rng = random.Random(seed)
    relations = []
    chosen_extra = set(rng.sample(range(L), min(extra, L)))
    for first in range(L):
        vs = [(first + i) % L for i in range(3)]
        block = np.concatenate([tensor[:, v, :] for v in vs], axis=1)
        want = rel_base + (first in chosen_extra)
        kern = core.nullspace([binary.packed(row) for row in block], 3 * m)
        want = min(want, len(kern))
        selected, _ = binary.select_relations([binary.packed(row) for row in block], 3 * m, want, rng)
        for rel in selected:
            relations.append(sum(((rel >> (j * m)) & ((1 << m) - 1)) << (v * m) for j, v in enumerate(vs)))
    rb = {}
    for r in relations:
        binary.append_row(r, rb)
    chunks = [[(r >> (j * m)) & ((1 << m) - 1) for j in range(L)] for r in relations]
    planes = [(1 << i, 1 << j) for i in range(L) for j in range(i + 1, L)]
    for step in [1, 2]:
        planes.extend(((1 << i) | (1 << ((i + step) % L)), 1 << k) for i in range(L) for k in range(L)
                      if k not in [i, (i + step) % L])
    templates = []
    for a, b in planes:
        eq = []
        for cs in chunks:
            for ell in [a, b]:
                value = 0
                for j in core.set_bits(ell):
                    value ^= cs[j]
                if value:
                    eq.append(value)
        V = core.nullspace(eq, m)
        directions = set(V)
        if len(V) <= 6:
            for code in range(1, 1 << len(V)):
                direction = 0
                for j in core.set_bits(code):
                    direction ^= V[j]
                directions.add(direction)
        for v in sorted(directions):
            if v:
                templates.append((binary.tensor_column(v, a, m), binary.tensor_column(v, b, m), v, a, b))
    span = {}
    for a, b, *_ in templates:
        binary.append_row(a, span)
        binary.append_row(b, span)
    target = L * m - len(rb)
    contains = all(not binary.reduce_row(row, span) for row in rows)
    if len(span) != target or not contains or target % 2:
        return None, {'target': target, 'span': len(span), 'contains': contains}
    selected, _ = quotient_selection(templates, target, L, seed + 7001)
    if selected is None:
        return None, {'target': target, 'paired': None}
    basis = core.BinaryBasis()
    for i in selected:
        a, b, *_ = templates[i]
        assert basis.add(a) and basis.add(b)
    solutions = [basis.solve(row) for row in rows]
    assert all(s is not None for s in solutions)
    terms = []
    reconstructed = [0] * H
    for j, i in enumerate(selected):
        a, b, v, ell1, ell2 = templates[i]
        h1 = sum(((s >> (2 * j)) & 1) << h for h, s in enumerate(solutions))
        h2 = sum(((s >> (2 * j + 1)) & 1) << h for h, s in enumerate(solutions))
        terms.append({'target_direction': v, 'factors': [(h1, ell1), (h2, ell2)]})
        for h in core.set_bits(h1):
            reconstructed[h] ^= a
        for h in core.set_bits(h2):
            reconstructed[h] ^= b
    assert reconstructed == rows
    for term in terms:
        term['factors'] = [(h, sum(1 << permutation[j] for j in core.set_bits(l))) for h, l in term['factors']]
    return (original_ac, terms), {'target': target, 'paired': len(selected), 'relations': len(rb),
                                  'rel_base': rel_base, 'extra': extra, 'seed': seed}


def best_plan(delta, budget_seconds=600, rel_bases=(4,)):
    """mrmeows' accepted plans all use 3 relations per block plus one extra on every block (60 in all),
    with a searched seed and low-monomial permutation; search the same space until a plan exists."""
    t0 = time.time()
    rng = random.Random(12345)
    attempt = 0
    while time.time() - t0 < budget_seconds:
        perm = list(range(15))
        if attempt:
            rng.shuffle(perm)
        for rel_base in rel_bases:
            for seed in range(8):
                try:
                    res, rep = plan_general(delta, rel_base, 15, seed, perm)
                except (ValueError, AssertionError):
                    continue
                if res is not None:
                    rep['permutation'] = perm
                    return res, rep
        attempt += 1
    return None


def build_one(args):
    k, words, out = args
    import kickmix as km
    from hyper_native import build_hmr
    from native_checks import check
    delta = core.delta_batch(words, 5, k)
    res = best_plan(delta)
    if res is None:
        return {'k': k, 'error': 'no plan'}
    (ac, terms), rep = res
    circuit, _, resources = build_hmr(delta, 8, 4, ac, terms)
    path = Path(out) / f'angle_k_{k}.kmb'
    circuit.to_file(path, format='kmb')
    circuit = km.Circuit.from_file(path, format='kmb')
    native = check(circuit, delta, brot=8, repeats=64, seed=150007 + k)
    assert native == resources['toffoli_count']
    return {'k': k, 'T': native, 'Q': int(circuit.num_qubits), **rep}


def main():
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    jobs = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    words = np.load(src)['angle_words'] if src.suffix == '.npz' else np.load(src)
    words = np.asarray(words, dtype=np.int64)
    out.mkdir(parents=True, exist_ok=True)
    K = (words.shape[1] + 4) // 5
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        reports = list(pool.map(build_one, [(k, words, str(out)) for k in range(K)]))
    total = sum(r.get('T', 0) for r in reports)
    (out / 'plans.json').write_text(json.dumps({'rows': int(words.shape[0]), 'per_pass': total, 'T_both_passes': 2 * total,
                                                'circuits': reports}, indent=1))
    print(json.dumps({'rows': int(words.shape[0]), 'per_pass': total, 'T_both': 2 * total,
                      'per_circuit': [r.get('T') for r in reports], 'paired': [r.get('paired') for r in reports]}))


if __name__ == '__main__':
    main()
