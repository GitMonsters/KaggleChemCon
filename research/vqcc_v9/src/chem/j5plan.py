"""Per-stage exact paired-product planner for J=5 / 8-bit lookups (35 outputs), generalized from mrmeows'
kernel-relation planner (Apache-2.0, chemistry_6p69 permuted_kernel / kernel_planes): the stage tensor covers only
the low monomials live in that stage; relations come from cyclic 3-blocks of those monomials.

plan_stage(tensor (H, Ls, m), rel_base, extra, seed, perm) -> (terms, report) with terms over stage-local
low indices; every plan reconstructs the stage tensor exactly over GF(2).
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

REP = Path('/home/omen/vqcwork/m1575/ds/source/chemistry_6p69/reproduction')
sys.path[:0] = [str(REP)]
import rank_parity as core  # noqa: E402
import kernel_planes as binary  # noqa: E402
from permuted_kernel import quotient_selection  # noqa: E402


def plan_stage(tensor, rel_base=4, extra=0, seed=0, perm=None, block=3, steps=(1, 2), trials=64):
    H, L, m = tensor.shape
    perm = list(range(L)) if perm is None else list(perm)
    t = tensor[:, perm, :]
    rows = [binary.packed(row) for row in t.reshape(H, L * m)]
    rng = random.Random(seed)
    relations = []
    if L >= block and rel_base + extra > 0:
        chosen_extra = set(rng.sample(range(L), min(extra, L)))
        firsts = range(L) if L > block else [0]
        for first in firsts:
            vs = [(first + i) % L for i in range(block)]
            blk = np.concatenate([t[:, v, :] for v in vs], axis=1)
            packed = [binary.packed(row) for row in blk]
            kern = core.nullspace(packed, block * m)
            want = min(rel_base + (first in chosen_extra), len(kern))
            if want <= 0:
                continue
            selected, _ = binary.select_relations(packed, block * m, want, rng)
            for rel in selected:
                relations.append(sum(((rel >> (j * m)) & ((1 << m) - 1)) << (v * m) for j, v in enumerate(vs)))
    rb = {}
    for r in relations:
        binary.append_row(r, rb)
    chunks = [[(r >> (j * m)) & ((1 << m) - 1) for j in range(L)] for r in relations]
    planes = [(1 << i, 1 << j) for i in range(L) for j in range(i + 1, L)]
    for step in steps:
        if L > 2:
            planes.extend(((1 << i) | (1 << ((i + step) % L)), 1 << k) for i in range(L) for k in range(L)
                          if k not in [i, (i + step) % L] and (i + step) % L != i)
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
        V = core.nullspace(eq, m) if eq else [1 << i for i in range(m)]
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
    if len(span) != target or not contains:
        return None, {'target': target, 'span': len(span), 'contains': contains, 'relations': len(rb)}
    if target % 2:
        return None, {'target': target, 'odd': True}
    selected, _ = quotient_selection(templates, target, L, seed + 7001, trials=trials)
    if selected is None:
        return None, {'target': target, 'paired': None, 'relations': len(rb)}
    basis = core.BinaryBasis()
    for i in selected:
        a, b, *_ = templates[i]
        assert basis.add(a) and basis.add(b)
    solutions = [basis.solve(row) for row in rows]
    assert all(s is not None for s in solutions)
    terms = []
    recon = [0] * H
    for j, i in enumerate(selected):
        a, b, v, ell1, ell2 = templates[i]
        h1 = sum(((s >> (2 * j)) & 1) << h for h, s in enumerate(solutions))
        h2 = sum(((s >> (2 * j + 1)) & 1) << h for h, s in enumerate(solutions))
        terms.append({'target_direction': v, 'factors': [(h1, ell1), (h2, ell2)]})
        for h in core.set_bits(h1):
            recon[h] ^= a
        for h in core.set_bits(h2):
            recon[h] ^= b
    assert recon == rows
    for term in terms:   # map permuted local low indices back
        term['factors'] = [(h, sum(1 << perm[j] for j in core.set_bits(l))) for h, l in term['factors']]
    return terms, {'target': target, 'paired': len(selected), 'relations': len(rb), 'rel_base': rel_base,
                   'extra': extra, 'seed': seed}
