"""J=5 lookup planning under a multi-phase low cache (1366-row tables, 128 qubits).

Low monomials are indexed 0..14 (= low masks 1..15).  The address bits (masks 1, 2, 4, 8 -> indices 0, 1, 3, 7) are
always live; composites are live only in their stage.  A plan is a global relation set (each relation supported on
one block of low indices that lies inside a single stage's live set) plus paired templates ("planes": two lowforms,
one shared output direction) whose lowforms are live together in some stage.  Exactness: the template basis must
span the annihilator of the relations, which contains every tensor row.  Derived from mrmeows' kernel-relation
planner (Apache-2.0, chemistry_6p69 permuted_kernel / kernel_planes).
"""
from __future__ import annotations

import itertools
import random
import sys
from pathlib import Path

import numpy as np

REP = Path('/home/omen/vqcwork/m1575/ds/source/chemistry_6p69/reproduction')
sys.path[:0] = [str(REP)]
import rank_parity as core  # noqa: E402
import kernel_planes as binary  # noqa: E402
from permuted_kernel import quotient_selection  # noqa: E402

ADDR = [0, 1, 3, 7]                     # low masks 1, 2, 4, 8
MASK = {i: i + 1 for i in range(15)}    # low index -> low mask
IDX = {v: k for k, v in MASK.items()}
# three stages with at most four live composites; 12 stays live as the parent of 13 and 14
STAGES = [ADDR + [IDX[3], IDX[5], IDX[6], IDX[7]],
          ADDR + [IDX[9], IDX[10], IDX[11], IDX[12]],
          ADDR + [IDX[12], IDX[13], IDX[14], IDX[15]]]


def stage_of(lowform):
    bits = set(core.set_bits(lowform))
    for s, live in enumerate(STAGES):
        if bits <= set(live):
            return s
    return None


def plan_layout(tensor, blocks, wants, seed=0, plane_kinds=('pair', 'triple'), max_enum=6, trials=64):
    """tensor (H, 15, m); blocks: list of low-index tuples (each inside one stage); wants: relations per block."""
    H, L, m = tensor.shape
    rows = [binary.packed(row) for row in tensor.reshape(H, L * m)]
    rng = random.Random(seed)
    relations = []
    for blk, want in zip(blocks, wants):
        if want <= 0:
            continue
        sub = np.concatenate([tensor[:, v, :] for v in blk], axis=1)
        packed = [binary.packed(row) for row in sub]
        kern = core.nullspace(packed, len(blk) * m)
        want = min(want, len(kern))
        if want <= 0:
            continue
        selected, _ = binary.select_relations(packed, len(blk) * m, want, rng)
        for rel in selected:
            relations.append(sum(((rel >> (j * m)) & ((1 << m) - 1)) << (v * m) for j, v in enumerate(blk)))
    rb = {}
    for r in relations:
        binary.append_row(r, rb)
    rel_list = list(rb.values())
    chunks = [[(r >> (j * m)) & ((1 << m) - 1) for j in range(L)] for r in rel_list]
    planes = set()
    for live in STAGES:
        if 'pair' in plane_kinds:
            for i, j in itertools.combinations(live, 2):
                planes.add((1 << i, 1 << j))
        if 'triple' in plane_kinds:
            for i, j in itertools.combinations(live, 2):
                for k in live:
                    if k not in (i, j):
                        planes.add(((1 << i) | (1 << j), 1 << k))
        if 'quad' in plane_kinds:
            for i, j in itertools.combinations(live, 2):
                for k, l in itertools.combinations(live, 2):
                    if len({i, j, k, l}) == 4 and (i, j) < (k, l):
                        planes.add(((1 << i) | (1 << j), (1 << k) | (1 << l)))
        if 'tri3' in plane_kinds:
            for i, j, k in itertools.combinations(live, 3):
                for l in live:
                    if l not in (i, j, k):
                        planes.add(((1 << i) | (1 << j) | (1 << k), 1 << l))
    templates = []
    for a, b in sorted(planes):
        eq = []
        for cs in chunks:
            for ell in (a, b):
                value = 0
                for j in core.set_bits(ell):
                    value ^= cs[j]
                if value:
                    eq.append(value)
        V = core.nullspace(eq, m) if eq else [1 << i for i in range(m)]
        dirs = set(V)
        if len(V) <= max_enum:
            for code in range(1, 1 << len(V)):
                d = 0
                for j in core.set_bits(code):
                    d ^= V[j]
                dirs.add(d)
        for v in sorted(dirs):
            if v:
                templates.append((binary.tensor_column(v, a, m), binary.tensor_column(v, b, m), v, a, b))
    span = {}
    for a, b, *_ in templates:
        binary.append_row(a, span)
        binary.append_row(b, span)
    target = L * m - len(rb)
    contains = all(not binary.reduce_row(row, span) for row in rows)
    rep = {'relations': len(rb), 'target': target, 'span': len(span), 'contains': contains, 'templates': len(templates)}
    if len(span) != target or not contains or target % 2:
        return None, rep
    selected, attempts = quotient_selection(templates, target, L, seed + 7001, trials=trials)
    if selected is None:
        rep['paired'] = None
        return None, rep
    basis = core.BinaryBasis()
    for i in selected:
        a, b, *_ = templates[i]
        assert basis.add(a) and basis.add(b)
    solutions = [basis.solve(row) for row in rows]
    assert all(s is not None for s in solutions)
    terms = [[] for _ in STAGES]
    recon = [0] * H
    for j, i in enumerate(selected):
        a, b, v, ell1, ell2 = templates[i]
        h1 = sum(((s >> (2 * j)) & 1) << h for h, s in enumerate(solutions))
        h2 = sum(((s >> (2 * j + 1)) & 1) << h for h, s in enumerate(solutions))
        st = stage_of(ell1 | ell2)
        assert st is not None
        terms[st].append({'target_direction': v, 'factors': [(h1, ell1), (h2, ell2)]})
        for h in core.set_bits(h1):
            recon[h] ^= a
        for h in core.set_bits(h2):
            recon[h] ^= b
    assert recon == rows
    rep['paired'] = len(selected)
    rep['per_stage'] = [len(t) for t in terms]
    return terms, rep


def blocks_in_stages(size=3, include_cross=True):
    """All low-index blocks of a given size that lie inside one stage's live set."""
    out = set()
    for live in STAGES:
        for blk in itertools.combinations(sorted(live), size):
            if not include_cross and all(i in ADDR for i in blk):
                continue
            out.add(blk)
    return sorted(out)
