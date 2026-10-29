"""Greedy relation augmentation for stage-respecting J5 plans.

Start from a feasible relation set; repeatedly propose a random kernel vector of a random stage-respecting block,
shrink every plane's direction space by its equations, and accept the relation iff the template span still equals
the annihilator dimension (L*m - #relations).  Every accepted state is exact (the rows always lie in the
annihilator), so the result can be paired with mrmeows' quotient selection and built by j5_native.
"""
from __future__ import annotations

import itertools
import random
import time

import numpy as np

import j5layout as JL

core, binary = JL.core, JL.binary


def nullspace_bits(eqs, m):
    return core.nullspace(eqs, m) if eqs else [1 << i for i in range(m)]


class State:
    def __init__(self, tensor, plane_kinds=('pair', 'triple'), max_enum=6):
        self.T = tensor
        self.H, self.L, self.m = tensor.shape
        self.rows = [binary.packed(row) for row in tensor.reshape(self.H, self.L * self.m)]
        self.max_enum = max_enum
        planes = set()
        for live in JL.STAGES:
            if 'pair' in plane_kinds:
                for i, j in itertools.combinations(live, 2):
                    planes.add((1 << i, 1 << j))
            if 'triple' in plane_kinds:
                for i, j in itertools.combinations(live, 2):
                    for k in live:
                        if k not in (i, j):
                            planes.add(((1 << i) | (1 << j), 1 << k))
        self.planes = sorted(planes)
        self.V = {p: [1 << i for i in range(self.m)] for p in self.planes}   # direction-space bases
        self.rels = []
        self.rb = {}

    def chunks(self, r):
        m = self.m
        return [(r >> (j * m)) & ((1 << m) - 1) for j in range(self.L)]

    def restricted(self, r):
        """Direction spaces after adding relation r (not committed)."""
        cs = self.chunks(r)
        out = {}
        for (a, b), V in self.V.items():
            ea = 0
            for j in core.set_bits(a):
                ea ^= cs[j]
            eb = 0
            for j in core.set_bits(b):
                eb ^= cs[j]
            if not ea and not eb:
                out[(a, b)] = V
                continue
            # keep v in span(V) with ea.v = eb.v = 0: solve over the coordinates of V
            cons = []
            for e in (ea, eb):
                if e:
                    cons.append(sum(((bin(e & v).count('1') & 1) << k) for k, v in enumerate(V)))
            coeffs = core.nullspace([c for c in cons if c], len(V)) if any(cons) else [1 << k for k in range(len(V))]
            newV = []
            for c in coeffs:
                v = 0
                for k in core.set_bits(c):
                    v ^= V[k]
                newV.append(v)
            out[(a, b)] = newV
        return out

    def span_dim(self, Vmap):
        span = {}
        m = self.m
        for (a, b), V in Vmap.items():
            dirs = set(V)
            if len(V) <= self.max_enum:
                for code in range(1, 1 << len(V)):
                    d = 0
                    for j in core.set_bits(code):
                        d ^= V[j]
                    dirs.add(d)
            for v in dirs:
                if v:
                    binary.append_row(binary.tensor_column(v, a, m), span)
                    binary.append_row(binary.tensor_column(v, b, m), span)
        return len(span)

    def try_add(self, r):
        if not binary.reduce_row(r, dict(self.rb)):
            return False
        Vn = self.restricted(r)
        target = self.L * self.m - len(self.rb) - 1
        if self.span_dim(Vn) != target:
            return False
        binary.append_row(r, self.rb)
        self.rels.append(r)
        self.V = Vn
        return True

    def templates(self):
        m = self.m
        out = []
        for (a, b), V in self.V.items():
            dirs = set(V)
            if len(V) <= self.max_enum:
                for code in range(1, 1 << len(V)):
                    d = 0
                    for j in core.set_bits(code):
                        d ^= V[j]
                    dirs.add(d)
            for v in sorted(dirs):
                if v:
                    out.append((binary.tensor_column(v, a, m), binary.tensor_column(v, b, m), v, a, b))
        return out


def block_kernel(T, blk):
    m = T.shape[2]
    sub = np.concatenate([T[:, v, :] for v in blk], axis=1)
    kern = core.nullspace([binary.packed(row) for row in sub], len(blk) * m)
    lift = []
    for rel in kern:
        lift.append(sum(((rel >> (j * m)) & ((1 << m) - 1)) << (v * m) for j, v in enumerate(blk)))
    return lift


def vertex_spaces(st):
    """E_v: echelon dicts of the per-vertex equation rows of the current relations."""
    E = [dict() for _ in range(st.L)]
    for r in st.rels:
        for v, c in enumerate(st.chunks(r)):
            if c:
                binary.append_row(c, E[v])
    return E


def structured_candidates(st, T, blk, kern, rng, free_vertices=0, count=4):
    """Kernel vectors of the block whose chunks lie in the existing E_v except on up to `free_vertices` vertices."""
    if not kern:
        return []
    E = vertex_spaces(st)
    m = st.m
    blk = list(blk)
    out = []
    for _ in range(count):
        free = set(rng.sample(blk, free_vertices)) if free_vertices else set()
        cons = []          # for each constrained vertex and bit: one linear equation on the kernel coefficients
        red = []
        for t, r in enumerate(kern):
            cs = st.chunks(r)
            red.append([binary.reduce_row(cs[v], E[v]) if v not in free else 0 for v in blk])
        for vi, v in enumerate(blk):
            if v in free:
                continue
            for bit in range(m):
                row = 0
                for t in range(len(kern)):
                    if (red[t][vi] >> bit) & 1:
                        row |= 1 << t
                if row:
                    cons.append(row)
        sol = core.nullspace(cons, len(kern)) if cons else [1 << t for t in range(len(kern))]
        if not sol:
            continue
        code = 0
        while not code:
            code = rng.getrandbits(len(sol))
        c = 0
        for j in range(len(sol)):
            if code >> j & 1:
                c ^= sol[j]
        r = 0
        for t in core.set_bits(c):
            r ^= kern[t]
        if r:
            out.append(r)
    return out


def search(T, seed=0, max_rel=90, budget=600, block_sizes=(3, 4), log=None, stall=400, structured=True, block_mode='all'):
    rng = random.Random(seed)
    st = State(T)
    blocks = []
    for size in block_sizes:
        blocks += JL.blocks_in_stages(size)
    if block_mode == 'comp_addr':
        comp = set(range(15)) - set(JL.ADDR)
        blocks = [b for b in blocks if set(b) <= comp or set(b) <= set(JL.ADDR)]
        if tuple(sorted(JL.ADDR)) not in blocks:
            blocks.append(tuple(sorted(JL.ADDR)))
    elif block_mode == 'comp_addr_mix1':
        comp = set(range(15)) - set(JL.ADDR)
        blocks = [b for b in blocks if len(set(b) & set(JL.ADDR)) <= 1 or set(b) <= set(JL.ADDR)]
        if tuple(sorted(JL.ADDR)) not in blocks:
            blocks.append(tuple(sorted(JL.ADDR)))
    kernels = {blk: block_kernel(T, blk) for blk in blocks}
    blocks = [b for b in blocks if kernels[b]]
    t0 = time.time()
    fails = 0
    while len(st.rb) < max_rel and time.time() - t0 < budget and fails < stall:
        blk = rng.choice(blocks)
        K = kernels[blk]
        cands = []
        if structured:
            for fv in (0, 1, 2):
                cands = structured_candidates(st, T, blk, K, rng, free_vertices=fv, count=2)
                if cands:
                    break
        if not cands:
            r = 0
            while not r:
                code = rng.getrandbits(len(K))
                for j in range(len(K)):
                    if code >> j & 1:
                        r ^= K[j]
            cands = [r]
        ok = False
        for r in cands:
            if st.try_add(r):
                ok = True
                break
        if ok:
            fails = 0
            if log and len(st.rb) % 5 == 0:
                log(f'  relations {len(st.rb)} ({time.time() - t0:.0f}s)')
        else:
            fails += 1
    return st


def pair_plan(st, seed=0, trials=64):
    """Pair the current state's templates; returns stage terms or None (needs an even target)."""
    L, m, H = st.L, st.m, st.H
    target = L * m - len(st.rb)
    if target % 2:
        return None, {'odd': True}
    templates = st.templates()
    selected, _ = JL.quotient_selection(templates, target, L, seed + 7001, trials=trials)
    if selected is None:
        return None, {'paired': None, 'relations': len(st.rb)}
    basis = core.BinaryBasis()
    for i in selected:
        a, b, *_ = templates[i]
        assert basis.add(a) and basis.add(b)
    solutions = [basis.solve(row) for row in st.rows]
    assert all(s is not None for s in solutions)
    terms = [[] for _ in JL.STAGES]
    recon = [0] * H
    for j, i in enumerate(selected):
        a, b, v, ell1, ell2 = templates[i]
        h1 = sum(((s >> (2 * j)) & 1) << h for h, s in enumerate(solutions))
        h2 = sum(((s >> (2 * j + 1)) & 1) << h for h, s in enumerate(solutions))
        stg = JL.stage_of(ell1 | ell2)
        terms[stg].append({'target_direction': v, 'factors': [(h1, ell1), (h2, ell2)]})
        for h in core.set_bits(h1):
            recon[h] ^= a
        for h in core.set_bits(h2):
            recon[h] ^= b
    assert recon == st.rows
    return terms, {'relations': len(st.rb), 'paired': len(selected), 'per_stage': [len(t) for t in terms]}


def rebuild(T, rels):
    st = State(T)
    for r in rels:
        if not st.try_add(r):
            return None
    return st


def greedy_fill(st, T, blocks, kernels, rng, stall=300):
    fails = 0
    while fails < stall:
        blk = rng.choice(blocks)
        K = kernels[blk]
        cands = []
        for fv in (0, 1, 2):
            cands = structured_candidates(st, T, blk, K, rng, free_vertices=fv, count=2)
            if cands:
                break
        ok = any(st.try_add(r) for r in cands)
        fails = 0 if ok else fails + 1
    return st


def improve(st, T, seed=0, budget=600, block_mode='comp_addr_mix1', block_sizes=(3,), log=None):
    """Remove-and-refill local search: drop 1-2 random relations, greedily refill; keep if the count grows."""
    rng = random.Random(seed + 991)
    blocks = []
    for size in block_sizes:
        blocks += JL.blocks_in_stages(size)
    if block_mode == 'comp_addr_mix1':
        blocks = [b for b in blocks if len(set(b) & set(JL.ADDR)) <= 1 or set(b) <= set(JL.ADDR)]
        if tuple(sorted(JL.ADDR)) not in blocks:
            blocks.append(tuple(sorted(JL.ADDR)))
    kernels = {blk: block_kernel(T, blk) for blk in blocks}
    blocks = [b for b in blocks if kernels[b]]
    t0 = time.time()
    best = st
    while time.time() - t0 < budget:
        k = rng.choice((1, 1, 2))
        drop = set(rng.sample(range(len(best.rels)), k))
        cand = rebuild(T, [r for i, r in enumerate(best.rels) if i not in drop])
        if cand is None:
            continue
        cand = greedy_fill(cand, T, blocks, kernels, rng, stall=200)
        if len(cand.rb) > len(best.rb):
            best = cand
            if log:
                log(f'  improve: {len(best.rb)} relations ({time.time() - t0:.0f}s)')
    return best
