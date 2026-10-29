"""Exact five-word / 8-bit ANGLE lookup for 1366-address tables in 128 qubits.

Layout (after heliosli's published description; code derived from mrmeows' streamed_native, Apache-2.0):
- 11 selection + 40 word qubits + 77 allocated high-monomial cache qubits = 128;
- the word registers' bit 7 is a redundant copy of bit 6 for every table word (exact projective sign gauge),
  so CX(bit6 -> bit7) clears it at the start; the five cleared qubits hold four low-cache slots and one
  high-cache entry; outputs target bits 0..6 only and CX(bit6 -> bit7) restores bit 7 at the end;
- the 11 composite low monomials are cached in three phases with at most four live:
  {3,5,6,7} -> {9,10,11,12} -> {12,13,14,15}; every paired product uses lowforms live in its phase.
Correct for every input whose word registers hold sign-gauge words (bit 7 == bit 6), which is what the
official verifier loads (the table's own words).
"""
from __future__ import annotations

import kickmix as km

from j5layout import STAGES

LOW_PHASES = [[3, 5, 6, 7], [9, 10, 11, 12], [12, 13, 14, 15]]


def set_bits(v):
    while v:
        b = (v & -v).bit_length() - 1
        yield b
        v &= v - 1


def build_j5(delta, ac, stage_terms, *, brot=8):
    D, J = delta.shape
    if J != 5 or brot != 8:
        raise ValueError('five eight-bit words expected')
    A, B, O = ac.shape
    if B != 16 or O != 35:
        raise ValueError('expected 4 low bits and 35 outputs')
    n = (D - 1).bit_length()
    H = A - 1
    builder = km.CircuitBuilder()
    qs = list(builder.create_quantum_register(n, 'selection'))
    for j in range(J):
        qs += list(builder.create_quantum_register(brot, f'word_{j}'))
    bit6 = [n + j * brot + 6 for j in range(J)]
    bit7 = [n + j * brot + 7 for j in range(J)]
    emitted = 0

    def gate(name, *ids):
        nonlocal emitted
        getattr(builder, name.lower())(*[qs[i] for i in ids])
        emitted += (name == 'CCX')

    for j in range(J):                       # clear the redundant sign copies
        gate('CX', bit6[j], bit7[j])
    scratch = list(bit7)                     # 5 clean qubits
    composites = [m for m in range(1, A) if m.bit_count() > 1]
    high = {0: None}
    high_cache = []
    borrowed_high = composites[-1]           # one high-cache entry lives in a cleared bit-7 qubit
    for mask in range(1, A):
        bit = mask.bit_length() - 1
        if mask.bit_count() == 1:
            high[mask] = 4 + bit
        else:
            parent = mask ^ (1 << bit)
            if mask == borrowed_high:
                q = scratch.pop()
            else:
                q = len(qs)
                qs.extend(builder.alloc_qubits(1))
            high[mask] = q
            triple = (high[parent], 4 + bit, q)
            gate('CCX', *triple)
            high_cache.append(triple)
    if len(qs) > 128:
        raise ValueError(f'layout requires {len(qs)} qubits')
    slots = list(scratch)                    # four low-cache slots
    low = {1: 0, 2: 1, 4: 2, 8: 3}           # low mask -> qubit (address bits)
    low_triples = {}

    def create(mask, parent):
        other = mask ^ parent
        assert other.bit_count() == 1 and parent in low and other in low
        q = slots.pop()
        triple = (low[parent], low[other], q)
        gate('CCX', *triple)
        low[mask] = q
        low_triples[mask] = triple

    def erase(mask):
        a, b, q = low_triples.pop(mask)
        builder.del_and(qs[a], qs[b], qs[q])
        del low[mask]
        slots.append(q)

    groups = [[n + j * brot + b] for j in range(J) for b in range(brot - 1)]   # bit 7 is restored at the end

    def targets(direction):
        return [q for bit in set_bits(direction) for q in groups[bit]]

    def linear(control, direction):
        for q in targets(direction):
            gate('CX', control, q)

    def product(left, right, direction, basis):
        if not left or not right or not direction:
            return
        if left == right:
            for i in set_bits(left):
                linear(basis[i], direction)
            return
        p = (left & -left).bit_length() - 1
        first = list(set_bits(left ^ (1 << p)))
        for i in first:
            gate('CX', basis[i], basis[p])
        if (right >> p) & 1:
            right ^= left ^ (1 << p)
        choices = right & ~(1 << p)
        if not choices:
            raise AssertionError('dependent controls')
        q = (choices & -choices).bit_length() - 1
        second = list(set_bits(right ^ (1 << q)))
        for i in second:
            gate('CX', basis[i], basis[q])
        pivot, *others = targets(direction)
        for out in others:
            gate('CX', pivot, out)
        gate('CCX', basis[p], basis[q], pivot)
        for out in others:
            gate('CX', pivot, out)
        for i in reversed(second):
            gate('CX', basis[i], basis[q])
        for i in reversed(first):
            gate('CX', basis[i], basis[p])

    high_only = [sum(int(ac[h, 0, j]) << j for j in range(O)) for h in range(A)]
    low_only = [sum(int(ac[0, l, j]) << j for j in range(O)) for l in range(B)]
    low_only[0] = 0

    # classical same-side corrections of the pairing identity, before emission
    for terms in stage_terms:
        for term in terms:
            fs, d = term['factors'], term['target_direction']
            for i in range(0, len(fs) - 1, 2):
                (h1, l1), (h2, l2) = fs[i], fs[i + 1]
                for a in set_bits(h1):
                    for b in set_bits(h2):
                        idx = (a + 1) | (b + 1)
                        if idx < A:
                            high_only[idx] ^= d
                for a in set_bits(l1):
                    for b in set_bits(l2):
                        idx = (a + 1) | (b + 1)
                        if idx < B:
                            low_only[idx] ^= d

    emitted_low = set()

    def emit_stage(s, terms):
        live = STAGES[s]                                   # global low indices live in this stage
        pos = {g: k for k, g in enumerate(live)}
        basis = [high[h] for h in range(1, A)] + [low[g + 1] for g in live]
        for term in terms:
            fs, d = term['factors'], term['target_direction']

            def loc(lf):
                return sum(1 << pos[g] for g in set_bits(lf))
            for i in range(0, len(fs), 2):
                h1, l1 = fs[i]
                if i + 1 == len(fs):
                    product(h1, loc(l1) << H, d, basis)
                else:
                    h2, l2 = fs[i + 1]
                    product(h1 ^ (loc(l2) << H), h2 ^ (loc(l1) << H), d, basis)
        for g in live:                                     # linear low outputs once, while live
            mask = g + 1
            if mask not in emitted_low and mask in low:
                linear(low[mask], low_only[mask])
                emitted_low.add(mask)

    create(3, 1); create(5, 1); create(6, 2); create(7, 3)
    emit_stage(0, stage_terms[0])
    erase(7); erase(3); erase(5); erase(6)
    create(9, 1); create(10, 2); create(12, 4); create(11, 9)
    emit_stage(1, stage_terms[1])
    erase(11); erase(9); erase(10)
    create(13, 12); create(14, 12); create(15, 13)
    emit_stage(2, stage_terms[2])
    erase(15); erase(14); erase(13); erase(12)
    assert emitted_low == set(range(1, 16)), sorted(set(range(1, 16)) - emitted_low)
    for q in targets(high_only[0]):
        gate('X', q)
    for h in range(1, A):
        linear(high[h], high_only[h])
    for a, b, q in reversed(high_cache):
        builder.del_and(qs[a], qs[b], qs[q])
    for j in range(J):                       # restore the redundant sign copies
        gate('CX', bit6[j], bit7[j])
    c = builder.finish_circuit()
    native = int(c.operation_statistics().num_toffoli_like)
    assert native == emitted
    return c, {'T': native, 'Q': int(c.num_qubits)}
