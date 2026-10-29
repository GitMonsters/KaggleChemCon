"""A four-address ANGLE_k circuit: one lookup step per four rows, three words per circuit, <= 128 qubits.

The hosts' reference ANGLE circuit loads J = 10 words of 10 bits per circuit (8 circuits),
which keeps 100 word qubits alive through the block encoding. This one loads J = 3 (25
circuits, 30 word qubits) at about the same Toffoli count, by iterating over a = 4h + l only
once per high address h:

1. **Load** (`_anf_load`, at most 6 AND ancillas): XOR f(4h) into the word registers, and three
   differences into a clean 81-qubit bank,
       D1 = f(4h) ^ f(4h+1),  D2 = f(4h) ^ f(4h+2),  D3 = f(4h) ^ f(4h+1) ^ f(4h+2) ^ f(4h+3).
   In the signed Givens gauge (theta/pi in [-1/2, 1/2)) bit 9 of every word equals bit 8, and
   so does bit 9 of any XOR of words, so each difference word needs only 9 bank qubits. The table
   is loaded in algebraic normal form: high addresses past the last row never occur, so they are
   don't-cares that cancel every monomial past it; one Toffoli per remaining monomial of degree >= 2,
   about H - 10 for H high addresses (a unary iteration costs about H + 2).
2. **Combine** (`_paired_combine`): add l0 D1 ^ l1 D2 ^ (l0 l1) D3 to the words with 41 Toffolis
   (81 directly); a bank bit 8 also flips word bit 9 through a CNOT fan-out at no extra Toffoli.
3. **Erase** the bank by measurement (`_erase`): like kickmix's `del_lookup`, but the phase lookup
   covers only the rows that exist, the low address bits are unfolded one-hot (the last entry per
   level by CNOTs) and the phases over the high bits are applied in algebraic normal form (about 25
   Toffolis at 319 high addresses, against 41).

ANF loading and erasure appear in the public banked-QROM code (`banked_lab.py` in the group-
compression notebook, after Kozykorpesh and Sumit Chongder); this is our own construction for the
four-address circuit. `load='unary'` keeps the earlier unary iteration: kickmix's own holds one
ancilla per address bit (9, and 11 + 30 + 81 + 9 = 131 > 128), so the top three address bits
become one control per block of 64 high addresses, visited in a Gray-code order (one Toffoli per
move between neighbours); the ANF load falls back to it if a deep path would need a 7th ancilla.
"""
from __future__ import annotations

import numpy as np
import kickmix as km

BROT, J, BANK_BITS = 10, 3, 9


def _bits(value: int) -> list[int]:
    out, i = [], 0
    while value:
        if value & 1:
            out.append(i)
        value >>= 1
        i += 1
    return out


def _block_path(n: int) -> list[int]:
    """An order of blocks 0..n-1 (n <= 8) minimizing the moves: 1 Toffoli between blocks one bit
    apart, 3 otherwise (erase and recompute). Dynamic programming over visited sets."""
    cost = {(1 << p, p): (0, [p]) for p in range(n)}
    for mask in range(1, 1 << n):
        for p in range(n):
            if (mask, p) not in cost:
                continue
            c0, path = cost[(mask, p)]
            for q in range(n):
                if mask >> q & 1:
                    continue
                step = 1 if bin(p ^ q).count('1') == 1 else 3
                key = (mask | 1 << q, q)
                if key not in cost or c0 + step < cost[key][0]:
                    cost[key] = (c0 + step, path + [q])
    full = (1 << n) - 1
    return min((cost[(full, p)] for p in range(n)), key=lambda v: v[0])[1]


def _erase(b: km.CircuitBuilder, bank: list, addr, table: list[int]) -> None:
    """Measurement-based erase of `bank` = table[h] (h < len(table)), like kickmix's del_lookup,
    but the phase lookup only covers the rows that exist.

    X-measure the bank, then flip the phase of address h by P[h] = parity(m & table[h]). The low k
    address bits are unfolded into 2**k one-hot qubits u (reusing the measured bank qubits). The phase
    is then applied monomial by monomial of the high bits (the algebraic normal form of P over them):
    for each monomial, Z on the parity of the u_l it selects (classically conditioned CNOTs into one
    accumulator), controlled by the monomial's product: free for degrees 0 and 1 (Z, CZ), one CCZ or
    AND per monomial of higher degree.
    """
    H, hb, n = len(table), len(addr), len(bank)
    m = b.alloc_bits(n)
    b.hmr(km.array(bank), m)
    P = b.alloc_bits(H)
    b.bit_store(P, False)
    for j in range(n):
        hs = [P[h] for h in range(H) if table[h] >> j & 1]
        if hs:
            b.push_condition(m[j])
            b.cx(True, km.array(hs))
            b.pop_condition()

    def cost(k: int) -> int:  # unfolding, plus one Toffoli per high monomial of degree >= 2
        V = -(-H >> k)
        return (1 << k) - k - 1 + V - 1 - sum(1 for j in range(hb - k) if 1 << j < V)

    k = min(range(1, hb + 1), key=cost)
    V = -(-H >> k)
    u, acc, pool = bank[:1 << k], bank[1 << k], list(bank[(1 << k) + 1:])
    hi = addr[k:]

    def split(i: int, forward: bool) -> None:
        """u_l -> (u_l & ~a_i, u_l & a_i) for l < 2**i. The new halves XOR to a_i, so the last
        one is a_i ^ (the others) by CNOTs: 2**i - 1 Toffolis; the reverse is measurement-based."""
        s, last = 1 << i, (1 << i) - 1

        def fill_last() -> None:  # u[last + s] ^= a_i ^ (u[l + s] for l < last)
            for q in [addr[i]] + [u[l + s] for l in range(last)]:
                b.cx(q, u[last + s])

        if forward:
            for l in range(last):
                b.init_and(u[l], addr[i], u[l + s])
                b.cx(u[l + s], u[l])
            fill_last()
            b.cx(u[last + s], u[last])
        else:
            b.cx(u[last + s], u[last])
            fill_last()
            for l in reversed(range(last)):
                b.cx(u[l + s], u[l])
                b.del_and(u[l], addr[i], u[l + s])

    b.x(u[0])
    for i in range(k):
        split(i, True)

    # The phase of address h = v 2**k + l is P[h] = XOR over monomials m of the high bits (m subset
    # of v) of Q[m][l]: the Moebius transform of P over v, in classical bits (rows past H count as 0,
    # and monomials m >= V are dropped, which is exact on the rows that exist).
    Q = [b.alloc_bits(1 << k) for _ in range(V)]
    for v in range(V):
        for l in range(1 << k):
            b.bit_store(Q[v][l], False)
            if (v << k) + l < H:
                b.cx(P[(v << k) + l], Q[v][l])
    for i in range(len(hi)):
        for mm in range(V):
            if mm >> i & 1:
                for l in range(1 << k):
                    b.cx(Q[mm ^ (1 << i)][l], Q[mm][l])

    def parity(mm: int) -> None:  # acc ^= XOR of the u_l with Q[mm][l]
        for l in range(1 << k):
            b.push_condition(Q[mm][l])
            b.cx(u[l], acc)
            b.pop_condition()

    def phase(ctrls: list, mm: int) -> None:  # (-1)^(AND of ctrls) * (acc after parity(mm))
        parity(mm)
        if not ctrls:
            b.z(acc)
        elif len(ctrls) == 1:
            b.cz(ctrls[0], acc)
        else:
            b.ccz(ctrls[0], ctrls[1], acc)
        parity(mm)

    def visit(prod, mm: int) -> None:
        """High monomials mm | (1 << j) for j above mm's top bit, given prod = the product of mm's bits."""
        for j in range(mm.bit_length(), len(hi)):
            child = mm | (1 << j)
            if child >= V:
                continue
            if not any(child | (1 << q) < V for q in range(j + 1, len(hi))):
                phase([prod, hi[j]], child)  # a leaf: one CCZ
                continue
            t = pool.pop()
            b.init_and(prod, hi[j], t)
            phase([t], child)
            visit(t, child)
            b.del_and(prod, hi[j], t)
            pool.append(t)

    phase([], 0)
    for j in range(len(hi)):
        if 1 << j < V:
            phase([hi[j]], 1 << j)
            visit(hi[j], 1 << j)

    for i in reversed(range(k)):
        split(i, False)
    b.x(u[0])
    for row in Q:
        b.free(row)
    b.free(m)
    b.free(P)


def _paired_combine(d1: np.ndarray, d2: np.ndarray, d3: np.ndarray, bank_bits: int = BANK_BITS):
    """The bank contents and Toffolis of the paired combine.

    Each of the 27 bit slots s (word w, bit k < 9) needs q_s = l0 a_s + l1 b_s + L c_s over GF(2),
    with L = l0 l1 and (a, b, c) the slot's bits of (D1, D2, D3). A Toffoli whose controls are a
    selection bit XORed onto a bank wire, (L + x)(l0 + y) = L (1 + y) + l0 x + x y, yields two
    useful terms at once; x y depends on h only and is folded into the loaded word. Pairing slots
    i, j covers their six terms with three Toffolis (exhaustively checked over GF(2)):

        S  -> i and j:  (L + a_i)(l0 + ~c_j)
        Pi -> i:        (L + b_i)(l1 + ~(c_i ^ c_j))
        Pj -> j:        (l0 + b_j)(l1 + a_i ^ a_j), plus a CNOT of L

    An odd last slot m takes (L + a_m)(l0 + ~c_m) and l1 b_m. For 27 slots: 13 * 3 + 2 = 41
    Toffolis instead of 81, in the same 81 bank wires.

    Returns the bank values (H,) (bit i = wire i), the word-table corrections (H,) (bit s = slot s),
    and the Toffolis as (control, control, target slots, slots that also get a CNOT of L); a
    control is (selection, wire): selection 'L', 0, 1 or None, wire an index or None.
    """
    slots = d1.shape[1] * bank_bits
    mask = (1 << bank_bits) - 1
    pack = lambda d: sum((d[:, i].astype(np.int64) & mask) << (bank_bits * i) for i in range(d.shape[1]))  # noqa: E731
    a, bb, c = pack(d1), pack(d2), pack(d3)
    bit = lambda v, s: (v >> s) & 1  # noqa: E731
    wires, corr, toffolis = [], np.zeros(len(a), np.int64), []

    def add(sel_x, x, sel_y, y, targets, cnot_l=()):
        """A Toffoli on (sel_x + wire x)(sel_y + wire y); x or y None means the selection bit alone."""
        nonlocal corr
        cx = (sel_x, None) if x is None else (sel_x, len(wires))
        if x is not None:
            wires.append(x)
        cy = (sel_y, None) if y is None else (sel_y, len(wires))
        if y is not None:
            wires.append(y)
        if x is not None and y is not None:  # the h-only term x y, at l0 = l1 = 0
            for s in targets:
                corr ^= (x & y) << s
        toffolis.append((cx, cy, list(targets), list(cnot_l)))

    for i in range(0, slots - 1, 2):
        j = i + 1
        add('L', bit(a, i), 0, 1 ^ bit(c, j), (i, j))
        add('L', bit(bb, i), 1, 1 ^ bit(c, i) ^ bit(c, j), (i,))
        add(0, bit(bb, j), 1, bit(a, i) ^ bit(a, j), (j,), cnot_l=(j,))
    if slots % 2:
        m = slots - 1
        add('L', bit(a, m), 0, 1 ^ bit(c, m), (m,))
        add(1, None, None, bit(bb, m), (m,))
    assert len(wires) == 3 * slots
    bits = np.stack(wires, axis=1)  # (H, 81): wire values per high address, as Python ints (> 64 bits)
    bank = [sum(1 << int(i) for i in np.flatnonzero(row)) for row in bits]
    return bank, corr, toffolis


def _anf_load(b: km.CircuitBuilder, addr, outputs: list, table: list[int]) -> None:
    """outputs ^= table[h] for the high address h < H = len(table), through the algebraic normal form.

    table[h] = XOR over monomials m (subsets of address bits) with m subset of h of c_m. The addresses
    h >= H never occur, so their values are free: choosing them so that every c_m with m >= H vanishes
    (the Moebius transform of the table padded with zeros, cut at H) keeps f exact on h < H. Degree-0
    and degree-1 monomials are free (X and CNOT fan-outs); each higher one costs one Toffoli. They are
    built depth-first by adding address bits in increasing order: a monomial with needed supersets gets
    an AND ancilla (fanned out to its outputs, kept while its supersets are visited), one without is
    applied straight into its outputs (a Toffoli on a fanned-out target), so the deepest path holds one
    ancilla per bit of its parent, at most popcount - 2 at once.
    """
    H, hb = len(table), len(addr)
    c = list(table) + [0] * ((1 << hb) - H)
    for i in range(hb):  # Moebius transform over GF(2): c[m] = XOR of table[x] over x subset of m
        bit = 1 << i
        for m in range(1 << hb):
            if m & bit:
                c[m] ^= c[m ^ bit]
    needed = [False] * H  # c_m != 0, or some superset below H needs it as a prefix
    for m in range(H - 1, -1, -1):
        needed[m] = c[m] != 0 or any(m | (1 << j) < H and needed[m | (1 << j)]
                                     for j in range(m.bit_length(), hb))

    def targets(m: int) -> list:
        return [outputs[i] for i in _bits(c[m])]

    def visit(prod, m: int) -> None:
        """Supersets of m that add bits above m's highest, given prod = the product of m's bits."""
        for j in range(m.bit_length(), hb):
            child = m | (1 << j)
            if child >= H or not needed[child]:
                continue
            deeper = any(child | (1 << k) < H and needed[child | (1 << k)] for k in range(j + 1, hb))
            if not deeper:  # a leaf: one Toffoli straight into its outputs
                ts = targets(child)
                if ts:
                    first, rest = ts[0], ts[1:]
                    for t in rest:
                        b.cx(first, t)
                    b.ccx(prod, addr[j], first)
                    for t in reversed(rest):
                        b.cx(first, t)
                continue
            t = b.alloc_qubits(1)[0]
            b.init_and(prod, addr[j], t)
            ts = targets(child)
            if ts:
                b.cx(t, km.array(ts))
            visit(t, child)
            b.del_and(prod, addr[j], t)
            b.free(t)

    if c[0]:
        b.x(km.array(targets(0)))
    for j in range(hb):  # degree 1: the address bit itself is the product
        m = 1 << j
        if m < H and needed[m]:
            if c[m]:
                b.cx(addr[j], km.array(targets(m)))
            visit(addr[j], m)


def build_circuit(delta: np.ndarray, n_sel: int, load: str = 'anf', brot: int = BROT) -> km.Circuit:
    """The XOR oracle |a>|z> -> |a>|z ^ f(a)> for the (A, J) table `delta` of brot-bit words (signed gauge)."""
    delta = np.asarray(delta, dtype=np.int64)
    A, J = delta.shape
    nb = brot - 1  # bank bits per difference word: the top word bit equals the one below it
    assert A % 4 == 0 and A <= (1 << n_sel)
    assert not np.any(((delta >> (brot - 1)) ^ (delta >> (brot - 2))) & 1), 'words must be in the signed gauge'
    H, hb = A // 4, n_sel - 2
    assert hb >= 4, "needs at least 4 high address bits (A > 32)"

    def pack(row) -> int:  # J full brot-bit words
        return sum(int(v) << (brot * i) for i, v in enumerate(row))

    def spread(slot_bits: int) -> int:  # slot (w, k) -> word bit brot w + k; slot (w, nb - 1) also -> the top bit
        out = 0
        for s in _bits(slot_bits):
            w, k = divmod(s, nb)
            out |= (1 << (brot * w + k)) | ((1 << (brot * w + nb)) if k == nb - 1 else 0)
        return out

    q = delta.reshape(H, 4, J)
    d1, d2, d3 = q[:, 0] ^ q[:, 1], q[:, 0] ^ q[:, 2], q[:, 0] ^ q[:, 1] ^ q[:, 2] ^ q[:, 3]
    bank_table, corr, toffolis = _paired_combine(d1, d2, d3, nb)
    W = brot * J
    # Each word starts at f(4h) plus the combine's h-only terms, so the combine leaves exactly f(4h + l).
    table = [(pack(q[h, 0]) ^ spread(int(corr[h]))) | (bank_table[h] << W) for h in range(H)]

    b = km.CircuitBuilder()
    sel = b.create_quantum_register(n_sel, 'selection')
    words = [b.create_quantum_register(brot, f'word_{i}') for i in range(J)]
    z = [qq for w in words for qq in w]
    bank = list(b.alloc_qubits(3 * J * nb))
    outputs = z + bank
    addr = sel[2:]

    def xor_from(ctrl, value: int) -> None:
        ts = [outputs[i] for i in _bits(value)]
        if ts:
            b.cx(ctrl, km.array(ts))

    def xor_and(c1, c2, targets: list) -> None:
        """targets ^= c1 & c2 with one Toffoli: fan the first target out to the rest around it."""
        if not targets:
            return
        first, rest = targets[0], targets[1:]
        for t in rest:
            b.cx(first, t)
        b.ccx(c1, c2, first)
        for t in reversed(rest):
            b.cx(first, t)

    def subtree(ctrl, bit: int, offset: int) -> None:
        """Load rows h in [offset, offset + 2**(bit+1)) (and < H), given ctrl = match of the higher bits."""
        if bit == 0:
            v0 = table[offset]
            v1 = table[offset + 1] if offset + 1 < H else v0  # past the end: never addressed
            xor_from(ctrl, v0)
            xor_and(ctrl, addr[0], [outputs[i] for i in _bits(v0 ^ v1)])
            return
        half = 1 << bit
        if offset + half >= H:  # right half past the last row: its addresses never occur
            subtree(ctrl, bit - 1, offset)
            return
        t = b.alloc_qubits(1)[0]
        b.x(addr[bit])
        b.init_and(ctrl, addr[bit], t)  # t = ctrl & ~addr[bit]
        b.x(addr[bit])
        subtree(t, bit - 1, offset)
        b.cx(ctrl, t)                     # t = ctrl & addr[bit]
        subtree(t, bit - 1, offset + half)
        b.del_and(ctrl, addr[bit], t)
        b.free(t)

    # Top three address bits -> one control c per block of 2**(hb-3) rows. Blocks are visited in
    # a Gray-code order where possible: between blocks that differ in one bit, c_q ^ c_p is the AND
    # of the other two bits, so moving costs one Toffoli instead of an erase and a recompute (3).
    top = hb - 3
    block = 1 << top
    tb = addr[top:top + 3]

    def and_into(p: int, bits: list[int], target, init: bool) -> None:
        """target ^= AND over i in bits of [tb[i] == bit i of p] (two bits; init_and or del_and)."""
        flips = [tb[i] for i in bits if not (p >> i) & 1]
        if flips:
            b.x(km.array(flips))
        (b.init_and if init else b.del_and)(tb[bits[0]], tb[bits[1]], target)
        if flips:
            b.x(km.array(flips))

    def set_block(p: int, c, init: bool) -> None:
        """Compute (init) or erase c = [top bits == p] through a temporary AND of the upper two."""
        t = b.alloc_qubits(1)[0]
        and_into(p, [2, 1], t, True)
        flip = not p & 1
        if flip:
            b.x(tb[0])
        (b.init_and if init else b.del_and)(t, tb[0], c)
        if flip:
            b.x(tb[0])
        and_into(p, [2, 1], t, False)  # erase and free the middle ancilla before descending
        b.free(t)

    order = _block_path((H + block - 1) // block)
    c = b.alloc_qubits(1)[0]
    if load == 'anf':
        order = []  # the ANF load below replaces the unary iteration
    else:
        set_block(order[0], c, True)
    for i, p in enumerate(order):
        if i:
            q = order[i - 1]
            if bin(p ^ q).count('1') == 1:
                t = b.alloc_qubits(1)[0]
                rest = [j for j in (2, 1, 0) if not (p ^ q) >> j & 1]
                and_into(p, rest, t, True)
                b.cx(t, c)
                and_into(p, rest, t, False)
                b.free(t)
            else:
                set_block(q, c, False)
                set_block(p, c, True)
        subtree(c, top - 1, p * block)
    if load == 'anf':
        b.free(c)
        _anf_load(b, addr, outputs, table)
    else:
        set_block(order[-1], c, False)
        b.free(c)

    both = b.alloc_qubits(1)[0]
    b.init_and(sel[0], sel[1], both)
    selection = {'L': both, 0: sel[0], 1: sel[1]}

    def slot_targets(slots) -> list:
        out = []
        for s in slots:
            w, k = divmod(s, nb)
            out += [z[brot * w + k]] + ([z[brot * w + nb]] if k == nb - 1 else [])
        return out

    for cx, cy, slots, cnot_l in toffolis:
        shifted = [(selection[s], bank[i]) for s, i in (cx, cy) if s is not None and i is not None]
        for s, wire in shifted:  # the control becomes selection + wire
            b.cx(s, wire)
        controls = [bank[i] if i is not None else selection[s] for s, i in (cx, cy)]
        xor_and(controls[0], controls[1], slot_targets(slots))
        for s, wire in shifted:
            b.cx(s, wire)
        for t in slot_targets(cnot_l):
            b.cx(both, t)
    b.del_and(sel[0], sel[1], both)
    b.free(both)

    _erase(b, bank, addr, bank_table)
    b.free(km.array(bank))
    circuit = b.finish_circuit()
    if circuit.num_qubits > 128 and load == 'anf':  # a deep ANF path needs one ancilla too many
        return build_circuit(delta, n_sel, load='unary', brot=brot)
    assert circuit.num_qubits <= 128, circuit.num_qubits
    return circuit


def batch_deltas(words: np.ndarray, j: int = J) -> np.ndarray:
    """f_k = batch k-1 XOR batch k of the (A, 75) word table, zero padded to a multiple of j."""
    words = np.asarray(words, dtype=np.int64)
    A, n = words.shape
    padded = np.zeros((A, -(-n // j) * j), dtype=np.int64)
    padded[:, :n] = words
    batches = padded.reshape(A, -1, j).transpose(1, 0, 2)
    out = batches.copy()
    out[1:] ^= batches[:-1]
    return out


def build_family(words: np.ndarray, brot: int = BROT, j: int = J) -> list[km.Circuit]:
    A = len(words)
    n_sel = max(1, (A - 1).bit_length())
    return [build_circuit(d, n_sel, brot=brot) for d in batch_deltas(words, j)]


def check(circuit: km.Circuit, delta: np.ndarray, repeats: int = 4, seed: int = 0, brot: int = BROT,
          signed: bool = False) -> float:
    """Simulate every address with random word contents; return Toffolis per shot.

    `signed` draws the contents in the signed gauge only (top bit = the one below), the inputs the
    official verifier feeds; circuits that borrow the top bit (tensorqrom `scratch`) need it."""
    A, J = delta.shape
    n_sel = max(1, (A - 1).bit_length())
    a = np.tile(np.arange(A), repeats)
    rng = np.random.default_rng(seed)
    if signed:
        q = 1 << (brot - 2)
        z0 = rng.integers(-q, q, size=(len(a), J)) & ((1 << brot) - 1)
    else:
        z0 = rng.integers(0, 1 << brot, size=(len(a), J))
    sim = km.Simulator(batch_size=len(a), seed=seed, counting_operations=True)
    sim.use_same_registers_as(circuit)
    sim.write_across_shots('selection', ((a[None, :] >> np.arange(n_sel)[:, None]) & 1).astype(bool))
    for j in range(J):
        sim.write_across_shots(f'word_{j}', ((z0[:, j][None, :] >> np.arange(brot)[:, None]) & 1).astype(bool))
    sim.do(circuit)
    got = np.stack([(np.int64(1) << np.arange(brot)) @ sim.read_across_shots(f'word_{j}').astype(np.int64)
                    for j in range(J)], axis=1)
    assert np.array_equal(got, z0 ^ delta[a]), 'wrong XOR'
    assert not np.any(sim.read_phase_flipped_across_shots()), 'uncorrected phase'
    sel = (np.int64(1) << np.arange(n_sel)) @ sim.read_across_shots('selection').astype(np.int64)
    assert np.array_equal(sel, a), 'selection changed'
    return sim.operation_statistics().num_toffoli_like / len(a)
