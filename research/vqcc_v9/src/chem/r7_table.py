"""Cost of R=7 (and nearby) B=129 shapes with 979 rows (10 address bits), full-cache J4/J5/J6 angle circuits."""
import sys
sys.path.insert(0, '/mnt/c/Users/omen/Desktop/quantumAI/paradigm')
from chem_costmodel import heuristic
REF = 3355599588.3155365


def cache(L):
    A = -(-L // 16)
    return sum(1 for m in range(1, A) if bin(m).count('1') >= 2) + 11


def angles(L, J):
    rel = {4: 60, 5: 75, 6: 90}[J]
    prod = (15 * 7 * J - rel) // 2
    per = cache(L) + prod
    circ = -(-75 // J)
    q = (L - 1).bit_length() + 8 * J + cache(L)
    return 2 * circ * per, per, circ, q


for R, B in ((7, 129), (6, 129), (10, 129)):
    L = 76 + R * B
    for C in (40, 46, 50, 56, 60, 66):
        h = heuristic(R, B, C, 8)
        for J in (4, 5, 6):
            aT, per, circ, q = angles(L, J)
            if q > 128:
                continue
            T = h['T'] + aT
            Q = h['persistent'] + 8 * J + max(h['temp'], cache(L))
            print(f'R{R} B{B} C{C} rows {L} J{J}: circuits {circ} x {per} (q {q}) angleT {aT} innerT {h["T_inner"]} '
                  f'T {T} Q {Q} ratio@47.8 {REF / (47.8 * T * Q):.3f}')
