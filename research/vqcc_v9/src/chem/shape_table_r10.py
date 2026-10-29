"""R=10 points vs (4,B,C): official heuristic + angle circuits (J=4 mrmeows 180 products when <=128 qubits,
else J=3 with alex v37's measured 257/circuit at 1366 rows = cache 89 + 168 products)."""
import sys
sys.path.insert(0, '/mnt/c/Users/omen/Desktop/quantumAI/paradigm/chem')
sys.path.insert(0, '/mnt/c/Users/omen/Desktop/quantumAI/paradigm')
from chem_costmodel import heuristic
REF = 3355599588.3155365


def cache(L):
    A = -(-L // 16)
    return sum(1 for m in range(1, A) if bin(m).count('1') >= 2) + 11


def best_angle(L):
    out = []
    nsel = (L - 1).bit_length()
    for J, prod in ((4, 180), (3, 168)):
        q = nsel + 8 * J + cache(L)
        if q <= 128:
            circ = -(-75 // J)
            out.append((2 * circ * (cache(L) + prod), J, q, circ))
    return min(out) if out else None


for R, B, C in [(10, 129, 60), (10, 127, 60), (10, 111, 60), (10, 129, 52), (10, 111, 52), (10, 95, 60),
                (4, 255, 128), (4, 223, 112), (4, 255, 96), (4, 191, 128)]:
    L = 76 + R * B
    h = heuristic(R, B, C, 8)
    a = best_angle(L)
    if a is None:
        print((R, B, C), L, 'no J fits 128 qubits')
        continue
    T = h['T'] + a[0]
    Q = h['persistent'] + 8 * a[1] + max(h['temp'], cache(L))
    print(f'{str((R, B, C)):14s} L {L:5d} J {a[1]} circ {a[3]:2d} angleT {a[0]:6d} innerT {h["T_inner"]:5d} T {T:6d} Q {Q:4d} '
          f'ratio@47.4 {REF / (47.4 * T * Q):6.3f}')
