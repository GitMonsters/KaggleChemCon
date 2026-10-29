"""Official (R, B, C) cost at 8 bits with mrmeows-style J=4 angle circuits (cache(L) + 180 products each)."""
import math
import sys
sys.path.insert(0, '/mnt/c/Users/omen/Desktop/quantumAI/paradigm')
from chem_costmodel import heuristic

REF = 3355599588.3155365


def angle(L, J=4, products=180, lowbits=4):
    A = -(-L // (1 << lowbits))
    cache = sum(1 for m in range(1, A) if bin(m).count('1') >= 2) + sum(1 for m in range(1, 1 << lowbits) if bin(m).count('1') >= 2)
    per = cache + products
    circuits = -(-75 // J)
    nsel = (L - 1).bit_length()
    return dict(per=per, circuits=circuits, T=2 * circuits * per, qubits=nsel + J * 8 + cache, temp=cache)


def official(R, B, C, lam=47.4, J=4, products=180):
    h = heuristic(R, B, C, 8)
    a = angle(76 + R * B, J, products)
    T = h['T'] + a['T']
    Q = h['persistent'] + J * 8 + max(h['temp'], a['temp'])
    return T, Q, REF / (lam * T * Q), a, h


rows = [(4, 303, 128), (4, 255, 128), (4, 239, 128), (4, 223, 128), (4, 207, 128), (4, 191, 128), (4, 255, 112),
        (4, 255, 96), (4, 223, 112), (4, 223, 96), (4, 191, 112), (4, 191, 96)]
print('shape          L     T      Q   angleT  per  ang_q  inner  ratio@47.4  vs 6.7795')
for R, B, C in rows:
    T, Q, ratio, a, h = official(R, B, C)
    print(f'{str((R, B, C)):14s} {76 + R * B:5d} {T:6d} {Q:4d} {a["T"]:6d} {a["per"]:4d} {a["qubits"]:4d} {h["T_inner"]:6d}   {ratio:7.4f}   {100 * (ratio / 6.7795 - 1):+6.1f}%')
