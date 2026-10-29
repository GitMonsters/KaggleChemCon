import sys
sys.path.insert(0, '/mnt/c/Users/omen/Desktop/quantumAI/paradigm')
from chem_costmodel import heuristic
REF = 3355599588.3155365
ANG = 2 * 19 * 274            # our streamed J4 build at 1366 rows
for C in (60, 50, 46, 44, 42, 40, 38, 36):
    h = heuristic(10, 129, C, 8)
    T = h['T'] + ANG
    Q = h['persistent'] + 32 + max(h['temp'], 96)
    print(f'C {C}: T {T}  Q {Q}  innerT {h["T_inner"]}  ratio@47.82 {REF / (47.82 * T * Q):.4f}  @47.4 {REF / (47.4 * T * Q):.4f}')
