import sys
sys.path.insert(0, '/mnt/c/Users/omen/Desktop/quantumAI/paradigm')
from chem_costmodel import heuristic
REF = 3355599588.3155365
lam = float(sys.argv[1]) if len(sys.argv) > 1 else 48.1
rows = []
for R in (4, 5, 6, 8):
    for B in (303, 287, 271, 255, 239, 223, 207, 191, 175, 159, 143, 127):
        for C in (128, 120, 112, 104, 96, 88, 80, 72, 64):
            if R * C < 300:
                continue
            h = heuristic(R, B, C, 8)
            L = 76 + R * B
            Ta = 7.79 * L
            J = 4
            Q = h['persistent'] + J * 8 + max(h['temp'], 127 - 11 - J * 8)
            T = h['T'] + Ta
            rows.append((REF / (lam * T * Q), R, B, C, L, round(T), Q, h['T_inner']))
rows.sort(reverse=True)
print('ratio  R   B   C     L      T    Q  T_inner')
base = [r for r in rows if r[1:4] == (4, 303, 128)][0]
print('base', base)
for r in rows:
    if r[1] == 4 and r[3] in (128, 112, 96) or r[1:4] in ((8, 127, 64), (8, 159, 64), (6, 191, 80)):
        print(f'{r[0]:5.2f} {r[1]:2d} {r[2]:4d} {r[3]:4d} {r[4]:5d} {r[5]:6d} {r[6]:4d} {r[7]:6d}')
