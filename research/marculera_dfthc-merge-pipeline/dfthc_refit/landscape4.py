"""Chemistry loss over (R, B, C) with the four-address ANGLE circuits (fourqrom.py), at fixed lambda_eff.

Measured on fourqrom.py: each of the 25 circuits (three 10-bit words) costs about rows/4 + 58
Toffolis per pass (57 at 1,276 rows, 61 at 1,516): the ANF load over the high addresses (about
rows/4 - 10), 42 for the paired combine and ~25 for the ANF bank erase, e.g. 18,800 over both passes
at 1,276 rows. The word registers add J * brot = 3 * 10 persistent qubits.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import DATA  # noqa: E402,F401  (puts the verifiers on sys.path)
from dfthcsos.verify import heuristic_costs  # noqa: E402

N, CIRCUITS, OVERHEAD, WORD_QUBITS, BROT = 76, 25, 58, 30, 10
REFERENCE = 3355599588.3155365


def loss(R: int, B: int, C: int, lam: float) -> tuple[float, float, int]:
    h = heuristic_costs(N, R, B, C, BROT)
    T = h['toffoli'] + 2 * CIRCUITS * ((N + R * B) / 4 + OVERHEAD)
    Q = h['qubits'] + WORD_QUBITS
    return lam * T * Q, T, Q


if __name__ == '__main__':
    print('shape          rows      T     Q   x ref @ lambda 49 / 52 / 56')
    for shape in [(32, 60, 30), (1, 640, 384), (65, 60, 30),
                  (16, 100, 60), (16, 80, 60), (8, 160, 120), (8, 120, 90), (4, 240, 192), (4, 180, 128),
                  (2, 320, 256), (16, 60, 60), (24, 60, 40)]:
        R, B, C = shape
        _, T, Q = loss(R, B, C, 1)
        xs = ' / '.join(f'{REFERENCE / loss(R, B, C, lam)[0]:.2f}' for lam in (49, 52, 56))
        print(f'{str(shape):14s} {N + R * B:5d} {T:7.0f} {Q:5d}   {xs}')
