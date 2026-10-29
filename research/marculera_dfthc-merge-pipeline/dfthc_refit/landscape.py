"""T_total * Q_total over factorization shapes, relative to the baseline, at fixed lambda_eff.

Angle circuits are modelled as the reference unary-iteration QROM: `rows - 2` Toffolis per
circuit and pass, `ceil(log2 rows)` selection qubits plus one fewer temporaries, and as many
`brot`-bit words per circuit as fit under the 128-qubit cap.
"""
import math
import sys

from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import DATA  # noqa: E402,F401  (puts the verifiers on sys.path)
from dfthcsos.verify import heuristic_costs  # noqa: E402

N, CAP = 76, 128


def angles(rows: int, brot: int) -> tuple[int, int, int]:
    sel = math.ceil(math.log2(rows))
    J = min((CAP - 2 * sel + 1) // brot, N - 1)
    K = math.ceil((N - 1) / J)
    return 2 * K * (rows - 2), J * brot, sel - 1  # toffoli, persistent, temporary


def cost(R: int, B: int, C: int, brot: int) -> tuple[float, int, int]:
    h = heuristic_costs(N, R, B, C, brot)
    t_a, p_a, tmp_a = angles(N + R * B, brot)
    T = t_a + h['toffoli']
    Q = h['persistent_qubits'] + p_a + max(h['temporary_qubits'], tmp_a)
    return T * Q, T, Q


if __name__ == '__main__':
    base = cost(65, 60, 60, 12)[0]
    print('baseline T*Q', base, cost(65, 60, 60, 12)[1:])
    rows = []
    for R in range(20, 71, 5):
        for B in range(20, 71, 5):
            for C in (20, 30, 40, 50, 60):
                for brot in (10, 11, 12, 13, 14):
                    try:
                        tq, T, Q = cost(R, B, C, brot)
                    except ValueError:
                        continue
                    rows.append((base / tq, R, B, C, brot, T, Q))
    rows.sort(reverse=True)
    print('gain  R  B  C brot      T    Q')
    for r in rows[:5]:
        print('%.2f %3d %3d %3d %3d %7d %4d' % r)
    print('-- at brot 12, varying one axis from the baseline --')
    for R, B, C in [(65, 60, 60), (50, 60, 60), (65, 45, 60), (65, 60, 40), (50, 45, 40), (40, 40, 40), (30, 30, 30)]:
        tq, T, Q = cost(R, B, C, 12)
        print(f'{R:3d} {B:3d} {C:3d}  gain {base / tq:.2f}  T {T}  Q {Q}')
