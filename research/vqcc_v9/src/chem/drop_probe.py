"""Products saved by the streamed planner when k outputs of a batch have no mixed (high x low) terms:
plan on the reduced tensor ac[:, :, keep] and count paired products (stage 1 + stage 2)."""
import sys
from pathlib import Path

import numpy as np

SRC = Path('/home/omen/vqcwork/m2100/sources/dfthcsos_cost_first_shape_screen/source')
sys.path[:0] = [str(SRC), '/home/omen/vqcwork/comp/kickmix']
import rank_parity as core
from balanced_plan import plan_balanced
from streamed_plan import plan_tensor

words = np.asarray(np.load(sys.argv[1])['angle_words'], dtype=np.int64)
k = int(sys.argv[2]) if len(sys.argv) > 2 else 3
delta = core.delta_batch(words, 4, k)
ac = core.coefficients(delta, 8, 4)


def products(acr, nrel1, nrel2, seeds=12):
    best = None
    for s in range(seeds):
        left, _ = plan_balanced(acr[1:, 1:12, :], group_size=3, nrelations=nrel1, seed=s, trials=16, plane_mode='pairs')
        if left is None:
            continue
        for s2 in range(4):
            right, _ = plan_tensor(acr[1:, 12:16, :], group_size=4, nrelations=nrel2, seed=s2, trials=24, plane_mode='original')
            if right is None:
                continue
            n = sum((len(t['factors']) + 1) // 2 for t in left) + sum((len(t['factors']) + 1) // 2 for t in right)
            best = n if best is None else min(best, n)
            break
    return best


full = products(ac, 38, 12)
print('full 28 outputs: products', full, flush=True)
for drop in (1, 2, 4, 8):
    keep = [o for o in range(28) if not (o % 7 == 0 and o // 7 < drop) and not (drop > 4 and o % 7 == 1 and o // 7 < drop - 4)]
    m = len(keep)
    res = {}
    for nrel1, nrel2 in ((38, 12), (int(38 * m / 28), int(12 * m / 28)), (30, 10), (44, 14)):
        res[(nrel1, nrel2)] = products(ac[:, :, keep], nrel1, nrel2)
    print(f'drop {drop} (m={m}): products by (nrel1,nrel2): {res}', flush=True)
