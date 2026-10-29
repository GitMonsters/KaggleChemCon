"""Does the streamed J4 planner accept more relations on this table?  (products = (L*m - relations)/2 per stage)"""
import sys
from pathlib import Path
import numpy as np
SRC = Path('/home/omen/vqcwork/m2100/sources/dfthcsos_cost_first_shape_screen/source')
sys.path[:0] = [str(SRC), '/home/omen/vqcwork/comp/kickmix']
import rank_parity as core
from balanced_plan import plan_balanced
from streamed_plan import plan_tensor
words = np.asarray(np.load(sys.argv[1])['angle_words'], dtype=np.int64)
for k in [int(x) for x in sys.argv[2].split(',')]:
    delta = core.delta_batch(words, 4, k)
    ac = core.coefficients(delta, 8, 4)
    res1, res2 = {}, {}
    for nr in (38, 40, 42, 44, 46, 48):
        ok = None
        for s in range(12):
            left, _ = plan_balanced(ac[1:, 1:12, :], group_size=3, nrelations=nr, seed=s, trials=16, plane_mode='pairs')
            if left is not None:
                ok = sum((len(t['factors']) + 1) // 2 for t in left)
                break
        res1[nr] = ok
    for nr in (12, 14, 16, 18, 20):
        ok = None
        for s in range(12):
            right, _ = plan_tensor(ac[1:, 12:16, :], group_size=4, nrelations=nr, seed=s, trials=24, plane_mode='original')
            if right is not None:
                ok = sum((len(t['factors']) + 1) // 2 for t in right)
                break
        res2[nr] = ok
    print('batch', k, 'stage1 products by nrel', res1, '| stage2', res2, flush=True)
