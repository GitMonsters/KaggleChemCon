"""Product counts for the 3-stage J5 layout on one batch of a 1366-row table."""
import sys
import time
import numpy as np
sys.path.insert(0, '/mnt/c/Users/omen/Desktop/quantumAI/paradigm/chem')
import j5plan as P
core = P.core
words = np.asarray(np.load(sys.argv[1])['angle_words'], dtype=np.int64)
k = int(sys.argv[2]) if len(sys.argv) > 2 else 3
delta = core.delta_batch(words, 5, k)
ac = core.coefficients(delta, 8, 4)        # (86, 16, 35)
print('ac', ac.shape)
STAGES = [[1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12], [13, 14, 15]]
total = 0
for S in STAGES:
    t = ac[1:, S, :]
    best = None
    t0 = time.time()
    for rb in (6, 5, 4, 3, 2, 1, 0):
        for ex in (0, len(S)):
            for seed in range(4):
                terms, rep = P.plan_stage(t, rb, ex, seed)
                if terms is not None and (best is None or rep['paired'] < best[1]['paired']):
                    best = (terms, rep)
            if best is not None and best[1]['rel_base'] == rb:
                break
        if best is not None:
            break
    print(S, None if best is None else best[1], f'{time.time() - t0:.0f}s', flush=True)
    total += best[1]['paired'] if best else 10 ** 6
print('products', total, '-> per circuit', 78 + 11 + total)
