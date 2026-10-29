"""How many relations can a stage-respecting layout carry on one J5 batch (1366 rows)?"""
import random
import sys
import time
import numpy as np
sys.path.insert(0, '/mnt/c/Users/omen/Desktop/quantumAI/paradigm/chem')
import j5layout as J
core = J.core
words = np.asarray(np.load(sys.argv[1])['angle_words'], dtype=np.int64)
k = int(sys.argv[2]) if len(sys.argv) > 2 else 3
delta = core.delta_batch(words, 5, k)
ac = core.coefficients(delta, 8, 4)
T = ac[1:, 1:, :]                     # (85, 15, 35)
print('tensor', T.shape, flush=True)
# layout A: per stage, cyclic 3-blocks over the stage's 8 live indices (addresses shared)
def cyc(live, size=3):
    return [tuple(live[(i + d) % len(live)] for d in range(size)) for i in range(len(live))]
KINDS = tuple(sys.argv[3].split(',')) if len(sys.argv) > 3 else ('pair', 'triple', 'quad', 'tri3')
base = [b for live in J.STAGES for b in cyc(live)]
layouts = {
    'cyc3': base,
    'cyc3+addr': base + [tuple(J.ADDR)],
}
for name, blocks in layouts.items():
    for total in (49, 55, 61, 67, 73, 79):
        per, extra = divmod(total, len(blocks))
        wants = [per + (1 if i < extra else 0) for i in range(len(blocks))]
        t0 = time.time()
        best = None
        rep = None
        for seed in range(3):
            terms, rep = J.plan_layout(T, blocks, wants, seed=seed, plane_kinds=KINDS)
            if terms is not None and (best is None or rep['paired'] < best[1]['paired']):
                best = (terms, rep)
        last = rep
        print(name, 'total', total, 'blocks', len(blocks), '->', best[1] if best else last, f'{time.time() - t0:.0f}s', flush=True)
