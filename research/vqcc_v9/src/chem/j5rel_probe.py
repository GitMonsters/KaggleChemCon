import sys, time
import numpy as np
sys.path.insert(0, '/mnt/c/Users/omen/Desktop/quantumAI/paradigm/chem')
import j5relsearch as RS
core = RS.core
words = np.asarray(np.load(sys.argv[1])['angle_words'], dtype=np.int64)
k = int(sys.argv[2]); seed = int(sys.argv[3]) if len(sys.argv) > 3 else 0
budget = float(sys.argv[4]) if len(sys.argv) > 4 else 600
ac = core.coefficients(core.delta_batch(words, 5, k), 8, 4)
T = ac[1:, 1:, :]
log = lambda s: print(time.strftime('%H:%M:%S'), s, flush=True)
mode = sys.argv[5] if len(sys.argv) > 5 else 'all'
sizes = tuple(int(x) for x in sys.argv[6].split(',')) if len(sys.argv) > 6 else (3, 4)
stall = int(sys.argv[7]) if len(sys.argv) > 7 else 400
st = RS.search(T, seed=seed, budget=budget, log=log, block_mode=mode, block_sizes=sizes, stall=stall)
log(f'final relations {len(st.rb)}')
imp = float(sys.argv[8]) if len(sys.argv) > 8 else 0
if imp > 0:
    st = RS.improve(st, T, seed=seed, budget=imp, log=log)
    log(f'after improve: {len(st.rb)}')
# make the target even (odd relation count) by dropping to the best odd prefix if needed, then pair
for drop in range(0, 8):
    n = len(st.rb)
    terms, rep = RS.pair_plan(st, seed=seed)
    log(f'pairing with {n} relations: {rep}')
    if terms is not None:
        log(f'PRODUCTS {rep["paired"]} -> per circuit {78 + 11 + rep["paired"]}')
        break
    # rebuild state without the last relation
    rels = st.rels[:-1]
    st2 = RS.State(T)
    for r in rels:
        assert st2.try_add(r)
    st = st2
