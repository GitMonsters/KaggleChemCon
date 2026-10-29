"""Time and cost of mrmeows' exact J4 lookup planner on batches of a 1288x75 word table."""
import json, sys, time
from pathlib import Path
import numpy as np
REP = Path('/home/omen/vqcwork/m1575/ds/source/chemistry_6p69/reproduction')
sys.path[:0] = [str(REP), '/home/omen/vqcwork/comp/kickmix']
import rank_parity as core
from permuted_kernel import plan

words = np.load(sys.argv[1]) if len(sys.argv) > 1 else np.load(REP / 'actual_words.npy')
ks = [int(a) for a in sys.argv[2].split(',')] if len(sys.argv) > 2 else [0]
for k in ks:
    cert = json.loads((REP / f'certificates/plan_{k}.json').read_text())['planning']
    delta = core.delta_batch(words, 4, k)
    t0 = time.time()
    res, rep = plan(delta, cert['extra'], seed=cert['seed'], permutation=cert.get('low_permutation'))
    print(k, 'paired', rep.get('paired_products'), 'T_HMR', rep.get('T_HMR'), 'target_rank', rep.get('target_rank'),
          'relations', rep.get('relations'), f'{time.time()-t0:.1f}s', flush=True)
