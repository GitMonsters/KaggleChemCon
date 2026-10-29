"""Parallel steepest-descent word moves from a saved (K, W) state.

Usage: python run_par.py <tree npz> <state npz> <out npz> <iters> [frac]
"""
import sys
import time

import numpy as np
import torch

sys.path.insert(0, '/mnt/c/Users/omen/Desktop/quantumAI/paradigm/chem')
import q8lib as L
from q8opt import Problem

torch.backends.cuda.matmul.allow_tf32 = False
tree_npz, state, out, iters = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
frac = float(sys.argv[5]) if len(sys.argv) > 5 else 0.25
z = L.load_submission(tree_npz)
pr = Problem(z['tree_edges'])
s = np.load(state)
K = torch.tensor(s['K'], device='cuda')
W = torch.tensor(s['W'], device='cuda', dtype=torch.float64)


def log(m):
    print(time.strftime('%H:%M:%S'), m, flush=True)


log(f'start {1e3 * pr.err2(pr.vectors(K), W):.4f} mHa')
chunk = 50
for start in range(0, iters, chunk):
    e, K, W = pr.parallel_descent(K, W, min(chunk, iters - start), frac=frac, log=log, tag=f'[{start}]')
    np.savez(out, K=K.cpu().numpy(), W=W.cpu().numpy())
    log(f'saved {1e3 * e:.4f} mHa')
