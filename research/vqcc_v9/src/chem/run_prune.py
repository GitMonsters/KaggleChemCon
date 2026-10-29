"""Shrink an 8-bit factorization in the quantized domain: remove vectors (or columns) in small steps,
recover with lambda-aware W refits and exact word moves, log the frontier.

Usage: python run_prune.py <tree npz> <state npz> <outprefix> <mode B|C> <target> <step> [sweeps_per_step] [kappa]
"""
import sys
import time

import numpy as np
import torch

sys.path.insert(0, '/mnt/c/Users/omen/Desktop/quantumAI/paradigm/chem')
import q8lib as L
from q8opt import Problem

torch.backends.cuda.matmul.allow_tf32 = False
tree_npz, state, prefix, mode = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
target, step = int(sys.argv[5]), int(sys.argv[6])
sweeps = int(sys.argv[7]) if len(sys.argv) > 7 else 3
kappa = float(sys.argv[8]) if len(sys.argv) > 8 else 2e-5
z = L.load_submission(tree_npz)
pr = Problem(z['tree_edges'])
s = np.load(state)
K = torch.tensor(s['K'], device='cuda')
W = torch.tensor(s['W'], device='cuda', dtype=torch.float64)
rng = np.random.default_rng(7)


def log(m):
    print(time.strftime('%H:%M:%S'), m, flush=True)


def recover(K, W, n):
    for it in range(n):
        R, B, C = W.shape
        order = [(r, b) for r in range(R) for b in rng.permutation(B)]
        K, acc, e0, e1 = pr.sweep(K, W, order=order, max_step=1, inner=20)
        U = pr.vectors(K)
        W = pr.fit_w(U, W, steps=300, kappa=kappa)
    return K, W


U = pr.vectors(K)
log(f'start shape {tuple(W.shape)} error_2 {1e3 * pr.err2(U, W):.4f} mHa D {L.dpen(W).item():.1f}')
while True:
    R, B, C = W.shape
    cur = B if mode == 'B' else C
    if cur <= target:
        break
    m = min(step, cur - target)
    if mode == 'B':
        K, W = pr.prune_vectors(K, W, m)
    else:
        W, _ = pr.prune_columns(W, m)
    U = pr.vectors(K)
    e_cut = pr.err2(U, W)
    W = pr.fit_w(U, W, steps=600, kappa=kappa)
    e_fit = pr.err2(U, W)
    K, W = recover(K, W, sweeps)
    U = pr.vectors(K)
    e = pr.err2(U, W)
    log(f'shape {tuple(W.shape)}: cut {1e3 * e_cut:.3f} -> W refit {1e3 * e_fit:.3f} -> recovered {1e3 * e:.4f} mHa D {L.dpen(W).item():.1f}')
    np.savez(f'{prefix}_{W.shape[1]}_{W.shape[2]}.npz', K=K.cpu().numpy(), W=W.cpu().numpy())
log('done')
