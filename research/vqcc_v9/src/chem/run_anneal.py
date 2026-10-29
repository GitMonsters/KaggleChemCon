"""Annealed quantization-aware fit from a continuous state, then rounding + lambda-aware local search.

Usage: python run_anneal.py <tree npz> <continuous npz (theta, W)> <out npz> [steps] [mu1] [sweeps] [kappa]
"""
import sys
import time

import numpy as np
import torch

sys.path.insert(0, '/mnt/c/Users/omen/Desktop/quantumAI/paradigm/chem')
import q8lib as L
from q8opt import Problem

torch.backends.cuda.matmul.allow_tf32 = False
tree_npz, cont, out = sys.argv[1], sys.argv[2], sys.argv[3]
steps = int(sys.argv[4]) if len(sys.argv) > 4 else 30000
mu1 = float(sys.argv[5]) if len(sys.argv) > 5 else 1e-2
sweeps = int(sys.argv[6]) if len(sys.argv) > 6 else 20
kappa = float(sys.argv[7]) if len(sys.argv) > 7 else 2e-5
z = L.load_submission(tree_npz)
pr = Problem(z['tree_edges'])
c = np.load(cont)
theta = torch.tensor(c['theta'], device='cuda')
W = torch.tensor(c['W'], device='cuda')


def log(m):
    print(time.strftime('%H:%M:%S'), m, flush=True)


th, w = pr.anneal(theta, W, steps, mu1=mu1, kappa=kappa, log=log)
K = torch.clamp(torch.round(th.double() * 128), -64, 63).long()
U = pr.vectors(K)
W = w.double()
log(f'rounded: error_2 {1e3 * pr.err2(U, W):.4f} mHa D {L.dpen(W).item():.1f}')
W = pr.fit_w(U, W, steps=600, kappa=kappa)
log(f'after W refit: {1e3 * pr.err2(U, W):.4f} mHa D {L.dpen(W).item():.1f}')
np.savez(out.replace('.npz', '_r.npz'), K=K.cpu().numpy(), W=W.cpu().numpy())
rng = np.random.default_rng(3)
best = (pr.err2(U, W), K.clone(), W.clone())
for it in range(sweeps):
    R, B, C = W.shape
    order = [(r, b) for r in range(R) for b in rng.permutation(B)]
    K, acc, e0, e1 = pr.sweep(K, W, order=order, max_step=1, inner=20)
    U = pr.vectors(K)
    W = pr.fit_w(U, W, steps=300, kappa=kappa)
    e = pr.err2(U, W)
    if e < best[0]:
        best = (e, K.clone(), W.clone())
    log(f'sweep {it}: moves {acc}, fixed-W {1e3 * e0:.3f} -> {1e3 * e1:.3f}, W refit {1e3 * e:.4f} mHa D {L.dpen(W).item():.1f} (best {1e3 * best[0]:.4f})')
    np.savez(out, K=best[1].cpu().numpy(), W=best[2].cpu().numpy())
log('done')
