"""Continue discrete sweeps + exact K refits from a saved (K, W) state.

Usage: python run_sweeps.py <tree npz> <state npz> <out npz> <sweeps> [max_step] [temp0]
"""
import sys
import time

import numpy as np
import torch

sys.path.insert(0, '/mnt/c/Users/omen/Desktop/quantumAI/paradigm/chem')
import q8lib as L
from q8opt import Problem

torch.backends.cuda.matmul.allow_tf32 = False
tree_npz, state, out, sweeps = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
max_step = int(sys.argv[5]) if len(sys.argv) > 5 else 1
temp0 = float(sys.argv[6]) if len(sys.argv) > 6 else 0.0
inner = int(sys.argv[7]) if len(sys.argv) > 7 else 1
kappa = float(sys.argv[8]) if len(sys.argv) > 8 else -1.0   # <0: exact K refit; >=0: lambda-aware Adam W refit
z = L.load_submission(tree_npz)
pr = Problem(z['tree_edges'])
s = np.load(state)
K = torch.tensor(s['K'], device='cuda')
W = torch.tensor(s['W'], device='cuda', dtype=torch.float64)
rng = np.random.default_rng(1)


def log(m):
    print(time.strftime('%H:%M:%S'), m, flush=True)


U = pr.vectors(K)
best = (pr.err2(U, W), K.clone(), W.clone())
log(f'start error_2 {1e3 * best[0]:.4f} mHa')
for it in range(sweeps):
    R, B, C = W.shape
    order = [(r, b) for r in range(R) for b in rng.permutation(B)]
    temp = temp0 * (1 - it / max(1, sweeps)) if temp0 else 0.0
    K, acc, e0, e1 = pr.sweep(K, W, order=order, max_step=max_step, temp=temp, rng=rng, inner=inner)
    U = pr.vectors(K)
    W = pr.refit_groups(U, W, sweeps=2) if kappa < 0 else pr.fit_w(U, W, steps=300, kappa=kappa)
    e = pr.err2(U, W)
    if e < best[0]:
        best = (e, K.clone(), W.clone())
    if it % 5 == 0 or it == sweeps - 1:
        log(f'sweep {it}: accepted {acc}, fixed-W {1e3 * e0:.4f} -> {1e3 * e1:.4f}, after W refit {1e3 * e:.4f} mHa D {L.dpen(W).item():.1f} (best {1e3 * best[0]:.4f})')
        np.savez(out, K=best[1].cpu().numpy(), W=best[2].cpu().numpy())
np.savez(out, K=best[1].cpu().numpy(), W=best[2].cpu().numpy())
log(f'done best {1e3 * best[0]:.4f} mHa')
