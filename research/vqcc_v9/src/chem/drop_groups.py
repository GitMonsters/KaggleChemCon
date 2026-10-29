"""Reduce R by dropping whole groups: at each step remove the group whose removal raises the (fixed-rest) error_2
least, refit continuously, repeat; then a long checkpointed refit is done separately (run_cont_ckpt.py).

Usage: python drop_groups.py <tree npz> <init npz (theta, W) or 8-bit npz> <R target> <out npz> [refit steps] [kappa]
"""
import sys
import time

import numpy as np
import torch

sys.path.insert(0, '/mnt/c/Users/omen/Desktop/quantumAI/paradigm/chem')
import q8lib as L
from q8opt import Problem

torch.backends.cuda.matmul.allow_tf32 = False
tree_npz, src, R1, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
steps = int(sys.argv[5]) if len(sys.argv) > 5 else 6000
kappa = float(sys.argv[6]) if len(sys.argv) > 6 else 2e-5
z = L.load_submission(tree_npz)
pr = Problem(z['tree_edges'])
c = np.load(src)
if 'theta' in c.files:
    theta = torch.tensor(c['theta'], device='cuda').double()
    W = torch.tensor(c['W'], device='cuda').double()
else:                                         # 8-bit submission npz: unfreeze its two-body words
    R, B, C = c['W'].shape
    K = L.signed_words(c['angle_words'], int(c['brot']))[76:].reshape(R, B, 75)
    theta = torch.tensor(K / 128.0, device='cuda').double()
    W = torch.tensor(c['W'], device='cuda').double()


def log(m):
    print(time.strftime('%H:%M:%S'), m, flush=True)


def err(th, w):
    return 1e3 * pr.err2(pr.tree64(th), w)


log(f'start {tuple(W.shape)}: {err(theta, W):.3f} mHa')
if 'theta' not in c.files:
    th, w = pr.fit_continuous(theta, W, steps=steps, lr_th=5e-5, lr_w=3e-5, kappa=kappa, log=lambda s: None, every=10 ** 9)
    theta, W = th.double(), w.double()
    log(f'unfrozen + {steps} steps: {err(theta, W):.3f} mHa')
while W.shape[0] > R1:
    R = W.shape[0]
    costs = []
    for r in range(R):
        keep = [s for s in range(R) if s != r]
        costs.append(err(theta[keep], W[keep]))
    r = int(np.argmin(costs))
    keep = [s for s in range(R) if s != r]
    theta, W = theta[keep].contiguous(), W[keep].contiguous()
    log(f'drop group {r} (cost {costs[r]:.2f} mHa; others {sorted(costs)[:3]})')
    th, w = pr.fit_continuous(theta, W, steps=steps, lr_th=5e-5, lr_w=3e-5, kappa=kappa, log=lambda s: None, every=10 ** 9)
    theta, W = th.double(), w.double()
    log(f'R={W.shape[0]} after {steps} refit steps: {err(theta, W):.3f} mHa D {L.dpen(W).item():.2f}')
np.savez(out, theta=theta.float().cpu().numpy(), W=W.float().cpu().numpy())
log(f'saved {out}')
