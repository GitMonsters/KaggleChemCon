"""Shrink a continuous (theta, W) state to (R, B1, C1): columns by eigen-truncation of K_r = W_r W_r^T followed by
a D-minimizing column rotation, vectors by exact fixed-W removal cost in chunks with short refits.

Usage: python prune_shape.py <tree npz> <in npz> <B1> <C1> <out npz> [kappa]
"""
import sys
import time

import numpy as np
import torch

sys.path.insert(0, '/mnt/c/Users/omen/Desktop/quantumAI/paradigm/chem')
import q8lib as L
from q8opt import Problem, P

torch.backends.cuda.matmul.allow_tf32 = False
tree_npz, src, B1, C1, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
kappa = float(sys.argv[6]) if len(sys.argv) > 6 else 3e-5
z = L.load_submission(tree_npz)
pr = Problem(z['tree_edges'])
c = np.load(src)
theta = torch.tensor(c['theta'], device='cuda').double()
W = torch.tensor(c['W'], device='cuda').double()
R, B, C = W.shape


def log(m):
    print(time.strftime('%H:%M:%S'), m, flush=True)


def err():
    return 1e3 * pr.err2(pr.tree64(theta), W)


log(f'start {(R, B, C)}: {err():.3f} mHa D {L.dpen(W).item():.1f}')
if C1 < C:
    Ws = []
    for r in range(R):
        ev, V = torch.linalg.eigh(W[r] @ W[r].T)
        Ws.append(V[:, -C1:] * ev[-C1:].clamp(min=0).sqrt())
    W = torch.stack(Ws)
    log(f'eigen-truncated C {C} -> {C1}: {err():.3f} mHa D {L.dpen(W).item():.1f}')
    W = pr.min_d_gauge(W, steps=4000)
    log(f'D-min gauge: {err():.3f} mHa D {L.dpen(W).item():.1f}')
    th, w = pr.fit_continuous(theta, W, steps=3000, lr_th=3e-5, lr_w=3e-5, kappa=kappa, log=lambda s: None, every=10 ** 9)
    theta, W = th.double(), w.double()
    log(f'after 3000-step refit: {err():.3f} mHa D {L.dpen(W).item():.1f}')
while B > B1:
    m = min(8, B - B1)
    U = pr.tree64(theta)
    p = pr.pair64(U)
    X = torch.einsum('rbm,rbc->rcm', p, W)
    Xf = X.reshape(-1, P)
    E = Xf.T @ Xf - pr.tb64.T
    Y = torch.einsum('rbc,rcm->rbm', W, X)
    om = (W * W).sum(-1)
    yy = (Y * Y).sum(-1)
    G = (Y.reshape(-1, P) @ E).reshape(R, B, P)
    d = -p
    n2 = (d * d).sum(-1)
    dy = (d * Y).sum(-1)
    t2 = 2.0 * om * ((d.reshape(-1, P) @ E).reshape(R, B, P) * d).sum(-1)
    cost = 4.0 * (d * G).sum(-1) + t2 + 2.0 * n2 * yy + 2.0 * dy * dy + 4.0 * om * dy * n2 + om * om * n2 * n2
    keep = torch.argsort(cost, dim=1)[:, m:].sort(dim=1).values
    theta = torch.gather(theta, 1, keep[:, :, None].expand(R, B - m, 75))
    W = torch.gather(W, 1, keep[:, :, None].expand(R, B - m, W.shape[2]))
    B -= m
    th, w = pr.fit_continuous(theta, W, steps=1500, lr_th=5e-5, lr_w=3e-5, kappa=kappa, log=lambda s: None, every=10 ** 9)
    theta, W = th.double(), w.double()
    log(f'B -> {B}: {err():.3f} mHa D {L.dpen(W).item():.1f}')
np.savez(out, theta=theta.float().cpu().numpy(), W=W.float().cpu().numpy())
log(f'saved {out} shape {tuple(W.shape)}')
