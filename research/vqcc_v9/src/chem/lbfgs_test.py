"""Does L-BFGS converge faster than Adam on the continuous (theta, W) objective?  err + kappa * D_smooth.

Usage: python lbfgs_test.py <tree npz> <state npz> <calls> [kappa] [out npz]
"""
import sys
import time

import numpy as np
import torch

sys.path.insert(0, '/mnt/c/Users/omen/Desktop/quantumAI/paradigm/chem')
import q8lib as L
from q8opt import Problem, P

torch.backends.cuda.matmul.allow_tf32 = False
tree_npz, src, calls = sys.argv[1], sys.argv[2], int(sys.argv[3])
kappa = float(sys.argv[4]) if len(sys.argv) > 4 else 3e-5
out = sys.argv[5] if len(sys.argv) > 5 else None
z = L.load_submission(tree_npz)
pr = Problem(z['tree_edges'])
c = np.load(src)
th = torch.tensor(c['theta'], device='cuda').double().requires_grad_(True)
w = torch.tensor(c['W'], device='cuda').double().requires_grad_(True)
T64 = pr.tb64.T
pair = pr.pair64
tree = pr.tree64


def dsmooth(w, eps=1e-6):
    a = torch.sqrt(w * w + eps * eps).sum(dim=1)
    s = w.sum(dim=1)
    return 0.5 * (a * a - s * s).sum()


def objective():
    U = tree(th)
    v = pair(U)
    a = torch.einsum('rbm,rbc->rcm', v, w).reshape(-1, P)
    e = a.T @ a - T64
    err = torch.sqrt((e * e).sum())
    return err, err + kappa * dsmooth(w)


opt = torch.optim.LBFGS([th, w], lr=1.0, max_iter=20, history_size=100, line_search_fn='strong_wolfe',
                        tolerance_grad=1e-12, tolerance_change=1e-15)
t0 = time.time()
evals = [0]


def closure():
    opt.zero_grad(set_to_none=True)
    err, tot = objective()
    tot.backward()
    evals[0] += 1
    return tot


for k in range(calls):
    opt.step(closure)
    with torch.no_grad():
        err, tot = objective()
    if k % 5 == 0 or k == calls - 1:
        print(time.strftime('%H:%M:%S'), f'call {k:4d} evals {evals[0]:6d} err {1e3 * err.item():.4f} mHa '
              f'D {L.dpen(w.detach()).item():.2f} ({time.time() - t0:.0f}s)', flush=True)
if out:
    np.savez(out, theta=th.detach().float().cpu().numpy(), W=w.detach().float().cpu().numpy())
