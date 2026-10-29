"""Sparse (masked-support) dictionary: each vector keeps only its s largest orbital components.

With exact zeros outside the support, every Givens angle that only moves the whole remaining mass (or none)
is exactly 0 or -pi/2 on the 8-bit grid, so quantization noise comes only from the support's branch angles.
Continuous masked refit, then statistics of nontrivial angles; saves (theta, W, mask).

Usage: python run_sparse.py <tree npz> <continuous npz (theta, W)> <s> <steps> <out npz>
"""
import sys
import time

import numpy as np
import torch

sys.path.insert(0, '/mnt/c/Users/omen/Desktop/quantumAI/paradigm/chem')
import q8lib as L
from q8opt import Problem, P

torch.backends.cuda.matmul.allow_tf32 = False
tree_npz, cont, s, steps, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
kappa = 2e-5
z = L.load_submission(tree_npz)
pr = Problem(z['tree_edges'])
c = np.load(cont)
theta = torch.tensor(c['theta'], device='cuda')
W = torch.tensor(c['W'], device='cuda').float()
U0 = pr.tree64(theta.double()).float()                       # (R, B, N)
R, B, N = U0.shape


def log(m):
    print(time.strftime('%H:%M:%S'), m, flush=True)


idx = torch.argsort(-(U0 * U0), dim=-1)[..., :s]
mask = torch.zeros_like(U0)
mask.scatter_(-1, idx, 1.0)
kept = ((U0 * U0) * mask).sum(-1)
log(f's={s}: kept mass quantiles {np.quantile(kept.cpu().numpy(), [0.01, 0.1, 0.5]).round(4)}')
V = (U0 * mask).clone().requires_grad_(True)
w = W.clone().requires_grad_(True)
opt = torch.optim.Adam([{'params': [V], 'lr': 3e-4}, {'params': [w], 'lr': 5e-5}])
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps, eta_min=1e-6)
best = (float('inf'), None, None)
for st in range(steps + 1):
    u = V * mask
    u = u / u.norm(dim=-1, keepdim=True)
    v = pr.pair32(u)
    a = torch.einsum('rbm,rbc->rcm', v, w).reshape(-1, P)
    e = a.T @ a - pr.T32
    err = torch.sqrt((e * e).sum())
    tot = err + kappa * L.dpen(w)
    if st % 100 == 0 and tot.item() < best[0]:
        best = (tot.item(), u.detach().clone(), w.detach().clone())
    if st % 2000 == 0 or st == steps:
        log(f'  step {st:6d} err {1e3 * err.item():8.3f} mHa D {L.dpen(w).item():.2f}')
    if st == steps:
        break
    opt.zero_grad(set_to_none=True)
    tot.backward()
    opt.step()
    sched.step()
u, w = best[1], best[2]
th = torch.tensor(L.gauge_angles(u.reshape(-1, N).double().cpu().numpy(), pr.edges), device='cuda').reshape(R, B, N - 1)
k = th * 128
trivial = (k - torch.round(k)).abs() < 1e-6
print('nontrivial (off-grid) angles per vector: quantiles', np.quantile((~trivial).sum(-1).cpu().numpy(), [0.1, 0.5, 0.9]))
# sensitivity-weighted: amplitude at p when edge processed ~ |du/dtheta|
np.savez(out, theta=th.float().cpu().numpy(), W=w.cpu().numpy(), mask=mask.cpu().numpy())
log(f'continuous masked error_2 {1e3 * pr.err2(u.double(), w.double()):.4f} mHa')
