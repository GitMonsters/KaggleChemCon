"""Frontier map at 8 bits: shrink the continuous (4, 303, 128) solution to a target shape, refit continuously,
then progressive 8-bit quantization (high-impact first, 5% steps) and a few lambda-aware word sweeps.

Usage: python run_frontier.py <tree npz> <continuous npz> <R> <B> <C> <out prefix> [cont_steps] [sweeps]
"""
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, '/mnt/c/Users/omen/Desktop/quantumAI/paradigm/chem')
import q8lib as L
from q8opt import Problem, P

torch.backends.cuda.matmul.allow_tf32 = False
tree_npz, cont = sys.argv[1], sys.argv[2]
R1, B1, C1 = int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
prefix = sys.argv[6]
cont_steps = int(sys.argv[7]) if len(sys.argv) > 7 else 20000
sweeps = int(sys.argv[8]) if len(sys.argv) > 8 else 5
KAPPA = float(os.environ.get('FR_KAPPA', '2e-5'))
FRACS = os.environ.get('PQ_FRACS', '0.0025,0.005,0.01,0.015,0.02,0.03,0.04,0.05,0.07,0.1,0.15,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95,1.0')
z = L.load_submission(tree_npz)
pr = Problem(z['tree_edges'])
c = np.load(cont)
theta = torch.tensor(c['theta'], device='cuda').double()
W = torch.tensor(c['W'], device='cuda').double()


def log(m):
    print(time.strftime('%H:%M:%S'), m, flush=True)


R, B, C = W.shape
assert R1 == R
# columns: drop the lowest-energy columns, in a gauge where columns are K's eigenvectors (energy-ordered)
if C1 < C:
    Ws = []
    for r in range(R):
        ev, V = torch.linalg.eigh(W[r] @ W[r].T)
        Ws.append(V[:, -C1:] * ev[-C1:].clamp(min=0).sqrt())
    W = torch.stack(Ws)
    log(f'eigen-truncated columns {C} -> {C1}: error_2 {1e3 * pr.err2(pr.tree64(theta), W):.3f} mHa D {L.dpen(W).item():.1f}')
    W = pr.min_d_gauge(W)
    log(f'D-minimizing gauge: error_2 {1e3 * pr.err2(pr.tree64(theta), W):.3f} mHa D {L.dpen(W).item():.1f}')
# vectors: greedy removal by exact fixed-W cost, a few at a time
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
    th, w = pr.fit_continuous(theta, W, steps=1500, lr_th=5e-5, lr_w=3e-5, kappa=2e-5, log=lambda s: None, every=10 ** 9)
    theta, W = th.double(), w.double()
log(f'pruned to {(R, B, W.shape[2])}: error_2 {1e3 * pr.err2(pr.tree64(theta), W):.3f} mHa')
th, w = pr.fit_continuous(theta, W, steps=cont_steps, lr_th=1e-4, lr_w=5e-5, kappa=KAPPA, log=log, every=5000)
theta, W = th.double(), w.double()
ec = pr.err2(pr.tree64(theta), W)
log(f'FRONTIER continuous {(R, B, W.shape[2])}: {1e3 * ec:.4f} mHa D {L.dpen(W).item():.1f}')
np.savez(f'{prefix}_cont.npz', theta=theta.float().cpu().numpy(), W=W.float().cpu().numpy())
fracs = tuple(float(v) for v in FRACS.split(','))
K, Wq = pr.progressive(theta, W, fracs, steps=1500, final_steps=4000, log=log, high_first=True, kappa=KAPPA)
U = pr.vectors(K)
Wq = pr.fit_w(U, Wq, steps=600, kappa=KAPPA)
e8 = pr.err2(U, Wq)
log(f'FRONTIER 8-bit PQ {(R, B, W.shape[2])}: {1e3 * e8:.4f} mHa D {L.dpen(Wq).item():.1f}')
np.savez(f'{prefix}_pq.npz', K=K.cpu().numpy(), W=Wq.cpu().numpy())
rng = np.random.default_rng(0)
best = (e8, K.clone(), Wq.clone())
for it in range(sweeps):
    order = [(r, b) for r in range(R) for b in rng.permutation(B)]
    K, acc, e0, e1 = pr.sweep(K, Wq, order=order, max_step=1, inner=20)
    U = pr.vectors(K)
    Wq = pr.fit_w(U, Wq, steps=300, kappa=KAPPA)
    e = pr.err2(U, Wq)
    if e < best[0]:
        best = (e, K.clone(), Wq.clone())
    log(f'sweep {it}: {1e3 * e:.4f} mHa (best {1e3 * best[0]:.4f})')
    np.savez(f'{prefix}_ls.npz', K=best[1].cpu().numpy(), W=best[2].cpu().numpy())
log(f'FRONTIER 8-bit PQ+LS {(R, B, W.shape[2])}: {1e3 * best[0]:.4f} mHa')
