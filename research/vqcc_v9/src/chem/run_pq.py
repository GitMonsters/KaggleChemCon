"""Progressive 8-bit quantization from a continuous state, then exact-K refit and discrete sweeps.

Usage: python run_pq.py <tree npz> <continuous npz (theta, W)> <out npz> [sweeps] [inner] [steps] [final_steps]
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
sweeps = int(sys.argv[4]) if len(sys.argv) > 4 else 30
inner = int(sys.argv[5]) if len(sys.argv) > 5 else 20
steps = int(sys.argv[6]) if len(sys.argv) > 6 else 1500
final_steps = int(sys.argv[7]) if len(sys.argv) > 7 else 3000
high_first = (sys.argv[8] != 'low') if len(sys.argv) > 8 else True
fine = len(sys.argv) > 9 and sys.argv[9] == 'fine'
custom = sys.argv[9][3:] if len(sys.argv) > 9 and sys.argv[9].startswith('fr:') else None
import os
RAMP = float(os.environ.get('PQ_RAMP', '0'))
z = L.load_submission(tree_npz)
pr = Problem(z['tree_edges'])
c = np.load(cont)
theta = torch.tensor(c['theta'], device='cuda')
W = torch.tensor(c['W'], device='cuda')


def log(m):
    print(time.strftime('%H:%M:%S'), m, flush=True)


fracs = (0.3, 0.5, 0.65, 0.75, 0.83, 0.89, 0.93, 0.95, 0.97, 0.98, 0.985, 0.99, 0.993, 0.996, 0.998, 1.0)
if fine:
    fracs = tuple(round(0.05 * i, 2) for i in range(1, 21))
if custom:
    fracs = tuple(float(v) for v in custom.split(','))
K, W = pr.progressive(theta, W, fracs, steps=steps, final_steps=final_steps, log=log, high_first=high_first, ramp=RAMP)
U = pr.vectors(K)
log(f'PQ result error_2 {1e3 * pr.err2(U, W):.4f} mHa D {L.dpen(W).item():.1f}')
np.savez(out.replace('.npz', '_pq.npz'), K=K.cpu().numpy(), W=W.cpu().numpy())
Wk = pr.fit_w(U, W, steps=600, kappa=2e-5)
log(f'lambda-aware W refit: {1e3 * pr.err2(U, Wk):.4f} mHa')
best = (pr.err2(U, Wk), K.clone(), Wk.clone())
rng = np.random.default_rng(0)
W = Wk
for it in range(sweeps):
    R, B, C = W.shape
    order = [(r, b) for r in range(R) for b in rng.permutation(B)]
    K, acc, e0, e1 = pr.sweep(K, W, order=order, max_step=1, inner=inner)
    U = pr.vectors(K)
    W = pr.fit_w(U, W, steps=300, kappa=2e-5)
    e = pr.err2(U, W)
    if e < best[0]:
        best = (e, K.clone(), W.clone())
    log(f'sweep {it}: moves {acc}, error_2 {1e3 * e:.4f} mHa (best {1e3 * best[0]:.4f})')
    np.savez(out, K=best[1].cpu().numpy(), W=best[2].cpu().numpy())
log(f'done best {1e3 * best[0]:.4f}')
