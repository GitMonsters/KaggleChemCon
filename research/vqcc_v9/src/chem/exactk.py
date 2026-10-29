"""Exact per-group least-squares refit of K_r = W_r W_r^T for fixed 8-bit words, then a D-minimizing column
gauge; prints error_2 and D before/after.  Usage: python exactk.py <tree npz> <state npz (K, W)> <out npz> [sweeps]"""
import sys

import numpy as np
import torch

sys.path.insert(0, '/mnt/c/Users/omen/Desktop/quantumAI/paradigm/chem')
import q8lib as L
from q8opt import Problem

torch.backends.cuda.matmul.allow_tf32 = False
tree_npz, state, out = sys.argv[1], sys.argv[2], sys.argv[3]
sweeps = int(sys.argv[4]) if len(sys.argv) > 4 else 3
z = L.load_submission(tree_npz)
pr = Problem(z['tree_edges'])
s = np.load(state)
K = torch.tensor(s['K'], device='cuda')
W = torch.tensor(s['W'], device='cuda').double()
U = pr.vectors(K)
print(f'start: error_2 {1e3 * pr.err2(U, W):.4f} mHa D {L.dpen(W).item():.2f}', flush=True)
Wk = pr.refit_groups(U, W, sweeps=sweeps)
print(f'exact K refit: error_2 {1e3 * pr.err2(U, Wk):.4f} mHa D {L.dpen(Wk).item():.2f}', flush=True)
Wg = pr.min_d_gauge(Wk, steps=6000)
print(f'D-min gauge: error_2 {1e3 * pr.err2(U, Wg):.4f} mHa D {L.dpen(Wg).item():.2f}', flush=True)
for kappa in (3e-5, 1e-4):
    Wf = pr.fit_w(U, Wg, steps=1500, kappa=kappa)
    print(f'lambda-aware polish kappa {kappa}: error_2 {1e3 * pr.err2(U, Wf):.4f} mHa D {L.dpen(Wf).item():.2f}', flush=True)
    np.savez(out.replace('.npz', f'_k{kappa:.0e}.npz'), K=K.cpu().numpy(), W=Wf.cpu().numpy())
np.savez(out, K=K.cpu().numpy(), W=Wg.cpu().numpy())
