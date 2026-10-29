"""Continue a continuous (theta, W) fit. Usage: python run_cont.py <tree npz> <in npz> <out npz> <steps> [kappa] [lr_th] [lr_w]"""
import sys
import time

import numpy as np
import torch

sys.path.insert(0, '/mnt/c/Users/omen/Desktop/quantumAI/paradigm/chem')
import q8lib as L
from q8opt import Problem

torch.backends.cuda.matmul.allow_tf32 = False
tree_npz, src, out, steps = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
kappa = float(sys.argv[5]) if len(sys.argv) > 5 else 3e-5
lr_th = float(sys.argv[6]) if len(sys.argv) > 6 else 5e-5
lr_w = float(sys.argv[7]) if len(sys.argv) > 7 else 3e-5
z = L.load_submission(tree_npz)
pr = Problem(z['tree_edges'])
c = np.load(src)
theta = torch.tensor(c['theta'], device='cuda').double()
W = torch.tensor(c['W'], device='cuda').double()


def log(m):
    print(time.strftime('%H:%M:%S'), m, flush=True)


th, w = pr.fit_continuous(theta, W, steps=steps, lr_th=lr_th, lr_w=lr_w, kappa=kappa, log=log, every=5000)
e = pr.err2(pr.tree64(th.double()), w.double())
log(f'continuous {tuple(w.shape)}: {1e3 * e:.4f} mHa D {L.dpen(w).item():.1f}')
np.savez(out, theta=th.cpu().numpy(), W=w.cpu().numpy())
