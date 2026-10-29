"""Calibration on mrmeows' (4, 303, 128): continuous refit from their words, then re-quantize with the
exact-K refit + discrete sweep quantizer; compare with their 19.999 mHa."""
import json
import sys
import time

import numpy as np
import torch

sys.path.insert(0, '/mnt/c/Users/omen/Desktop/quantumAI/paradigm/chem')
import q8lib as L
from q8opt import Problem

torch.backends.cuda.matmul.allow_tf32 = False
out = sys.argv[1]
z = L.load_submission('/home/omen/vqcwork/chem8/base/dfthcsos/submission.npz')
R, B, C = z['W'].shape
pr = Problem(z['tree_edges'])
sw = L.signed_words(z['angle_words'], 8)
K0 = torch.tensor(sw[76:].reshape(R, B, 75), device='cuda')
W0 = torch.tensor(z['W'], device='cuda')


def log(m):
    print(time.strftime('%H:%M:%S'), m, flush=True)


U0 = pr.vectors(K0)
log(f'base error_2 {1e3 * pr.err2(U0, W0):.4f} mHa')
W1 = pr.refit_groups(U0, W0, sweeps=3)
log(f'exact K refit on base words: error_2 {1e3 * pr.err2(U0, W1):.4f} mHa D {L.dpen(W1).item():.1f}')
# continuous refit from the quantized point
th0 = K0.double() / 128
th, w = pr.fit_continuous(th0, W0, steps=int(sys.argv[2]) if len(sys.argv) > 2 else 20000, kappa=float(sys.argv[3]) if len(sys.argv) > 3 else 1e-4, log=log)
Uc = pr.tree64(th.double())
log(f'continuous: error_2 {1e3 * pr.err2(Uc, w):.4f} mHa D {L.dpen(w.double()).item():.1f}')
np.savez(out, theta=th.cpu().numpy(), W=w.cpu().numpy())
# naive rounding + exact K refit
Kn = torch.clamp(torch.round(th.double() * 128), -64, 63).long()
Un = pr.vectors(Kn)
Wn = pr.refit_groups(Un, w.double(), sweeps=3)
log(f'naive rounding: fixed-W {1e3 * pr.err2(Un, w):.3f} -> K refit {1e3 * pr.err2(Un, Wn):.3f} mHa')
K, W = Kn, Wn
for it in range(int(sys.argv[4]) if len(sys.argv) > 4 else 12):
    K, acc, e0, e1 = pr.sweep(K, W, max_step=1)
    U = pr.vectors(K)
    W = pr.refit_groups(U, W, sweeps=2)
    log(f'sweep {it}: accepted {acc}, fixed-W {1e3 * e0:.3f} -> {1e3 * e1:.3f}, after K refit {1e3 * pr.err2(U, W):.4f} mHa, D {L.dpen(W).item():.1f}')
np.savez(out.replace('.npz', '_q.npz'), K=K.cpu().numpy(), W=W.cpu().numpy())
