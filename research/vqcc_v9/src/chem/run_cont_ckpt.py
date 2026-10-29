"""Resumable continuous (theta, W) fit with cosine LR; checkpoints every `every` steps (WSL restarts kill jobs).

Usage: python run_cont_ckpt.py <tree npz> <init npz (theta, W)> <ckpt prefix> <total steps> [kappa] [lr_th] [lr_w]
Writes <prefix>.pt (resume state) and <prefix>_best.npz (theta, W of the best objective so far).
"""
import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, '/mnt/c/Users/omen/Desktop/quantumAI/paradigm/chem')
import q8lib as L
from q8opt import Problem, P

torch.backends.cuda.matmul.allow_tf32 = False
tree_npz, init, prefix, total = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
kappa = float(sys.argv[5]) if len(sys.argv) > 5 else 3e-5
lr_th = float(sys.argv[6]) if len(sys.argv) > 6 else 5e-5
lr_w = float(sys.argv[7]) if len(sys.argv) > 7 else 3e-5
every = 2000
z = L.load_submission(tree_npz)
pr = Problem(z['tree_edges'])


def log(m):
    print(time.strftime('%H:%M:%S'), m, flush=True)


ck = prefix + '.pt'
if os.path.exists(ck):
    st = torch.load(ck, map_location='cuda')
    th = st['theta'].requires_grad_(True)
    w = st['W'].requires_grad_(True)
    step0, best = st['step'], st['best']
    opt = torch.optim.Adam([{'params': [th], 'lr': lr_th}, {'params': [w], 'lr': lr_w}])
    opt.load_state_dict(st['opt'])
    log(f'resumed at step {step0}, best objective {best:.6e}')
else:
    c = np.load(init)
    th = torch.tensor(c['theta'], device='cuda').float().requires_grad_(True)
    w = torch.tensor(c['W'], device='cuda').float().requires_grad_(True)
    step0, best = 0, float('inf')
    opt = torch.optim.Adam([{'params': [th], 'lr': lr_th}, {'params': [w], 'lr': lr_w}])
base = [lr_th, lr_w]
t0 = time.time()
for s in range(step0, total + 1):
    f = 0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * s / total))
    for g, b in zip(opt.param_groups, base):
        g['lr'] = b * f
    U = pr.tree32(th)
    v = pr.pair32(U)
    a = torch.einsum('rbm,rbc->rcm', v, w).reshape(-1, P)
    e = a.T @ a - pr.T32
    err = torch.sqrt((e * e).sum())
    dp = L.dpen(w)
    tot = err + kappa * dp
    if s % 100 == 0 and tot.item() < best:
        best = tot.item()
        np.savez(prefix + '_best.npz', theta=th.detach().cpu().numpy(), W=w.detach().cpu().numpy())
    if s % every == 0 or s == total:
        log(f'step {s:6d} err {1e3 * err.item():8.4f} mHa D {dp.item():7.2f} lr {f:.3f}x ({time.time() - t0:.0f}s)')
        torch.save({'theta': th.detach(), 'W': w.detach(), 'step': s, 'best': best, 'opt': opt.state_dict()}, ck + '.tmp')
        os.replace(ck + '.tmp', ck)
    if s == total:
        break
    opt.zero_grad(set_to_none=True)
    tot.backward()
    opt.step()
b = np.load(prefix + '_best.npz')
e = pr.err2(pr.tree64(torch.tensor(b['theta'], device='cuda').double()), torch.tensor(b['W'], device='cuda').double())
log(f'done: best state error_2 {1e3 * e:.4f} mHa')
