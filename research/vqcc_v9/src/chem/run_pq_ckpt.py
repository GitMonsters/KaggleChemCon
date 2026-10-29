"""Resumable progressive 8-bit quantization (PQ): freeze angles to the grid in decreasing predicted-impact
order; after every freeze, Adam re-optimizes the free angles and all weights.  State is checkpointed after
every stage (WSL restarts kill jobs).

Usage: python run_pq_ckpt.py <tree npz> <continuous npz (theta, W)> <prefix>
Env: PQ_FRACS (comma list), PQ_EARLY (steps for stages with frac <= PQ_SPLIT), PQ_LATE, PQ_SPLIT, PQ_FINAL,
     PQ_KAPPA, PQ_LRTH, PQ_LRW, PQ_DECAY (lr multiplier over a stage), PQ_PROBES
Writes <prefix>.pt (resume) and <prefix>_q.npz (K signed words, W) when done.
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
tree_npz, cont, prefix = sys.argv[1], sys.argv[2], sys.argv[3]
env = os.environ.get
FRACS = [float(v) for v in env('PQ_FRACS', '0.0005,0.001,0.0015,0.002,0.003,0.004,0.005,0.0075,0.01,0.015,0.02,'
                                          '0.03,0.04,0.05,0.07,0.1,0.15,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0').split(',')]
EARLY, LATE, SPLIT = int(env('PQ_EARLY', '4000')), int(env('PQ_LATE', '1200')), float(env('PQ_SPLIT', '0.05'))
MID, SPLIT2 = int(env('PQ_MID', env('PQ_LATE', '1200'))), float(env('PQ_SPLIT2', env('PQ_SPLIT', '0.05')))
FINAL = int(env('PQ_FINAL', '4000'))
KAPPA = float(env('PQ_KAPPA', '3e-5'))
LRTH, LRW = float(env('PQ_LRTH', '1e-4')), float(env('PQ_LRW', '5e-5'))
DECAY = float(env('PQ_DECAY', '0.2'))
PROBES = int(env('PQ_PROBES', '8'))
z = L.load_submission(tree_npz)
pr = Problem(z['tree_edges'])
half = float(pr.half)
lo, hi = -0.5, 0.5 - 1.0 / half


def log(m):
    print(time.strftime('%H:%M:%S'), m, flush=True)


def qround(t):
    return torch.clamp(torch.round(t * half) / half, lo, hi)


def loss(th, w):
    U = pr.tree32(th)
    v = pr.pair32(U)
    a = torch.einsum('rbm,rbc->rcm', v, w).reshape(-1, P)
    e = a.T @ a - pr.T32
    return torch.sqrt((e * e).sum()), e


ck = prefix + '.pt'
if os.path.exists(ck):
    st = torch.load(ck, map_location='cuda')
    theta, W, frozen, qv, stage0 = st['theta'], st['W'], st['frozen'], st['qv'], st['stage']
    log(f'resumed after stage {stage0 - 1}, frozen {float(frozen.float().mean()):.4f}')
else:
    c = np.load(cont)
    theta = torch.tensor(c['theta'], device='cuda').float()
    W = torch.tensor(c['W'], device='cuda').float()
    frozen = torch.zeros_like(theta, dtype=torch.bool)
    qv = torch.zeros_like(theta)
    stage0 = 0
    e0, _ = loss(theta, W)
    log(f'start: continuous error_2 {1e3 * e0.item():.4f} mHa D {L.dpen(W).item():.2f}, shape {tuple(W.shape)}')
stages = FRACS + [None]
for stage in range(stage0, len(stages)):
    frac = stages[stage]
    if frac is not None:
        need = int(frac * theta.numel()) - int(frozen.sum())
        if need > 0:
            th = torch.where(frozen, qv, theta).clone().requires_grad_(True)
            _, e = loss(th, W)
            acc = torch.zeros_like(th)
            for _ in range(PROBES):
                zz = torch.randn_like(e)
                g, = torch.autograd.grad((e * (zz + zz.T)).sum(), th, retain_graph=True)
                acc += g * g
            imp = acc * (qround(theta) - theta) ** 2
            imp[frozen] = -1.0
            idx = torch.argsort(imp.flatten(), descending=True)[:need]
            fl, q = frozen.flatten().clone(), qv.flatten().clone()
            fl[idx] = True
            q[idx] = qround(theta).flatten()[idx]
            frozen, qv = fl.view_as(frozen), q.view_as(qv)
    n = FINAL if frac is None else (EARLY if frac <= SPLIT else (MID if frac <= SPLIT2 else LATE))
    thp = theta.clone().requires_grad_(True)
    wp = W.clone().requires_grad_(True)
    groups = [{'params': [wp], 'lr': LRW}] + ([] if bool(frozen.all()) else [{'params': [thp], 'lr': LRTH}])
    opt = torch.optim.Adam(groups)
    ff = frozen.float()
    best = (float('inf'), theta, W, float('inf'))
    first = None
    for s in range(n + 1):
        err, _ = loss(ff * qv + (1 - ff) * thp, wp)
        if first is None:
            first = err.item()
        tot = err + KAPPA * L.dpen(wp)
        if s % 25 == 0 and tot.item() < best[0]:
            best = (tot.item(), thp.detach().clone(), wp.detach().clone(), err.item())
        if s == n:
            break
        opt.zero_grad(set_to_none=True)
        tot.backward()
        opt.step()
        with torch.no_grad():
            thp.clamp_(lo, hi)
        for gr in opt.param_groups:
            gr['lr'] *= DECAY ** (1.0 / n)
    theta, W = best[1], best[2]
    log(f'stage {stage:2d} frozen {float(frozen.float().mean()):.4f}: after freeze {1e3 * first:8.3f} -> '
        f'{1e3 * best[3]:8.3f} mHa D {L.dpen(W).item():.2f} ({n} steps)')
    torch.save({'theta': theta, 'W': W, 'frozen': frozen, 'qv': qv, 'stage': stage + 1}, ck + '.tmp')
    os.replace(ck + '.tmp', ck)
final = torch.where(frozen, qv, qround(theta))
K = torch.round(final.double() * half).long()
Wd = W.double()
e8 = pr.err2(pr.vectors(K), Wd)
np.savez(prefix + '_q.npz', K=K.cpu().numpy(), W=Wd.cpu().numpy())
log(f'PQ DONE {tuple(W.shape)}: 8-bit error_2 {1e3 * e8:.4f} mHa D {L.dpen(Wd).item():.2f} -> {prefix}_q.npz')
