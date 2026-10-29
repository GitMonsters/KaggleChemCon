"""Offset-only lambda optimization: the identity offsets W_rc (= H2_I - sum_b w) do not touch error_2, and
lambda_eff depends on them only through h1' = A - sum W_rc X_rc (76x76 spectrum), the lambda_h2 term and
-1/2 sum W^2 in E_SOS.  Minimize lambda_eff s.t. E_gap >= gap_min with Adam + quadratic penalty (float64,
autograd through eigvalsh), then write H2_I/offset_W.  One-body rows must be refreshed afterwards.

Usage: python offset_opt.py <in npz> <out npz> [steps] [gap_min]
"""
import json
import sys

import numpy as np
import torch

sys.path[:0] = ['/home/omen/vqcwork/comp/verifiers', '/home/omen/vqcwork/comp/kickmix']
from dfthcsos import verify as V
from dfthcsos.scoreinfra import dfthc_math as M

src, dst = sys.argv[1], sys.argv[2]
steps = int(sys.argv[3]) if len(sys.argv) > 3 else 4000
gap_min = float(sys.argv[4]) if len(sys.argv) > 4 else 3.91508
torch.set_num_threads(2)
z = dict(np.load(src))
inp = V.load_inputs(src)
brot = int(z['brot'])
half = 1 << (brot - 1)
words = np.asarray(z['angle_words'], dtype=np.int64)
signed = ((words + half) % (1 << brot)) - half
vec = M.reconstruct_u_vectors(signed / half, z['tree_edges'])
vec = vec / np.linalg.norm(vec, axis=1, keepdims=True)
w = np.asarray(inp.w, dtype=np.float64)
_, u_b = M.split_u_vectors(vec, w)
X = M.reconstruct_x(u_b, w)
X = np.tril(X) + np.swapaxes(np.tril(X, k=-1), -1, -2)
ref = inp.ref
eta = ref['electrons']
h1_thc = ref['oei'] - 0.5 * np.einsum('prrq->pq', ref['tei'])
A = h1_thc - 0.5 * eta * inp.S + inp.beta_1 * np.eye(h1_thc.shape[0])
const = float(ref['ecore'] - eta * inp.beta_1)
E_up = float(inp.energy_upper)
l1 = np.abs(w).sum(axis=1)                       # ||w_rc||_1  (R, C)
sw = w.sum(axis=1)                               # sum_b w     (R, C)

At = torch.tensor(A)
Xt = torch.tensor(X)
l1t, swt = torch.tensor(l1), torch.tensor(sw)
Wt = torch.tensor(np.asarray(inp.W, dtype=np.float64), requires_grad=True)


def metrics(W):
    h1p = At - torch.einsum('rc,rcpq->pq', W, Xt)
    ev = torch.linalg.eigvalsh(h1p)
    lam_h1 = ev.abs().sum()
    lam_h2 = 0.25 * ((l1t + (W + swt).abs()) ** 2).sum()
    Lam = lam_h1 + lam_h2
    E_sos = const - (lam_h1 - ev.sum()) - 0.5 * (W * W).sum()
    gap = E_up - E_sos
    lam_eff = torch.sqrt(gap * (2 * Lam - gap))
    return lam_eff, gap, Lam


with torch.no_grad():
    l0, g0, L0 = metrics(Wt)
print(f'start: lambda_eff {l0.item():.5f} E_gap {g0.item():.5f} Lambda {L0.item():.4f}', flush=True)
opt = torch.optim.Adam([Wt], lr=2e-3)
best = (float('inf'), Wt.detach().clone())
mu = 1e4
for s in range(steps + 1):
    lam, gap, Lam = metrics(Wt)
    viol = torch.relu(gap_min - gap)
    loss = lam + mu * viol * viol
    if gap.item() >= gap_min and lam.item() < best[0]:
        best = (lam.item(), Wt.detach().clone())
    if s % 500 == 0 or s == steps:
        print(f'step {s:5d} lambda_eff {lam.item():.5f} E_gap {gap.item():.5f} Lambda {Lam.item():.4f} '
              f'(best feasible {best[0]:.5f})', flush=True)
    if s == steps:
        break
    opt.zero_grad()
    loss.backward()
    opt.step()
    for g in opt.param_groups:
        g['lr'] = 2e-3 * (0.05 ** (s / steps))
Wb = best[1].numpy()
z['offset_W'] = Wb
z['H2_I'] = Wb + sw
np.savez(dst, **z)
m = V.evaluate_words(V.load_inputs(dst), z['angle_words'], z['tree_edges'], brot)
print('official', json.dumps({k: float(m[k]) for k in ('error_1', 'error_2', 'Lambda', 'E_gap', 'lambda_eff')}))
