"""Distribution of predicted rounding noise across vectors and angles (fixed W, first order).

noise_b,k = 4 ||y_b||^2 |dp/dtheta_k|^2 (Delta^2 / 12), Delta = 1 word; reports concentration."""
import sys

import numpy as np
import torch
from torch.func import jacfwd, vmap

sys.path.insert(0, '/mnt/c/Users/omen/Desktop/quantumAI/paradigm/chem')
import q8lib as L
from q8opt import Problem

z = L.load_submission('/home/omen/vqcwork/chem8/base/dfthcsos/submission.npz')
pr = Problem(z['tree_edges'])
c = np.load(sys.argv[1])
theta = torch.tensor(c['theta'], device='cuda').double()
W = torch.tensor(c['W'], device='cuda').double()
R, B, _ = theta.shape
U = pr.tree64(theta)
p = pr.pair64(U)
X = torch.einsum('rbm,rbc->rcm', p, W)
Y = torch.einsum('rbc,rcm->rbm', W, X)
yy = (Y * Y).sum(-1)                                            # (R, B)


def uvec(th):
    return pr.tree64(th[None, :])[0]


Ju = vmap(jacfwd(uvec))(theta.reshape(-1, 75)) / 128.0          # (RB, 76, 75), per word
s = (Ju * Ju).sum(1).reshape(R, B, 75)                          # |du/dword_k|^2 ; |dp|^2 = 2|du|^2
noise = 4 * yy[..., None] * 2 * s / 12.0                        # (R, B, 75) in Ha^2
nb = noise.sum(-1).flatten().cpu().numpy()
na = noise.flatten().cpu().numpy()
tot = nb.sum()
print(f'predicted independent-rounding noise (fixed W): {1e3 * np.sqrt(tot):.2f} mHa rms')
for q in (0.01, 0.05, 0.1, 0.2, 0.5):
    kb = int(q * len(nb)); ka = int(q * len(na))
    print(f'top {q:.0%} vectors carry {np.sort(nb)[::-1][:kb].sum() / tot:.3f}; top {q:.0%} angles carry {np.sort(na)[::-1][:ka].sum() / tot:.3f}')
per_edge = noise.sum((0, 1)).cpu().numpy()
order = np.argsort(-per_edge)
edges = pr.edges
print('top edges by noise share:')
for k in order[:12]:
    print(f'  edge {k:2d} {tuple(edges[k])}: {per_edge[k] / tot:.3f}')
# depth of the parent in the tree
depth = {int(edges[0][0]): 0}
for p, q in edges:
    depth[int(q)] = depth[int(p)] + 1
by_depth = {}
for k, (p, q) in enumerate(edges):
    by_depth.setdefault(depth[int(p)], 0.0)
    by_depth[depth[int(p)]] += per_edge[k]
print('noise share by parent depth:', {d: round(v / tot, 3) for d, v in sorted(by_depth.items())})
