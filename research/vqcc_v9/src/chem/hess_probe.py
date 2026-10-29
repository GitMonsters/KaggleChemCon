"""How coupled are a vector's 75 angle words?  Local Gauss-Newton Hessian (fixed W) in word units and the
quantization MSE of independent rounding vs. Babai (sorted) vs. coordinate descent on the quadratic model."""
import sys

import numpy as np
import torch
from torch.func import jacfwd, vmap

sys.path.insert(0, '/mnt/c/Users/omen/Desktop/quantumAI/paradigm/chem')
import q8lib as L
from q8opt import Problem

z = L.load_submission('/home/omen/vqcwork/chem8/base/dfthcsos/submission.npz')
R, B, C = z['W'].shape
pr = Problem(z['tree_edges'])
sw = L.signed_words(z['angle_words'], 8)
K = torch.tensor(sw[76:].reshape(R, B, 75), device='cuda')
W = torch.tensor(z['W'], device='cuda')
Wf, p, X, E = pr.state(K, W)
Y = torch.einsum('rbc,rcm->rbm', Wf, X)
om = (Wf * Wf).sum(-1)
rng = np.random.default_rng(0)
tree = pr.tree64
pair = pr.pair64


def pvec(th):
    return pair(tree(th[None, :]))[0]


res = []
for r, b in zip(rng.integers(0, R, 40), rng.integers(0, B, 40)):
    th = (K[r, b].double() / 128.0)
    J = jacfwd(pvec)(th) / 128.0                      # dp / d(word), (P, 75)
    y = Y[r, b].double()
    Ed = E.double()
    Hm = 2 * (y @ y) * (J.T @ J) + 2 * torch.outer(J.T @ y, J.T @ y) + 2 * om[r, b].double() * (J.T @ Ed @ J)
    H = Hm.cpu().numpy()
    d = np.diag(H)
    Dm = np.sqrt(np.outer(d, d)) + 1e-30
    corr = np.abs(H / Dm - np.eye(75)).max()
    # MSE of quantization for random targets z in [0,1)^75 (quadratic model)
    ev = np.linalg.eigvalsh(H)
    mse_round = d.sum() / 12.0
    # Babai nearest plane with columns sorted by decreasing diagonal
    trials = 200
    order = np.argsort(-d)
    Hs = H[np.ix_(order, order)]
    try:
        Lc = np.linalg.cholesky(Hs + 1e-15 * np.eye(75) * d.max())
    except np.linalg.LinAlgError:
        continue
    Rm = Lc.T                                           # Hs = R^T R, R upper
    errs_b, errs_r, errs_cd = [], [], []
    for _ in range(trials):
        zt = rng.random(75)
        zz = zt[order]
        # rounding
        dr = np.round(zz) - zz
        errs_r.append(dr @ Hs @ dr)
        # Babai: solve R (k - z) small, back-substitution from the last coordinate
        k = np.zeros(75)
        for i in reversed(range(75)):
            s = zz[i] - (Rm[i, i + 1:] @ (k[i + 1:] - zz[i + 1:])) / Rm[i, i]
            k[i] = np.round(s)
        db = k - zz
        errs_b.append(db @ Hs @ db)
        # coordinate descent on the quadratic from rounding
        kk = np.round(zz)
        for _ in range(20):
            improved = False
            for i in range(75):
                for st in (-1, 1):
                    trial = kk.copy(); trial[i] += st
                    a = trial - zz; b0 = kk - zz
                    if a @ Hs @ a < b0 @ Hs @ b0 - 1e-18:
                        kk = trial; improved = True
            if not improved:
                break
        dc = kk - zz
        errs_cd.append(dc @ Hs @ dc)
    res.append((np.mean(errs_r), np.mean(errs_b), np.mean(errs_cd), corr, ev.max() / max(ev.min(), 1e-30)))
res = np.array(res)
print('mean MSE: rounding %.3e  babai(sorted) %.3e  coord-descent %.3e' % tuple(res[:, :3].mean(0)))
print('ratio babai/round %.3f  cd/round %.3f' % (res[:, 1].mean() / res[:, 0].mean(), res[:, 2].mean() / res[:, 0].mean()))
print('max off-diagonal correlation (median over vectors) %.3f, condition number median %.2e' % (np.median(res[:, 3]), np.median(res[:, 4])))
