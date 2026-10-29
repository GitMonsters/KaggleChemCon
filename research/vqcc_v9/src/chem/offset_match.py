"""Offset initialization for a re-shaped dictionary: per group rho, choose the new offsets W_rc so that the
one-body correction sum_c W_rc X_rc matches the base's (whose offsets were lambda-optimized), i.e. h1' stays
close to the base h1'.  Least squares over the lower-triangle entries (ridge-regularized), then H2_I = W + sum_b w.

Usage: python offset_match.py <assembled npz> <base npz> <out npz> [ridge]
"""
import json
import sys

import numpy as np

sys.path[:0] = ['/home/omen/vqcwork/comp/verifiers', '/home/omen/vqcwork/comp/kickmix']
from dfthcsos import verify as V
from dfthcsos.scoreinfra import dfthc_math as M

src, base, dst = sys.argv[1], sys.argv[2], sys.argv[3]
ridge = float(sys.argv[4]) if len(sys.argv) > 4 else 1e-6


def xmats(path):
    z = dict(np.load(path))
    inp = V.load_inputs(path)
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
    return z, inp, X


z, inp, X = xmats(src)
zb, inpb, Xb = xmats(base)
R, C = X.shape[:2]
N = X.shape[-1]
rows, cols = np.tril_indices(N)
Wb = np.asarray(inpb.W, dtype=np.float64)
Wn = np.zeros((R, C))
for r in range(R):
    target = np.einsum('c,cpq->pq', Wb[r], Xb[r])[rows, cols]
    A = X[r][:, rows, cols].T                                   # (P, C)
    Wn[r] = np.linalg.solve(A.T @ A + ridge * np.eye(C), A.T @ target)
    res = np.linalg.norm(A @ Wn[r] - target) / np.linalg.norm(target)
    print(f'group {r}: relative mismatch of sum_c W X: {res:.4f}, |W| max {np.abs(Wn[r]).max():.3f}', flush=True)
sw = np.asarray(inp.w, dtype=np.float64).sum(axis=1)
z['offset_W'] = Wn
z['H2_I'] = Wn + sw
np.savez(dst, **z)
m = V.evaluate_words(V.load_inputs(dst), z['angle_words'], z['tree_edges'], int(z['brot']))
print('official', json.dumps({k: float(m[k]) for k in ('error_1', 'error_2', 'Lambda', 'E_gap', 'lambda_eff')}))
