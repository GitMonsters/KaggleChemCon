"""Design the Givens tree for 8-bit quantization: most rounding noise sits on the top routing edges, where a
vector's full amplitude passes but its split is generic.  A tree whose top splits separate orbitals that rarely
share a vector makes those angles 0 or -pi/2 for most vectors (exact on the grid).

noise(tree) = sum_b ||y_b||^2 * 2 ||u_b - u_b(rounded angles in that tree)||^2   (first order, fixed W)
"""
from __future__ import annotations

import sys

import numpy as np

N = 76


def gauge_angles(u, edges):
    a = np.array(u, dtype=np.float64, copy=True)
    ang = np.zeros((a.shape[0], len(edges)))
    for j in reversed(range(len(edges))):
        p, q = edges[j]
        norm = np.hypot(a[:, p], a[:, q])
        sign = np.where(np.abs(a[:, p]) < 1e-12, np.where(a[:, q] > 0, -1.0, 1.0), np.where(a[:, p] < 0, -1.0, 1.0))
        ang[:, j] = np.where(norm < 1e-12, 0.0, np.arctan2(a[:, q] * sign, a[:, p] * sign))
        a[:, p] = np.where(norm < 1e-12, 0.0, norm * sign)
        a[:, q] = 0.0
    return ang / np.pi


def decode(theta, edges):
    u = np.zeros((theta.shape[0], N))
    u[:, 0] = 1.0
    for j, (p, q) in enumerate(edges):
        c, s = np.cos(np.pi * theta[:, j]), np.sin(np.pi * theta[:, j])
        a, b = u[:, p].copy(), u[:, q].copy()
        u[:, p] = c * a - s * b
        u[:, q] = s * a + c * b
    return u


def rounding_noise(U, yy, edges, half=128):
    th = gauge_angles(U, edges)
    k = np.clip(np.round(th * half), -half // 2, half // 2 - 1)
    ur = decode(k / half, edges)
    # projector distance is sign-independent; compare u u^T
    d2 = 2 * (1 - np.sum(U * ur, axis=1) ** 2)            # ||uu^T - vv^T||_F^2 for unit vectors
    return float(np.sum(yy * d2)), d2


def dendrogram_tree(S, root=0):
    """Average-linkage agglomerative clustering on similarity S -> binary tree -> Givens edges rooted at `root`."""
    clusters = {i: [i] for i in range(N)}
    nodes = {i: ('leaf', i) for i in range(N)}
    sim = S.copy().astype(float)
    np.fill_diagonal(sim, -np.inf)
    active = set(range(N))
    nxt = N
    size = {i: 1 for i in range(N)}
    simd = {(i, j): sim[i, j] for i in range(N) for j in range(N) if i < j}
    while len(active) > 1:
        (i, j), _ = max(((k, v) for k, v in simd.items() if k[0] in active and k[1] in active), key=lambda kv: kv[1])
        new = nxt
        nxt += 1
        nodes[new] = ('node', nodes[i], nodes[j])
        clusters[new] = clusters[i] + clusters[j]
        size[new] = size[i] + size[j]
        active.discard(i)
        active.discard(j)
        for k in active:
            a, b = (min(k, i), max(k, i)), (min(k, j), max(k, j))
            v = (simd.get(a, 0) * size[i] + simd.get(b, 0) * size[j]) / (size[i] + size[j])
            simd[(min(k, new), max(k, new))] = v
        active.add(new)
    return realize(nodes[active.pop()], root)


def leaves(t):
    return [t[1]] if t[0] == 'leaf' else leaves(t[1]) + leaves(t[2])


def realize(tree, root=0, weight=None):
    """Binary split tree -> Givens edges; every split moves one side to a new representative orbital."""
    edges = []

    def rep(t):
        ls = leaves(t)
        return ls[0] if weight is None else max(ls, key=lambda v: weight[v])

    def emit(t, p):
        if t[0] == 'leaf':
            return
        A, B = t[1], t[2]
        if p in leaves(A):
            A, B = B, A
        q = rep(A)
        edges.append((p, q))
        emit(A, q)
        emit(B, p)
    assert root in leaves(tree)
    emit(tree, root)
    return np.array(edges, dtype=np.int64)


if __name__ == '__main__':
    import torch
    sys.path.insert(0, '/mnt/c/Users/omen/Desktop/quantumAI/paradigm/chem')
    import q8lib as Lb
    from q8opt import Problem
    z = Lb.load_submission('/home/omen/vqcwork/chem8/base/dfthcsos/submission.npz')
    pr = Problem(z['tree_edges'])
    c = np.load(sys.argv[1])
    theta = torch.tensor(c['theta'], device='cuda').double()
    W = torch.tensor(c['W'], device='cuda').double()
    Ut = pr.tree64(theta)
    p = pr.pair64(Ut)
    X = torch.einsum('rbm,rbc->rcm', p, W)
    Y = torch.einsum('rbc,rcm->rbm', W, X)
    yy = (Y * Y).sum(-1).flatten().cpu().numpy()
    U = Ut.reshape(-1, N).cpu().numpy()
    base_edges = np.asarray(z['tree_edges'])
    n0, _ = rounding_noise(U, yy, base_edges)
    print(f'log-depth tree: naive rounding noise {1e3 * np.sqrt(n0):.2f} mHa (rms, fixed W, first order)')
    A = np.abs(U)
    for name, S in (('|u| co-occurrence', (A * yy[:, None]).T @ A),
                    ('u^2 co-occurrence', ((A ** 2) * yy[:, None]).T @ (A ** 2)),
                    ('|u| unweighted', A.T @ A)):
        e = dendrogram_tree(S)
        n, _ = rounding_noise(U, yy, e)
        print(f'{name:20s}: noise {1e3 * np.sqrt(n):.2f} mHa  (ratio {n / n0:.3f})')
        np.save(f'/home/omen/vqcwork/chem8/tree_{name.split()[0].replace("|", "a").replace("^", "")}.npy', e)
