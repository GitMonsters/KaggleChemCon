"""Quantization sensitivity of a Givens tree: d(v) = depth of orbital v in the binary split tree.

For an edge k=(p->q) the derivative norm |du/dtheta_k| equals the amplitude still at p when edge k is
processed, so s_b = sum_k |du_b/dtheta_k|^2 = sum_v u_bv^2 d(v) with d(v) the number of rotations whose
remaining mass includes v.  Weighted by the two-body impact ||y_b||^2 this is the rounding-noise proxy.
"""
import heapq
import sys
import numpy as np
import torch
sys.path.insert(0, '/mnt/c/Users/omen/Desktop/quantumAI/paradigm/chem')
import q8lib as L

N = 76


def depth_of_tree(edges):
    """d(v) for the official parent-before-child edge order (children ordered as they appear)."""
    children = {v: [] for v in range(N)}
    for p, q in edges:
        children[int(p)].append(int(q))
    d = np.zeros(N)

    def walk(v, acc):
        kids = children[v]
        d[v] = acc + len(kids)
        for j, w in enumerate(kids):
            walk(w, acc + j + 1)
    walk(int(edges[0][0]), 0)
    return d


def huffman_tree(m, root_pref=None):
    """Binary split tree minimizing sum m_v depth(v); realized as Givens edges (parent-before-child)."""
    heap = [(float(m[v]), v, ('leaf', v)) for v in range(N)]
    heapq.heapify(heap)
    cnt = N
    while len(heap) > 1:
        a = heapq.heappop(heap)
        b = heapq.heappop(heap)
        heapq.heappush(heap, (a[0] + b[0], cnt, ('node', a[2], b[2])))
        cnt += 1
    root = heap[0][2]
    # Realize: each subtree has a representative orbital = its heaviest leaf; a split (L, R) at
    # representative p (in the heavier side R) emits edge p -> rep(L) and recurses.
    def leaves(t):
        return [t[1]] if t[0] == 'leaf' else leaves(t[1]) + leaves(t[2])

    def rep(t):
        ls = leaves(t)
        return max(ls, key=lambda v: m[v])
    edges = []

    def emit(t, p):
        # t's mass currently sits at orbital p (p is a leaf of t)
        if t[0] == 'leaf':
            return
        A, Bt = t[1], t[2]
        if p in leaves(A):
            A, Bt = Bt, A          # Bt holds p; A splits off
        q = rep(A)
        edges.append((p, q))
        emit(A, q)
        emit(Bt, p)
    r = rep(root)
    emit(root, r)
    return np.array(edges, dtype=np.int64), r


def relabel_root(edges, r):
    return edges  # tree root is edges[0][0]; the verifier needs root 0? (checked separately)


if __name__ == '__main__':
    z = L.load_submission(sys.argv[1])
    R, B, C = z['W'].shape
    U = torch.tensor(z['U'], device='cuda')
    w = torch.tensor(z['W'], device='cuda')
    tb = L.TwoBody()
    X = tb.X(U, w).reshape(R, C, -1)                       # (R, C, P)
    Y = torch.einsum('rbc,rcp->rbp', w, X)                 # y_b per vector
    imp = (Y * Y).sum(-1).cpu().numpy().reshape(-1)        # ||y_b||^2
    u2 = (z['U'].reshape(-1, N)) ** 2
    m = (imp[:, None] * u2).sum(0)
    d_cur = depth_of_tree(z['tree_edges'])
    print('current tree: weighted depth', float((m * d_cur).sum() / m.sum()), ' unweighted mean depth', d_cur.mean())
    print('orbital usage m (sorted, normalized) top10', np.round(np.sort(m)[::-1][:10] / m.sum(), 4), ' entropy bits', float(-(m / m.sum() * np.log2(m / m.sum())).sum()))
    e, r = huffman_tree(m)
    d_h = depth_of_tree(e)
    print('huffman tree root', r, 'weighted depth', float((m * d_h).sum() / m.sum()), 'max depth', d_h.max())
    # one-body: rows 0..75 are eigenvectors of h1'; weight by |eigenvalue| unknown here -> uniform
    one = None
    np.save('/home/omen/vqcwork/chem8/huff_edges.npy', e)
