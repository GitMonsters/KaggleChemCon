"""Assemble a submission npz from a base npz and a new two-body state (K signed words, W).

The new W is brought into the base's column gauge by orthogonal Procrustes per group (W_new ~ W_base O),
and the identity offsets H2_I are transported with the same rotation (coefficient transport).  One-body rows
and every other field are kept from the base.

Usage: python assemble.py <base npz> <state npz> <out npz>
"""
import sys

import numpy as np

sys.path.insert(0, '/mnt/c/Users/omen/Desktop/quantumAI/paradigm/chem')
import q8lib as L

base, state, out = sys.argv[1], sys.argv[2], sys.argv[3]
z = L.load_submission(base)
s = np.load(state)
K = np.asarray(s['K'], dtype=np.int64)
Wn = np.asarray(s['W'], dtype=np.float64)
R, B, C = Wn.shape
brot = int(z['brot'])
Wb = z['W']
Hb = z['H2_I']
if Wb.shape == Wn.shape:
    Wout = np.empty_like(Wn)
    Hout = np.empty_like(Hb)
    for r in range(R):
        # minimize ||Wn O - Wb||: O = U V^T from SVD of Wn^T Wb
        Uo, _, Vt = np.linalg.svd(Wn[r].T @ Wb[r])
        O = Uo @ Vt
        Wout[r] = Wn[r] @ O
        Hout[r] = Hb[r]                       # offsets stay attached to the (aligned) columns
    rel = np.linalg.norm(Wout - Wb) / np.linalg.norm(Wb)
    print(f'procrustes alignment: relative W change {rel:.4f}')
elif Wb.shape[0] == R and Wb.shape[2] == C:
    Wout, Hout = Wn, Hb.copy()
    print('vector count changed: columns kept in place, base offsets reused')
else:
    Wout, Hout = Wn, np.zeros((R, C))
    print('shape change: offsets reset to zero (must be re-optimized)')
words = np.array(z['angle_words'], dtype=np.int64)
if words.shape[0] != 76 + R * B:
    words = np.concatenate([words[:76], np.zeros((R * B, 75), dtype=np.int64)])
words[76:] = np.mod(K.reshape(-1, 75), 1 << brot)
import torch
tree = L.Tree(z['tree_edges'], 'cpu', torch.float64)
U = tree(torch.tensor(L.theta_of_words(words, brot))).numpy()
outd = dict(z)
outd['angle_words'] = words
outd['U'] = U[76:].reshape(R, B, 76)
outd['continuous_U'] = outd['U']
outd['W'] = Wout
outd['H2_I'] = Hout
outd['offset_W'] = Hout - Wout.sum(axis=1)
if 'kept_indices' in outd and outd['kept_indices'].shape != (R, B):
    outd['kept_indices'] = np.tile(np.arange(B), (R, 1))
np.savez(out, **outd)
print('wrote', out)
