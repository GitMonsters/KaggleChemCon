"""Try offset initializations for a re-shaped dictionary; write each variant and print official metrics.

Usage: python offset_init.py <assembled npz (one-body refreshed)> <base npz> <out prefix>
"""
import json
import sys

import numpy as np

sys.path[:0] = ['/home/omen/vqcwork/comp/verifiers', '/home/omen/vqcwork/comp/kickmix']
from dfthcsos import verify as V

src, base, prefix = sys.argv[1], sys.argv[2], sys.argv[3]
z = dict(np.load(src))
b = dict(np.load(base))
w = z['W']
sw = w.sum(axis=1)
C = w.shape[2]
variants = {'keepH2I': z['H2_I']}
if b['W'].shape[2] == C:
    variants['keepOffset'] = b['offset_W'] + sw
variants['offset0'] = sw.copy()
variants['H2I0'] = np.zeros_like(sw)
variants['halfway'] = 0.5 * sw
best = None
for name, h2 in variants.items():
    d = dict(z)
    d['H2_I'] = h2
    d['offset_W'] = h2 - sw
    path = f'{prefix}_{name}.npz'
    np.savez(path, **d)
    inp = V.load_inputs(path)
    m = V.evaluate_words(inp, d['angle_words'], d['tree_edges'], int(d['brot']))
    row = {k: float(m[k]) for k in ('error_1', 'error_2', 'Lambda', 'E_gap', 'lambda_eff')}
    print(name, json.dumps(row), flush=True)
    if np.isfinite(row['lambda_eff']) and (best is None or row['lambda_eff'] < best[1]):
        best = (name, row['lambda_eff'])
print('best', best)
