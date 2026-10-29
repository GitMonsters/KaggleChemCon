"""J=4 ANGLE family for 1366-address (R10/B129) tables with mrmeows' streamed exact lookup compiler
(cost-first package, Apache-2.0: balanced_plan + streamed_plan + streamed_native), replanned for new words with a
seed search; every circuit is audited symbolically and checked natively on all addresses.

Usage: python emit_r10.py <npz or words .npy> <out dir> [jobs] [seeds]
"""
from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

SRC = Path('/home/omen/vqcwork/m2100/sources/dfthcsos_cost_first_shape_screen/source')
sys.path[:0] = [str(SRC), '/home/omen/vqcwork/comp/kickmix']


def build_one(args):
    k, words, out, seeds = args
    import kickmix as km
    import rank_parity as core
    from balanced_plan import plan_balanced
    from streamed_plan import plan_tensor
    from streamed_native import build_streamed
    from audit_affine_hmr import audit
    from native_checks import check
    t0 = time.time()
    delta = core.delta_batch(words, 4, k)
    ac = core.coefficients(delta, 8, 4)
    best = None
    for s in range(seeds):
        left, _ = plan_balanced(ac[1:, 1:12, :], group_size=3, nrelations=38, seed=s, trials=16, plane_mode='pairs')
        if left is None:
            continue
        for s2 in range(max(1, seeds // 4)):
            right, _ = plan_tensor(ac[1:, 12:16, :], group_size=4, nrelations=12, seed=s2, trials=24, plane_mode='original')
            if right is None:
                continue
            circuit, stats = build_streamed(delta, ac, [left, right])
            if best is None or stats['T'] < best[1]['T']:
                best = (circuit, stats, s, s2)
            break
    if best is None:
        return {'k': k, 'error': 'no exact plan'}
    circuit, stats, s, s2 = best
    dest = Path(out) / f'angle_k_{k}.kmb'
    circuit.to_file(dest, format='kmb')
    circuit = km.Circuit.from_file(dest, format='kmb')
    proof = audit(str(circuit), delta, circuit.num_qubits)
    native = check(circuit, delta, brot=8, repeats=4, seed=652209 + k)
    assert native == stats['T'], (native, stats['T'])
    return {'k': k, 'T': int(stats['T']), 'Q': int(circuit.num_qubits), 'seeds': [s, s2],
            'proof_ok': bool(proof), 'seconds': round(time.time() - t0, 1)}


def main():
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    jobs = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    seeds = int(sys.argv[4]) if len(sys.argv) > 4 else 4
    words = np.load(src)['angle_words'] if src.suffix == '.npz' else np.load(src)
    words = np.asarray(words, dtype=np.int64)
    out.mkdir(parents=True, exist_ok=True)
    nb = (words.shape[1] + 3) // 4
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        reports = list(pool.map(build_one, [(k, words, str(out), seeds) for k in range(nb)]))
    total = sum(r.get('T', 0) for r in reports)
    (out / 'plans.json').write_text(json.dumps({'rows': int(words.shape[0]), 'per_pass': total, 'circuits': reports}, indent=1))
    print(json.dumps({'rows': int(words.shape[0]), 'per_pass': total, 'T_both': 2 * total,
                      'per_circuit': [r.get('T') for r in reports], 'errors': [r for r in reports if 'error' in r]}))


if __name__ == '__main__':
    main()
