"""Turn a factorization into a dfthcsos submission directory and score it with the official verifier.

    python notebooks/dfthc-refit/build.py FACTORIZATION.npz OUT_DIR [--brot 12] [--tree log_depth] [--shots 1]
        [--qrom four] [--json RESULT.json]

`J` (words per circuit) is the most that fits under the 128-qubit cap.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import encode  # noqa: E402  (also puts data/verifiers on sys.path)
from dfthcsos.example_submission.angle_k_reference_circuit import build_angle_k_family, write_kmx_family  # noqa: E402
from dfthcsos.verify import QUBIT_CAP, load_inputs, verify  # noqa: E402

TREES = {'log_depth': encode.log_depth_tree, 'path': encode.path_tree}
BASELINE_SCORE = 3355599588.3155365  # the sample submission's dfthcsos cost
KEYS = ('U', 'W', 'H2_I', 'H1_I', 'Hsym_lower')


def max_batch(rows: int, brot: int) -> int:
    sel = math.ceil(math.log2(rows))
    return min((QUBIT_CAP - 2 * sel + 1) // brot, encode.N - 1)


def build(factorization: Path, out: Path, brot: int, tree: str, qrom: str = 'reference') -> Path:
    """Encode (or, if the npz carries `angle_words` and `tree_edges`, reuse) the words; write circuits."""
    with np.load(factorization) as z:
        arrays = {k: np.array(z[k]) for k in KEYS}
        given = 'angle_words' in z.files
        if given:  # refine.py output: words already quantized, gauge-fixed and one-body refreshed
            words, edges = np.asarray(z['angle_words'], np.int64), np.asarray(z['tree_edges'], np.int64)
            brot = int(z['brot']) if 'brot' in z.files else brot
    if not given:
        edges = TREES[tree]()
        words = encode.encode(load_inputs(factorization), edges, brot)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    np.savez(out / 'submission.npz', **arrays, tree_edges=edges, angle_words=words)
    if qrom == 'four':  # fourqrom.py: three words per circuit, one lookup step per four rows
        import fourqrom
        family = fourqrom.build_family(words, brot)
    elif qrom == 'tensor':  # tensorqrom.py: cached address monomials, five words per circuit
        import tensorqrom
        family = tensorqrom.build_family(words, brot)
    else:
        family = build_angle_k_family(words, brot, max_batch(len(words), brot))
    write_kmx_family(family, out, format='kmb')
    return out


def score(factorization: Path, out: Path, brot: int = 12, tree: str = 'log_depth', shots: int = 1,
          qrom: str = 'reference') -> dict:
    """Build the submission directory and verify it; a rejection is returned as `error`."""
    out = build(factorization, out, brot, tree, qrom)
    try:
        r = verify(out, shots_per_address=shots)
    except AssertionError as e:
        return {'brot': brot, 'error': str(e)}
    m, res = r['metrics'], r['resources']
    return {
        'score': r['score'], 'improvement': BASELINE_SCORE / r['score'],
        'lambda_eff': r['lambda_eff'], 'toffoli': r['toffoli_count'], 'qubits': r['num_qubits'],
        'brot': r['brot'], 'J': r['batch_size'], 'K': r['num_circuits'],
        'error_1_mHa': 1e3 * m['error_1'], 'error_2_mHa': 1e3 * m['error_2'],
        'Lambda': m['Lambda'], 'E_gap': m['E_gap'], 'angles_toffoli': res['angles_toffoli'],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('factorization', type=Path)
    ap.add_argument('out', type=Path)
    ap.add_argument('--brot', type=int, default=12)
    ap.add_argument('--tree', choices=sorted(TREES), default='log_depth')
    ap.add_argument('--shots', type=int, default=1, help='shots per address (official default 64)')
    ap.add_argument('--qrom', choices=('reference', 'four', 'tensor'), default='reference')
    ap.add_argument('--json', type=Path, help='also write the result here (only if it was accepted)')
    a = ap.parse_args()
    result = score(a.factorization, a.out, a.brot, a.tree, a.shots, a.qrom)
    print(json.dumps(result, indent=1))
    if 'error' in result:
        sys.exit(f"rejected: {result['error']}")
    if a.json:
        a.json.write_text(json.dumps(result, indent=1) + '\n')


if __name__ == '__main__':
    main()
