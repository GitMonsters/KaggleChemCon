"""Convert the public 10-bit FeMoCo factorization into a trainable factorization `.npz`.

Source: Kaggle dataset dmitriigluzdov/vqcc-femoco-public-factorization (v2, Apache-2.0),
`femoco_public_words.npz`: `words` (3976, 75) 10-bit angle words on the log-depth tree,
`W` (65, 60, 30), `H2_I` (65, 30). U is the decoded two-body rows (rows 76..), which the
encoder rounds back to exactly the same 10-bit words.

    python notebooks/dfthc-refit/public_seed.py [OUT.npz]
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import encode  # noqa: E402
from paths import DATA, SHIFTS, WORK  # noqa: E402


def find_public() -> Path:
    local = DATA / 'public' / 'femoco-10bit' / 'femoco_public_words.npz'
    if local.is_file():
        return local
    hits = glob.glob('/kaggle/input/**/femoco_public_words.npz', recursive=True)
    if not hits:
        raise FileNotFoundError('femoco_public_words.npz not found; attach dmitriigluzdov/vqcc-femoco-public-factorization')
    return Path(hits[0])


# ahmedberatozer/df-thc-group-compression-factorization (Apache-2.0), 10-bit words.
GROUP_FILES = {'group32': 'dfthc_group_32_60_30.npz', 'group26': 'dfthc_group_v2_26_60_30.npz',
               'group24': 'dfthc_group_v3_24_60_30.npz'}


def find_group(source: str) -> Path:
    """`source` is 'group32' (32, 60, 30), 'group26' (26, 60, 30, version 2) or 'group24' (24, 60, 30, version 3)."""
    name = GROUP_FILES[source]
    local = DATA / 'public' / source / name
    if local.is_file():
        return local
    hits = glob.glob(f'/kaggle/input/**/{name}', recursive=True)
    if not hits:
        raise FileNotFoundError(f'{name} not found; attach ahmedberatozer/df-thc-group-compression-factorization')
    return Path(hits[0])


def convert(out: Path, source: str = 'public') -> Path:
    """`source` is 'public' (65, 60, 30) or a group-compression version: 'group32', 'group26', 'group24'."""
    with np.load(find_public() if source == 'public' else find_group(source), allow_pickle=False) as z:
        words = np.asarray(z['words' if 'words' in z.files else 'angle_words'], np.int64)
        W, H2_I = np.array(z['W']), np.array(z['H2_I'])
        edges = np.asarray(z['tree_edges'], np.int64) if 'tree_edges' in z.files else encode.log_depth_tree()
    if not np.array_equal(edges, encode.log_depth_tree()):
        raise ValueError('expected the log-depth Givens tree')
    R, B, _ = W.shape
    U = encode.decode(words[encode.N:], edges, 10).reshape(R, B, encode.N)
    with np.load(SHIFTS) as z:
        shifts = {k: np.array(z[k]) for k in ('H1_I', 'Hsym_lower')}
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, U=U, W=W, H2_I=H2_I, **shifts)
    return out


if __name__ == '__main__':
    source = sys.argv[2] if len(sys.argv) > 2 else 'public'
    out = convert(Path(sys.argv[1]) if len(sys.argv) > 1 else WORK / f'{source}_10bit.npz', source)
    edges = encode.log_depth_tree()
    for brot in (10, 11, 12):
        m = encode.metrics(out, edges, brot)
        print(f"{out.name} brot {brot}: error_1 {1e3 * m['error_1']:.2f}  error_2 {1e3 * m['error_2']:.2f} mHa  "
              f"Lambda {m['Lambda']:.2f}  E_gap {m['E_gap']:.4f}  lambda_eff {m['lambda_eff']:.3f}")
