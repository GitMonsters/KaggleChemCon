"""Fast checks of the pipeline's exact steps (numpy and kickmix only, ~30 s):

    .venv/bin/python -m pytest notebooks/dfthc-refit/tests
"""
from __future__ import annotations

import sys
from itertools import permutations
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import encode  # noqa: E402
import fourqrom  # noqa: E402
import onebody  # noqa: E402
import tensorqrom  # noqa: E402
from refine import rotate_columns  # noqa: E402
from structure import merge, two_body  # noqa: E402


def random_factorization(R=6, B=5, C=3, N=8, seed=0):
    rng = np.random.default_rng(seed)
    w = rng.normal(size=(R, B, C))
    return {'U': rng.normal(size=(R, B, N)), 'W': w, 'H2_I': w.sum(axis=1) + rng.normal(size=(R, C)),
            'H1_I': np.zeros(()), 'Hsym_lower': np.zeros(1)}


def signed_table(A: int, seed: int = 0, J: int = 3, brot: int = 10) -> np.ndarray:
    """A random (A, J) table of brot-bit words in the signed gauge (top bit = the one below)."""
    q = 1 << (brot - 2)
    return np.random.default_rng(seed).integers(-q, q, size=(A, J)) & ((1 << brot) - 1)


@pytest.mark.parametrize('k', [2, 3])
def test_merge_is_exact(k):
    p = random_factorization()
    q = merge(p, k, C=k * 3)
    assert q['W'].shape == (-(-6 // k), k * 5, k * 3)
    assert np.allclose(two_body(q['U'], q['W']), two_body(p['U'], p['W']), rtol=1e-12, atol=1e-12)
    # The identity offsets W = H2_I - sum_b w carry over copy by copy.
    assert np.allclose(np.sort((q['H2_I'] - q['W'].sum(axis=1)).ravel()),
                       np.sort((p['H2_I'] - p['W'].sum(axis=1)).ravel()))


def test_merge_pads_extra_copies_small():
    p = random_factorization()
    q = merge(p, 2, C=8)                       # 2 * 3 = 6 merged copies, padded to 8
    assert q['W'].shape[2] == 8
    assert np.abs(q['W'][:, :, 6:]).max() < 1e-2


@pytest.mark.parametrize('load', ['anf', 'unary'])
@pytest.mark.parametrize('A', [36, 64, 132, 516, 1276, 2048])
def test_four_address_circuit(A, load):
    """XOR the right words on every address, keep the selection, leave no phase or workspace."""
    delta = signed_table(A, seed=A)
    circuit = fourqrom.build_circuit(delta, max(1, (A - 1).bit_length()), load=load)
    assert circuit.num_qubits <= 128
    fourqrom.check(circuit, delta, repeats=2, seed=A)


def test_four_address_cost():
    """Regression guard on the measured cost: 376 Toffolis per pass at 1,276 rows (rows/4 + 57)."""
    delta = signed_table(1276)
    assert fourqrom.check(fourqrom.build_circuit(delta, 11), delta, repeats=1) <= 376


@pytest.mark.parametrize('n', range(1, 9))
def test_block_path_is_optimal(n):
    def cost(order):
        return sum(1 if bin(p ^ q).count('1') == 1 else 3 for p, q in zip(order, order[1:]))
    order = fourqrom._block_path(n)
    assert sorted(order) == list(range(n))
    if n <= 7:
        assert cost(order) == min(cost(o) for o in permutations(range(n)))


def test_signed_gauge_round_trip():
    rng = np.random.default_rng(1)
    v = rng.normal(size=(50, encode.N))
    edges = encode.log_depth_tree()
    words = encode.signed_words(v, edges, 10)
    assert not np.any(((words >> 9) ^ (words >> 8)) & 1), 'bit 9 must equal bit 8'
    back = encode.decode(words, edges, 10)
    unit = v / np.linalg.norm(v, axis=1, keepdims=True)
    assert np.min(np.abs(np.sum(back * unit, axis=1))) > 0.99     # equal up to sign and rounding


def test_rotate_columns_keeps_the_tensor():
    p = random_factorization(R=3, B=6, C=5)
    w, W = p['W'], p['H2_I'] - p['W'].sum(axis=1)
    A0 = np.concatenate([w, (W + w.sum(axis=1))[:, None]], axis=1)
    w2, W2 = rotate_columns(w, W, sweeps=2, seed=0)
    A1 = np.concatenate([w2, (W2 + w2.sum(axis=1))[:, None]], axis=1)
    assert np.allclose(A1 @ A1.swapaxes(1, 2), A0 @ A0.swapaxes(1, 2))
    assert np.sum(np.abs(A1).sum(axis=1) ** 2) <= np.sum(np.abs(A0).sum(axis=1) ** 2) + 1e-12


def test_refit_error_formula_and_gradient():
    """refit_weights: error_2 by the K/Q formula equals the direct tensor norm, and its gradient is exact."""
    from refit_weights import WeightProblem, error_terms
    p = random_factorization(R=3, B=4, C=2, N=5, seed=3)
    u = p['U'] / np.linalg.norm(p['U'], axis=2, keepdims=True)
    rng = np.random.default_rng(4)
    tei = rng.normal(size=(5,) * 4)
    tei = tei + tei.transpose(1, 0, 2, 3) + tei.transpose(0, 1, 3, 2) + tei.transpose(2, 3, 0, 1)
    S = rng.normal(size=(5, 5))
    S = S + S.T
    K, Q, T2 = error_terms(u, tei, S)
    wp = WeightProblem(u, K, Q, T2, np.diag(np.arange(5.0)) - 2.0, 1.0)
    w, W = p['W'], rng.normal(size=(3, 2))
    target = tei.reshape(25, 25) + 0.5 * (np.outer(S.ravel(), np.eye(5).ravel()) + np.outer(np.eye(5).ravel(), S.ravel()))
    direct = np.linalg.norm(two_body(u, w) - target)
    (e2, _, _), (de2, _, _) = wp.evaluate(w, W)
    assert np.isclose(e2, direct, rtol=1e-10)
    d = rng.normal(size=w.shape)
    fd = (wp.error_2(w + 1e-6 * d)[0] - wp.error_2(w - 1e-6 * d)[0]) / 2e-6
    assert np.isclose(np.sum(de2 * d), fd, rtol=1e-5)


def test_polish_moves_are_exact():
    """polish_words: each move's predicted change of error_2^2 equals a full recomputation."""
    from polish_words import Polisher
    n, R, B, C = 8, 2, 3, 2
    edges = encode.log_depth_tree(n)
    rng = np.random.default_rng(5)
    words = rng.integers(-200, 200, size=(R * B, n - 1)) % 1024
    T = rng.normal(size=(n * n, n * n))
    T = T + T.T
    p = Polisher(words, rng.normal(size=(R, B, C)), T, edges)
    for b in range(R * B):
        m = p.best_move(b)
        if m is None:
            continue
        before = p.error2()
        p.apply(b, *m[1:])
        full = Polisher(p.k % 1024, p.w, T, edges).error2()
        assert np.isclose(full - before, m[0], rtol=1e-8, atol=1e-12)
        assert m[0] < 0
        assert np.allclose(p.G, p.M @ p.K @ p.M)


@pytest.mark.parametrize('brot', [8, 10])
@pytest.mark.parametrize('A,J,lowbits', [(100, 2, 3), (300, 3, 4), (1096, 4, 4), (700, 2, 6)])
def test_tensor_circuit(A, J, lowbits, brot):
    """The product-cache circuit XORs the right words everywhere and its count matches the model."""
    delta = signed_table(A, J=J, brot=brot, seed=A + J)
    C = tensorqrom.anf_tensor(delta, brot, lowbits)
    terms = tensorqrom.shared(C, [5] * J * brot, transpose=lowbits <= 4, trials=2)
    circuit = tensorqrom.build_circuit(delta, terms, max(1, (A - 1).bit_length()), brot, lowbits)
    assert circuit.num_qubits <= 128
    assert fourqrom.check(circuit, delta, repeats=2, seed=A, brot=brot) == tensorqrom.cost(terms, *C.shape[:2])


def test_tensor_cost():
    """Regression guard: at most 300 Toffolis for five 8-bit words over 1,096 rows."""
    delta = signed_table(1096, J=5, brot=8)
    C = tensorqrom.anf_tensor(delta, 8, 4)
    assert tensorqrom.cost(tensorqrom.shared(C, list(tensorqrom.SIZES)), *C.shape[:2]) <= 300


def test_onebody_signed_gauge_is_exact():
    rng = np.random.default_rng(2)
    edges = encode.log_depth_tree()
    words = rng.integers(0, 256, size=(40, encode.N - 1))
    g = onebody.signed_gauge(words, edges, 8)
    assert not np.any(((g >> 7) ^ (g >> 6)) & 1)
    a, b = encode.decode(words, edges, 8), encode.decode(g, edges, 8)
    assert np.allclose(np.abs(np.sum(a * b, axis=1)), 1, atol=1e-12)
