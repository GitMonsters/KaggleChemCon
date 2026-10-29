"""Regression tests for verification-driven DFTHC-SOS compilation."""

from __future__ import annotations

import sys

import numpy as np
import pytest

sys.path.insert(0, "starter_kit")

from qcompile.prefix_compiler import delta_batch  # noqa: E402
from starter.task3_circuits.build_circuits import _validate_evaluation  # noqa: E402


def test_delta_batch_uses_zero_padded_adjacent_batches() -> None:
    words = np.arange(30, dtype=np.int64).reshape(3, 10)
    assert np.array_equal(delta_batch(words, 4, 0), words[:, :4])
    assert np.array_equal(delta_batch(words, 4, 1), words[:, 4:8] ^ words[:, :4])
    expected = words[:, 4:8].copy()
    expected[:, :2] ^= words[:, 8:10]
    assert np.array_equal(delta_batch(words, 4, 2), expected)


def _result(words: np.ndarray) -> dict:
    return {
        "brot": 12,
        "batch_size": 2,
        "angle_words": words,
        "metrics": {
            "error_1": 0.0,
            "error_2": 0.0,
            "gap_in_domain": True,
            "lambda_eff": 49.0,
            "Lambda": 100.0,
            "E_gap": 4.0,
            "Lambda_floor": 49.0,
            "E_gap_floor": 3.9,
        },
        "resources": {"angles_qubits": 128},
    }


def test_validate_evaluation_accepts_valid_result() -> None:
    words = np.array([[1, 2]])
    _validate_evaluation(_result(words), words, 12, 2)


def test_validate_evaluation_rejects_wrong_words() -> None:
    words = np.array([[1, 2]])
    result = _result(words + 1)
    with pytest.raises(ValueError, match="do not load"):
        _validate_evaluation(result, words, 12, 2)


def test_validate_evaluation_rejects_verifier_failure() -> None:
    words = np.array([[1, 2]])
    result = _result(words)
    result["metrics"]["error_2"] = 0.021
    with pytest.raises(ValueError, match="verifier rejected"):
        _validate_evaluation(result, words, 12, 2)
