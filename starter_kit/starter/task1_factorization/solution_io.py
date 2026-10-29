"""Safe NPZ storage for factorization parameters and training diagnostics.

The five scoring arrays retain their names and dtypes. Numeric diagnostics
use ``values__<name>`` keys; optional metadata is a Unicode JSON scalar.
This module never reads pickles, including files renamed to ``.npz``.
"""

from __future__ import annotations

from collections.abc import Mapping
import json
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np

from verifier.reference import FACTORIZATION_KEYS, load_factorization

BASELINES_DIRECTORY = Path(__file__).resolve().parents[2] / "baselines"
VALUES_PREFIX = "values__"
METADATA_KEY = "metadata_json"


def check_output_path(path: str | Path, *, overwrite: bool = False) -> Path:
    """Validate the output suffix and collision policy before doing work."""
    path = Path(path)
    if path.suffix.lower() != ".npz":
        raise ValueError(f"solution path must end in .npz, got {path}")
    if not overwrite and (path.exists() or path.is_symlink()):
        raise FileExistsError(f"refusing to overwrite existing solution: {path}")
    return path


def _numeric_array(value: Any, name: str) -> np.ndarray:
    array = np.array(value)
    if array.dtype.kind not in "biufc":
        raise ValueError(f"{name} must be a numeric or boolean array, got {array.dtype}")
    return array


def _field_name(name: Any) -> str:
    if not isinstance(name, str) or not name or "/" in name or "\\" in name:
        raise ValueError(f"invalid NPZ field name: {name!r}")
    return name


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"metadata value of type {type(value).__name__} is not JSON serializable")


def read_factorization(path: str | Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Read validated parameters and numeric diagnostics from an NPZ only."""
    path = Path(path)
    # This checks the extension, duplicate ZIP members, required shapes, real
    # finite parameters and rejects pickle payloads before diagnostics are read.
    params = load_factorization(path)
    with np.load(path, allow_pickle=False) as archive:
        values = {
            key[len(VALUES_PREFIX):]: _numeric_array(archive[key], key)
            for key in archive.files if key.startswith(VALUES_PREFIX)
        }
    return params, values


def save_solution_npz(
    path: str | Path,
    params: Mapping[str, Any],
    *,
    values: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
    extra_arrays: Mapping[str, Any] | None = None,
    overwrite: bool = False,
) -> Path:
    """Validate and atomically save a solution without storing Python objects.

    ``None`` diagnostic/extra fields are omitted. Existing files are protected
    by an atomic exclusive link unless ``overwrite=True`` is explicit. All
    validation happens before creating the output or its parent directory.
    """
    path = check_output_path(path, overwrite=overwrite)
    arrays = load_factorization(params)
    for name, value in (values or {}).items():
        name = _field_name(name)
        if value is not None:
            key = VALUES_PREFIX + name
            arrays[key] = _numeric_array(value, key)
    for name, value in (extra_arrays or {}).items():
        name = _field_name(name)
        if name in FACTORIZATION_KEYS or name == METADATA_KEY or name.startswith(VALUES_PREFIX):
            raise ValueError(f"extra array uses reserved NPZ field name: {name}")
        if value is not None:
            arrays[name] = _numeric_array(value, name)
    if metadata is not None:
        arrays[METADATA_KEY] = np.asarray(json.dumps(metadata, default=_json_default, allow_nan=False))

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            np.savez(stream, **arrays)
        if overwrite:
            os.replace(temporary, path)
        else:
            os.link(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return path


__all__ = ["BASELINES_DIRECTORY", "check_output_path", "read_factorization", "save_solution_npz"]
