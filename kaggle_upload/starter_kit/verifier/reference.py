"""Verifier inputs: the system registry, reference integrals and factorizations.

Nothing here imports ``dfthc``. The FCIDUMP reader reproduces
``dfthc.load_integrals.fci_from_fcidump`` value for value, and the
factorization loader replaces ``dfthc.dfthc_u_to_tei.load_solution``. It reads
``.npz`` archives only, never pickles, because ``pickle.load`` on a submitted
file would run arbitrary code.

Data directory
--------------
``data_dir`` holds ``systems.toml`` and ``integrals/<fcidump>``. It is, in
order of precedence: the ``data_dir`` argument, the ``$DFTHCSOS_DATA_DIR``
environment variable, or ``data/`` next to the directory that holds this
package (``<repository>/data`` in a checkout). If the data directory has no
``systems.toml``, a copy packaged next to this module is used when there is
one, so a directory holding only ``integrals/`` is enough.

Factorization ``.npz``
----------------------
The arrays of the trainer's ``params_best`` (``U`` (R, B, N), ``W`` (R, B, C),
``H2_I`` (R, C), ``H1_I`` (scalar), ``Hsym_lower`` (N(N+1)/2,)); other keys
are ignored. Write one with ``numpy.savez(path, U=..., W=..., H2_I=...,
H1_I=..., Hsym_lower=...)``; the starter trainer writes NPZ directly.
No legacy pickle reader or converter is provided.
"""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import io
import os
from pathlib import Path
import tomllib
from typing import Any
import zipfile

import numpy as np
import numpy.typing as npt

DATA_DIR_ENVIRONMENT_VARIABLE = "DFTHCSOS_DATA_DIR"
DEFAULT_DATA_DIRECTORY = Path(__file__).resolve().parents[1] / "data"
SYSTEMS_FILE = "systems.toml"
# A registry shipped with the package (absent in a plain checkout, where the
# registry lives in the data directory).
PACKAGED_SYSTEMS_FILE = Path(__file__).resolve().with_name(SYSTEMS_FILE)
INTEGRALS_SUBDIRECTORY = "integrals"
FACTORIZATION_KEYS = ("U", "W", "H2_I", "H1_I", "Hsym_lower")

# The competition system's data, packaged one level above this dfthc-sos tree
# (the ``dfthcsos`` package; see ``dfthcsos/scoreinfra/reference.py``). It is
# the registry when no ``systems.toml`` is installed, and the source of the
# fixed symmetry shift ``(S, beta_1)`` the competition scores against.
FIXED_SHIFT_SYSTEM = "FeMoCo76o113e"
PACKAGE_DIRECTORY = Path(__file__).resolve().parents[2]
PACKAGED_FCIDUMP = PACKAGE_DIRECTORY / f"{FIXED_SHIFT_SYSTEM}.fcidump"
PACKAGED_FIXED_SHIFTS = PACKAGE_DIRECTORY / f"{FIXED_SHIFT_SYSTEM}.fixed_shifts.npz"
FIXED_SHIFTS_SHA256 = "7b7facf63fe35a9d9d3d00787ffdb700c863fa845fe4724d7f7c54778151cd87"
FIXED_SHIFTS_KEYS = ("H1_I", "Hsym_lower")
PACKAGED_BASELINE = (
    Path(__file__).resolve().parents[1] / "starter" / "task1_factorization" / "FeMoco76_65_60_60.npz"
)
PACKAGED_SYSTEMS = {
    FIXED_SHIFT_SYSTEM: {
        "fcidump": str(PACKAGED_FCIDUMP),
        "baseline": str(PACKAGED_BASELINE),
        "upper_energy": -22140.27335,
        "rank": 65,
        "bases": 60,
        "copies": 60,
    }
}


# ---------------------------------------------------------------------------
# System registry
# ---------------------------------------------------------------------------


def data_directory(data_dir: str | Path | None = None) -> Path:
    """Resolve the data directory (argument, then environment, then default)."""
    if data_dir is None:
        data_dir = os.environ.get(DATA_DIR_ENVIRONMENT_VARIABLE) or DEFAULT_DATA_DIRECTORY
    return Path(data_dir)


def systems_file(data_dir: str | Path | None = None) -> Path | None:
    """``systems.toml`` of the data directory, else the copy packaged with the module.

    ``None`` if neither exists: ``load_systems`` then falls back to
    ``PACKAGED_SYSTEMS``.
    """
    path = data_directory(data_dir) / SYSTEMS_FILE
    if path.is_file():
        return path
    if PACKAGED_SYSTEMS_FILE.is_file():
        return PACKAGED_SYSTEMS_FILE
    return None


def load_systems(data_dir: str | Path | None = None) -> dict[str, dict[str, Any]]:
    """The ``[systems.*]`` tables of ``systems.toml``, keyed by system name.

    Without a registry (no research data bundle), the packaged
    FeMoCo76o113e system alone, with absolute ``fcidump`` and ``baseline`` paths.
    """
    path = systems_file(data_dir)
    if path is None:
        return {name: dict(entry) for name, entry in PACKAGED_SYSTEMS.items()}
    with path.open("rb") as stream:
        return tomllib.load(stream)["systems"]


def system_entry(system: str, data_dir: str | Path | None = None) -> dict[str, Any]:
    """The registry entry of ``system``; ``KeyError`` for unknown systems.

    ``FileNotFoundError`` if ``system`` is not the packaged one and no
    ``systems.toml`` is installed.
    """
    systems = load_systems(data_dir)
    if system not in systems:
        if systems_file(data_dir) is None:
            raise FileNotFoundError(
                f"system registry not found: {data_directory(data_dir) / SYSTEMS_FILE} "
                f"(only the packaged {FIXED_SHIFT_SYSTEM} is available; pass data_dir or set "
                f"${DATA_DIR_ENVIRONMENT_VARIABLE})"
            )
        raise KeyError(f"unknown system {system!r}; known systems: {', '.join(sorted(systems))}")
    return systems[system]


def fcidump_path(system: str, data_dir: str | Path | None = None) -> Path:
    """Resolve the reference, using the packaged FeMoCo integral by default.

    The shipped registry does not require a second copy of that integral in
    ``src/data/integrals``. Explicit ``data_dir`` and environment overrides
    keep their own paths, including when their integral has not been installed.
    """
    entry = system_entry(system, data_dir)
    path = data_directory(data_dir) / INTEGRALS_SUBDIRECTORY / entry["fcidump"]
    if (
        system == FIXED_SHIFT_SYSTEM
        and data_dir is None
        and not os.environ.get(DATA_DIR_ENVIRONMENT_VARIABLE)
        and not path.is_file()
        and PACKAGED_FCIDUMP.is_file()
    ):
        return PACKAGED_FCIDUMP
    return path


def load_fixed_shifts(path: str | Path = PACKAGED_FIXED_SHIFTS) -> dict[str, np.ndarray]:
    """The FeMoCo76o113e baseline's ``H1_I`` and ``Hsym_lower``, after checking the file's SHA-256.

    The competition verifier (``dfthcsos/verify.py``) scores every
    FeMoCo76o113e submission with this symmetry shift and ignores the
    submitted ``H1_I`` and ``Hsym_lower``; ``score.load_inputs`` does the same.
    """
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != FIXED_SHIFTS_SHA256:
        raise ValueError(
            f"{path} is not the {FIXED_SHIFT_SYSTEM} fixed shifts file (sha256 {FIXED_SHIFTS_SHA256})"
        )
    with np.load(io.BytesIO(raw), allow_pickle=False) as data:
        return {key: np.array(data[key]) for key in FIXED_SHIFTS_KEYS}


# ---------------------------------------------------------------------------
# Reference integrals
# ---------------------------------------------------------------------------


def _pair_index(i: int, j: int) -> int:
    """Row-major index of the pair (max(i, j), min(i, j)) in a packed triangle."""
    high, low = max(i, j), min(i, j)
    return high * (high + 1) // 2 + low


def _parse_header(header: str) -> dict[str, str]:
    """Key-value pairs of an ``&FCI ... &END`` header (``FCIDumpFileContent``)."""
    rest_header = header[4:-4]
    values = {}
    while (ind := rest_header.find("=")) > -1:
        key = rest_header[0:ind].strip()
        rest_header = rest_header[ind + 1:]
        # The value runs to the next comma, or to the next space for ORBSYM.
        ind = rest_header.find("," if key != "ORBSYM" else " ")
        value = rest_header[0:ind].strip()
        rest_header = rest_header[ind + 1:]
        if key == "ORBSYM":
            value = value[:-1]
        values[key] = value
    return values


def _h2_full_from_lower(h2_lower: np.ndarray, num_orbitals: int) -> np.ndarray:
    """Expand packed-pair ``(pq|rs)`` to the full chemist-order N^4 tensor."""
    n = num_orbitals
    rows, cols = np.tril_indices(n)
    sym = np.zeros((n * (n + 1) // 2, n, n))
    sym[:, rows, cols] = np.array(h2_lower)
    sym[:, cols, rows] = np.array(h2_lower)
    h2_full = np.zeros((n, n, n, n))
    h2_full[rows, cols, :, :] = sym
    h2_full[cols, rows, :, :] = sym
    return h2_full


def load_reference(fcidump: str | Path) -> dict[str, Any]:
    """Read an FCIDUMP into the reference used by the verifier.

    Returns ``orbitals`` (N), ``electrons`` (eta), ``ecore``, ``oei`` (N, N),
    ``tei`` (N, N, N, N) in chemist order and ``h1_thc``
    (``oei - 1/2 sum_r (pr|rq)``).

    Like ``dfthc.load_integrals``, the two-electron integrals are stored as
    float32 before they are expanded, so ``tei`` holds float32-rounded values
    (``oei`` is float64). The score depends on this rounding; keep it.
    """
    path = Path(fcidump)
    with path.open("r") as stream:
        first = stream.readline().strip()
        if not first.startswith("&FCI"):
            raise ValueError(f"{path} is not an FCIDUMP file: it does not start with &FCI")
        header = ""
        line = first
        while True:
            header += line + " "
            if line.endswith("&END"):
                break
            line = stream.readline()
            if not line:
                raise ValueError(f"{path}: the FCIDUMP header has no &END")
            line = line.strip()
        values = _parse_header(header)
        num_orbitals = int(values["NORB"])
        num_electrons = int(values["NELEC"])

        num_pairs = num_orbitals * (num_orbitals + 1) // 2
        oei = np.zeros((num_orbitals, num_orbitals))
        h2_lower = np.zeros((num_pairs, num_pairs), dtype=np.float32)
        ecore = 0.0
        for line in stream:
            tokens = line.split()
            if not tokens:
                continue
            coefficient = float(tokens[0])
            indices = [int(token) - 1 for token in tokens[1:] if token != "0"]
            if len(indices) == 2:
                i, j = sorted(indices)
                oei[i, j] = coefficient
                oei[j, i] = coefficient
            elif len(indices) == 4:
                # Mulliken (ij|kl) to Dirac order, then the canonical member of
                # the eightfold permutation group.
                i, j, k, l = indices[0], indices[2], indices[3], indices[1]
                i, j, k, l = min(
                    [i, j, k, l], [j, i, l, k], [k, l, i, j], [l, k, j, i],
                    [i, k, j, l], [k, i, l, j], [j, l, i, k], [l, j, k, i],
                )
                ii, jj = _pair_index(i, l), _pair_index(j, k)
                h2_lower[ii, jj] = coefficient
                h2_lower[jj, ii] = coefficient
            elif not indices:
                ecore = coefficient

    tei = _h2_full_from_lower(h2_lower, num_orbitals)
    h1_thc = np.array(oei - 0.5 * np.einsum("pqqs->ps", tei))
    return {
        "orbitals": num_orbitals,
        "electrons": num_electrons,
        "ecore": ecore,
        "oei": oei,
        "tei": tei,
        "h1_thc": h1_thc,
    }


# ---------------------------------------------------------------------------
# Factorizations
# ---------------------------------------------------------------------------


def load_factorization(
    factorization: str | Path | Mapping[str, npt.ArrayLike],
) -> dict[str, np.ndarray]:
    """Read and validate a factorization: an ``.npz`` path or a mapping of arrays.

    Returns the five arrays of ``FACTORIZATION_KEYS`` with their stored
    dtypes. ``.npz`` files are opened with ``allow_pickle=False``.
    """
    if isinstance(factorization, (str, os.PathLike)):
        path = Path(factorization)
        if path.suffix.lower() != ".npz":
            raise ValueError(
                f"{path}: factorizations require .npz files, not pickles or other formats; save the "
                f"arrays {FACTORIZATION_KEYS} with numpy.savez (the starter trainer "
                "writes NPZ directly)"
            )
        if not path.is_file():
            raise FileNotFoundError(f"no such factorization file: {path}")
        if zipfile.is_zipfile(path):
            with zipfile.ZipFile(path) as zf:
                names = zf.namelist()
            if len(names) != len(set(names)):
                raise ValueError(f"{path} contains duplicate archive entries")
        try:
            loaded = np.load(path, allow_pickle=False)
        except ValueError as error:
            raise ValueError(f"{path} is not an .npz archive of arrays: {error}") from error
        if isinstance(loaded, np.ndarray):
            raise ValueError(f"{path} holds a single array; expected an .npz with {FACTORIZATION_KEYS}")
        with loaded as archive:
            missing = [key for key in FACTORIZATION_KEYS if key not in archive.files]
            if missing:
                raise KeyError(f"{path} has no {missing} entries")
            params = {key: np.array(archive[key]) for key in FACTORIZATION_KEYS}
        source = str(path)
    elif isinstance(factorization, Mapping):
        missing = [key for key in FACTORIZATION_KEYS if key not in factorization]
        if missing:
            raise KeyError(f"factorization has no {missing} entries")
        params = {key: np.array(factorization[key]) for key in FACTORIZATION_KEYS}
        source = "factorization"
    else:
        raise TypeError("factorization must be an .npz path or a mapping of arrays")

    for key, value in params.items():
        if not (np.issubdtype(value.dtype, np.floating) or np.issubdtype(value.dtype, np.integer)):
            raise ValueError(f"{source}: {key} must be a real numeric array, got {value.dtype}")
        if not np.all(np.isfinite(value)):
            raise ValueError(f"{source}: {key} must contain finite values")
    U, W = params["U"], params["W"]
    if U.ndim != 3 or W.ndim != 3:
        raise ValueError(f"{source}: U and W must have shapes (R, B, N) and (R, B, C)")
    if U.shape[:2] != W.shape[:2]:
        raise ValueError(f"{source}: U and W must have the same (R, B) leading dimensions")
    rank_r, bases_b, copies_c = W.shape
    if min(rank_r, bases_b, copies_c, U.shape[-1]) <= 0:
        raise ValueError(f"{source}: factorization dimensions must be positive")
    if params["H2_I"].shape != (rank_r, copies_c):
        raise ValueError(f"{source}: H2_I must have shape (R, C) = {(rank_r, copies_c)}")
    if params["H1_I"].size != 1:
        raise ValueError(f"{source}: H1_I must be a scalar")
    params["H1_I"] = params["H1_I"].reshape(())
    orbitals = U.shape[-1]
    if params["Hsym_lower"].shape != (orbitals * (orbitals + 1) // 2,):
        raise ValueError(f"{source}: Hsym_lower has the wrong packed-pair dimension")
    return params


__all__ = [
    "DATA_DIR_ENVIRONMENT_VARIABLE",
    "DEFAULT_DATA_DIRECTORY",
    "FACTORIZATION_KEYS",
    "FIXED_SHIFT_SYSTEM",
    "FIXED_SHIFTS_KEYS",
    "FIXED_SHIFTS_SHA256",
    "PACKAGED_FIXED_SHIFTS",
    "PACKAGED_SYSTEMS",
    "PACKAGED_SYSTEMS_FILE",
    "data_directory",
    "fcidump_path",
    "load_factorization",
    "load_fixed_shifts",
    "load_reference",
    "load_systems",
    "system_entry",
    "systems_file",
]
