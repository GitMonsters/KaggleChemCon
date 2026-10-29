"""Scores for Tasks 1 and 2, plus the score formula ``verify`` uses for Task 3.

Which function to call
----------------------
============================  ===============================================================
After Task 1 (factorization)  ``score(system, factorization)``
After Task 2 (angle words)    ``score(system, factorization, angle_words, tree_edges, brot)``
After Task 3 (.kmx files)     ``verifier.verify.verify(system, circuits, factorization)``
============================  ===============================================================

``factorization`` is an ``.npz`` file (or a mapping of arrays) with the
trainer's ``U``, ``W``, ``H2_I``, ``H1_I`` and ``Hsym_lower``; pickles are not
accepted (see ``verifier.reference``).

Each returns a float: lower is better, ``numpy.inf`` means rejected. Tasks 1
and 2 have no circuits, so their score is the ``lambda_eff`` factor alone;
compare it only with other Task 1/2 scores.

Formula (Task 3)
----------------
::

    score      = lambda_eff * T_total * Q_total
    lambda_eff = sqrt(E_gap * (2 Lambda - E_gap))       of the words the circuits load
    T_total    = ANGLES Toffolis                        mean per shot in kickmix, U_T and U_T^dagger
               + outer_state_prep + inner_state_prep    Toffolis, SpinFreeData_V1 (incl. _dag)
               + 2 * (N - 1) * (2 * brot - 1)           Toffolis, Givens rotations (U_T and U_T^dagger)
    Q_total    = outer_persistent + inner_persistent    persistent state-prep qubits, SpinFreeData_V1
               + J * brot                               persistent angle-word registers
               + max(outer_temp, inner_temp, ang_temp)  shared temporary workspace

The ANGLE_k circuits run sequentially and share their workspace with the outer
and inner state preparations, so temporary workspace is the maximum across
stages, not the sum. ``SpinFreeData_V1`` uses ``R, B, C, N``,
``chi = KEEP_BITS`` and ``beta = brot``.

Rejected (``inf``) if
---------------------
* ``error_1`` or ``error_2`` exceeds 20 mHa (defined in ``evaluate_inputs``);
* the SOS gap is outside ``(0, 2 * Lambda)``;
* Task 3 only: any ANGLE_k circuit uses more than 128 qubits.

Command line (Task 1): ``python verifier/score.py SYSTEM FACTORIZATION.npz``.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

import numpy as np
import numpy.typing as npt

# Development shim: allow `python verifier/score.py` without installing.
_REPOSITORY_ROOT = str(Path(__file__).resolve().parents[1])
if _REPOSITORY_ROOT not in sys.path:
    sys.path.insert(0, _REPOSITORY_ROOT)

# The verifier is self-contained: it imports nothing from dfthc (checked by
# dfthc/test_verifier_isolation.py) and needs only numpy, scipy and kickmix.
from verifier.circuits import (  # noqa: E402
    SHOTS_PER_ADDRESS,
    initial_seed,
    load_circuits,
    recover_words_from_circuits,
    register_layout,
    submission_sha256,
)
from verifier.costs import SpinFreeData_V1  # noqa: E402
from verifier.dfthc_math import (  # noqa: E402
    build_h1_prime,
    evaluate_dfthc_angle_words,
    evaluate_dfthc_vectors,
    reconstruct_x,
    symmetric_from_lower,
)
from verifier.reference import (  # noqa: E402
    DATA_DIR_ENVIRONMENT_VARIABLE,
    FIXED_SHIFT_SYSTEM,
    fcidump_path,
    load_factorization,
    load_fixed_shifts,
    load_reference,
    load_systems,
    system_entry,
)

# Qaggle Eq. (48): 20 mHa Frobenius-norm bound on the two-body residual. The
# same bound is applied to the one-body residual after quantization.
ERROR_2_THRESHOLD = 0.020
ERROR_1_THRESHOLD = 0.020

# Strict acceptance floors of the competition system on raw Lambda and the
# SOS gap (keep in sync with LAMBDA_FLOOR and E_GAP_FLOOR in dfthcsos/verify.py).
# ``evaluate_inputs`` records a system's floors in its metrics as
# ``Lambda_floor`` and ``E_gap_floor``, so every scoring path (Task 1, 2
# and 3) applies them; other systems have none.
ACCEPTANCE_FLOORS = {FIXED_SHIFT_SYSTEM: {"Lambda_floor": 49.0, "E_gap_floor": 3.914898}}

# Hard limit on the ANGLE_k circuit width: if any ANGLE_k circuit uses more
# qubits than this, the score is inf.
QUBIT_CAP = 128

# Keep-value bits (chi) of the outer and inner alias-sampling state
# preparations in the heuristic cost model.
KEEP_BITS = 10


def native_inputs(
    params: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Convert optimizer coordinates to the native Qaggle (w, W, S, beta_1).

    The factorization stores lowercase w as ``W``, ``w_B = W + sum_b w`` as
    ``H2_I``, ``S`` through its lower triangle ``Hsym_lower``, and
    ``beta_1 + Tr(S)/2`` as ``H1_I``. ``S`` is rounded to float32 first, as
    the trainer's ``H1_from_lower`` does.
    """
    w = np.asarray(params["W"], dtype=np.float64)
    W = np.asarray(params["H2_I"], dtype=np.float64) - np.sum(w, axis=1)
    S = np.asarray(
        symmetric_from_lower(params["Hsym_lower"], np.shape(params["U"])[-1]),
        dtype=np.float64,
    )
    beta_1 = float(params["H1_I"]) - 0.5 * float(np.trace(S))
    return w, W, S, beta_1


def symmetric_lower(array: np.ndarray) -> np.ndarray:
    """Symmetrize from the lower triangle (the verifier's X convention)."""
    return np.tril(array) + np.swapaxes(np.tril(array, k=-1), -1, -2)


@dataclass(frozen=True)
class SubmissionInputs:
    """A factorization together with its system's reference data.

    ``ref`` is ``verifier.reference.load_reference`` of the system's FCIDUMP:
    ``{orbitals, electrons, ecore, oei, tei, h1_thc}``. ``factorization_file``
    is the ``.npz`` the arrays came from, or ``None`` for in-memory arrays.
    """

    system: str
    factorization_file: Path | None
    params: dict[str, np.ndarray]
    ref: dict[str, Any]
    w: np.ndarray
    W: np.ndarray
    S: np.ndarray
    beta_1: float
    energy_upper: float

    @property
    def num_orbitals(self) -> int:
        return int(self.ref["orbitals"])

    def h1_prime(self, X: np.ndarray) -> np.ndarray:
        """h1' of Eq. (38) for the given (symmetrized) X."""
        return build_h1_prime(
            self.ref["h1_thc"], self.S, self.beta_1,
            self.ref["electrons"], self.W, X,
        )


def load_inputs(
    system: str,
    factorization: str | Path | Mapping[str, npt.ArrayLike],
    *,
    data_dir: str | Path | None = None,
) -> SubmissionInputs:
    """Load a factorization and its system's FCIDUMP and E_upper.

    ``factorization`` is an ``.npz`` path or a mapping of arrays
    (``verifier.reference.load_factorization``). ``data_dir`` holds
    ``systems.toml`` and ``integrals/`` (default: ``$DFTHCSOS_DATA_DIR``, else
    ``<repository>/data``). For FeMoCo76o113e the symmetry shift is the
    competition's fixed one (``verifier.reference.load_fixed_shifts``): the
    submitted ``H1_I`` and ``Hsym_lower`` are ignored, as ``dfthcsos/verify.py`` does.
    """
    entry = system_entry(system, data_dir)
    fcidump = fcidump_path(system, data_dir)
    if not fcidump.is_file():
        raise FileNotFoundError(f"FCIDUMP not installed: {fcidump}")
    params = load_factorization(factorization)
    if system == FIXED_SHIFT_SYSTEM:
        params.update(load_fixed_shifts())
    ref = load_reference(fcidump)
    num_orbitals = int(ref["orbitals"])
    if np.shape(params["U"])[-1] != num_orbitals:
        raise ValueError(
            f"factorization has {np.shape(params['U'])[-1]} orbitals, "
            f"{system} has {num_orbitals}"
        )
    w, W, S, beta_1 = native_inputs(params)
    return SubmissionInputs(
        system=system,
        factorization_file=None if isinstance(factorization, Mapping) else Path(factorization),
        params=params,
        ref=ref,
        w=w,
        W=W,
        S=S,
        beta_1=beta_1,
        energy_upper=float(entry["upper_energy"]),
    )


def evaluate_inputs(
    inputs: SubmissionInputs,
    angle_words: npt.ArrayLike | None = None,
    tree_edges: npt.ArrayLike | None = None,
    brot: int | None = None,
) -> dict[str, float | bool]:
    """Run the competition verifier on a loaded factorization.

    What is evaluated
    -----------------
    The verifier (``verifier.dfthc_math.evaluate_dfthc_vectors``) takes a table
    of ``N + R*B`` unit vectors of length ``N``:

    * rows ``0 .. N-1``: the one-body vectors, meant to be the eigenvectors of
      ``h1_prime`` (below);
    * rows ``N .. N+R*B-1``: the two-body vectors ``u_b^(rho)`` in
      ``(rho, b)`` order.

    It also takes the scalars and tensors ``w`` (R, B, C), ``W`` (R, C),
    ``S`` (N, N) and ``beta_1`` from the factorization (see ``native_inputs``),
    and the original FCIDUMP integrals ``oei = h^(1)`` and ``tei = (pq|rs)``.

    There are two modes:

    * **Continuous** (``angle_words is None``): the two-body rows are the
      normalized ``U`` rows of the factorization, and the one-body rows are the
      exact eigenvectors of ``h1_prime``. Nothing is rounded, so this measures
      the factorization itself (Part 1).
    * **Finite-bit** (``angle_words``, ``tree_edges`` and ``brot`` given):
      each row of the ``(N + R*B, N - 1)`` integer table is decoded on the
      Givens tree, with ``theta / pi = word / 2**(brot - 1)``, into a unit
      vector. Those decoded vectors are evaluated instead (Part 2). Rounding
      and circuit simulation happen before this call.

    Intermediate quantities (from the decoded two-body vectors)
    -----------------------------------------------------------
    ::

        X^(rho c)_pq = sum_b w_b^(rho c) u_b^(rho)_p u_b^(rho)_q
                       (symmetrized from the lower triangle)
        H1_THC       = oei_pq - 1/2 sum_r (pr|rq)
        h1_prime     = H1_THC - (eta/2) S + beta_1 I - sum_(rho c) W^(rho c) X^(rho c)
                       (qaggle.tex Eq. exact-one-body-absorption)
        lambda_j     = eigenvalues of h1_prime

    ``h1_prime`` is always built from the *decoded* ``X``, so rounding the
    two-body angles changes ``h1_prime`` (and hence ``lambda_j``) consistently.
    It never shows up as a one-body error.

    Returned metrics (all energies in Hartree)
    ------------------------------------------
    ``error_1``
        One-body residual, ``||H1_THC(submitted) - H1_THC||_F``. The verifier
        pairs each submitted one-body row ``v_j`` with an exact eigenvector of
        ``h1_prime`` (a one-to-one assignment on ``|overlap|``, so row order and
        signs do not matter). It gives that row the matching eigenvalue
        ``lambda_j`` and rebuilds ``sum_j lambda_j v_j v_j^T``. Undoing the
        shifts gives ``H1_THC(submitted)``. The shifts cancel, so this equals
        ``||sum_j lambda_j v_j v_j^T - h1_prime||_F``: how well the one-body
        rows encode ``h1_prime``. It is ~1e-15 in continuous mode and shrinks
        with ``brot`` in finite-bit mode. It is compared with the 20 mHa bound
        (``ERROR_1_THRESHOLD``). This is the "residual of the 1-electron
        component" in qaggle.tex.
    ``error_2``
        Two-body residual ``epsilon_fro``, the Frobenius norm over all ``N**4``
        chemist-order entries::

            || sum_(rho c) X^(rho c) (x) X^(rho c) - (S (x) I + I (x) S)/2 - (pq|rs) ||_F

        This is compared with the 20 mHa bound (``ERROR_2_THRESHOLD``). In
        finite-bit mode it includes the rounding error of the two-body rows.
    ``Lambda``
        LCU one-norm (qaggle.tex Eq. dfthc-normalization)::

            sum_j |lambda_j| + 1/4 sum_(rho c) (|w_B^(rho c)| + sum_b |w_b^(rho c)|)**2,
            w_B^(rho c) = W^(rho c) + sum_b w_b^(rho c)
    ``E_SOS``
        Sum-of-squares lower bound on the DFTHC Hamiltonian's energy,
        including the FCIDUMP core energy::

            ecore - eta beta_1 - (sum_j |lambda_j| - sum_j lambda_j) - 1/2 sum_(rho c) W^(rho c)**2
    ``E_gap``
        ``E_upper - E_SOS``, where ``E_upper`` is the reference energy in
        ``data/systems.toml``. It stands in for the ground-state energy, so
        this is an estimate; the many-body Hamiltonian is not diagonalized.
    ``gap_in_domain``
        ``0 < E_gap < 2 * Lambda``, including the high-precision radicand check.
    ``lambda_eff``
        Spectrum-amplified one-norm ``sqrt(E_gap * (2 Lambda - E_gap))``
        (qaggle.tex Eq. effective-normalization). It is ``nan`` when the gap is
        out of domain. It is the first factor of the score.

    ``score_from_metrics`` turns these into the score: ``inf`` if
    ``error_1`` or ``error_2`` exceeds 20 mHa, the gap is out of domain, or
    ``Lambda`` or ``E_gap`` is at or below the system's acceptance floor
    (``ACCEPTANCE_FLOORS``, recorded in the metrics as ``Lambda_floor``
    and ``E_gap_floor``), otherwise ``lambda_eff`` times the Toffoli and qubit
    counts (see ``resource_costs``), or ``lambda_eff`` alone when there are no circuits.
    """
    floors = ACCEPTANCE_FLOORS.get(inputs.system, {})
    if angle_words is not None:
        if tree_edges is None or brot is None:
            raise ValueError("angle_words require tree_edges and brot")
        # Decode each word row on tree_edges into a unit vector, then evaluate.
        metrics = evaluate_dfthc_angle_words(
            angle_words, inputs.w, inputs.W, inputs.S, inputs.beta_1, inputs.ref,
            brot=brot, tree_edges=tree_edges, energy_upper=inputs.energy_upper,
        )
        return {**metrics, **floors}
    if tree_edges is not None or brot is not None:
        raise ValueError("tree_edges and brot are only used with angle_words")

    # Continuous vectors: normalized U rows for the two-body part, and the
    # exact eigenvectors of h1_prime for the one-body part. h1_prime is built
    # from the same X the verifier uses (symmetric lower-triangle convention),
    # so error_1 is at round-off level here.
    raw_u = np.asarray(inputs.params["U"], dtype=np.float64)
    u = raw_u / np.linalg.norm(raw_u, axis=-1, keepdims=True)
    X = symmetric_lower(reconstruct_x(u, inputs.w))
    _, eigenvector_columns = np.linalg.eigh(inputs.h1_prime(X))
    u_all = np.vstack([eigenvector_columns.T, u.reshape(-1, inputs.num_orbitals)])
    metrics = evaluate_dfthc_vectors(
        u_all, inputs.w, inputs.W, inputs.S, inputs.beta_1, inputs.ref,
        energy_upper=inputs.energy_upper,
    )
    return {**metrics, **floors}


def evaluate_submission(
    system: str,
    factorization: str | Path | Mapping[str, npt.ArrayLike],
    angle_words: npt.ArrayLike | None = None,
    tree_edges: npt.ArrayLike | None = None,
    brot: int | None = None,
    *,
    data_dir: str | Path | None = None,
) -> dict[str, float | bool]:
    """Return the verifier metrics for a submission (see ``score``)."""
    return evaluate_inputs(
        load_inputs(system, factorization, data_dir=data_dir), angle_words, tree_edges, brot
    )


def _floor_rejection_reasons(metrics: Mapping[str, Any]) -> list[str]:
    """Reject nonfinite values and equality at the system's strict floors."""
    reasons = []
    for field in ("Lambda", "E_gap"):
        floor_key = f"{field}_floor"
        if floor_key not in metrics:
            continue
        value = metrics.get(field, np.nan)
        floor = metrics[floor_key]
        if not (np.isfinite(value) and np.isfinite(floor) and value > floor):
            reasons.append(f"{field} = {value:.17g} must be finite and > {floor}")
    return reasons


def passes_accuracy(metrics: Mapping[str, Any]) -> bool:
    """True if ``error_1`` and ``error_2`` are within 20 mHa, the gap is valid and any floors hold.

    The strict floors are the metrics' own ``Lambda_floor`` and ``E_gap_floor``
    entries (set by ``evaluate_inputs`` from ``ACCEPTANCE_FLOORS``); metrics
    without them have no floors.
    """
    return bool(
        metrics["error_1"] <= ERROR_1_THRESHOLD
        and metrics["error_2"] <= ERROR_2_THRESHOLD
        and metrics["gap_in_domain"]
        and np.isfinite(metrics["lambda_eff"])
        and not _floor_rejection_reasons(metrics)
    )


def angles_costs(circuits: Iterable[Any], toffoli: float) -> dict[str, int | float]:
    """Toffoli and qubit counts of an ANGLE_k family of kickmix circuits.

    ``toffoli`` is the simulator's count from ``recover_words_from_circuits``:
    the mean number of Toffoli-like gates executed per shot over the forward
    (``U_T``) and reverse (``U_T^dagger``) pass; ``toffoli_per_pass`` is half of
    it. ``qubits`` is the maximum over k of ``Circuit.num_qubits`` (workspace
    included). In the block encoding, ``selection`` is the address register from
    the state preparations, the ``J`` word registers (``J * brot`` qubits) stay
    live across the Givens rotations (``persistent_qubits``), and the remaining
    workspace qubits return to ``|0>`` after each circuit (``temporary_qubits``).
    """
    circuits = tuple(circuits)
    if not circuits:
        raise ValueError("circuits must be nonempty")
    layout = register_layout(circuits)
    qubits = max(int(circuit.num_qubits) for circuit in circuits)
    selection_qubits = len(layout.selection)
    persistent_qubits = layout.batch_size * layout.brot
    temporary_qubits = qubits - selection_qubits - persistent_qubits
    return {
        "toffoli_per_pass": toffoli / 2,
        "toffoli": toffoli,
        "qubits": qubits,
        "selection_qubits": selection_qubits,
        "persistent_qubits": persistent_qubits,
        "temporary_qubits": temporary_qubits,
    }


def heuristic_costs(
    num_orbitals: int,
    rank_r: int,
    bases_b: int,
    ensemble_c: int,
    brot: int,
    keep_bits: int = KEEP_BITS,
) -> dict[str, int]:
    """State-preparation and Givens-rotation costs from ``SpinFreeData_V1`` (``verifier.costs``).

    Toffoli counts are ``outer_state_prep()``, ``inner_state_prep()`` and
    ``givens_rotations()`` (one block-encoding pass ``O_x``: each state prep
    already includes its uncomputation, and the ``N - 1`` Givens rotations run
    in ``U_T`` and ``U_T^dagger``). Qubit counts share temporary workspace
    between the sequential outer and inner state preparations:
    ``persistent_qubits + max(outer_temporary, inner_temporary)``. ``chi = keep_bits``
    and ``beta = brot``.
    """
    model = SpinFreeData_V1(
        R=int(rank_r),
        B=int(bases_b),
        C=int(ensemble_c),
        N=int(num_orbitals),
        chi=int(keep_bits),
        beta=int(brot),
    )
    outer_qubits = model.outer_state_prep_qubit_cost()
    inner_qubits = model.inner_state_prep_qubit_cost()
    outer_persistent = int(outer_qubits["persistent"])
    outer_temporary = int(outer_qubits["temporary"])
    inner_persistent = int(inner_qubits["persistent"])
    inner_temporary = int(inner_qubits["temporary"])
    costs = {
        "outer_state_prep_toffoli": int(model.outer_state_prep()),
        "inner_state_prep_toffoli": int(model.inner_state_prep()),
        "givens_rotations_toffoli": int(model.givens_rotations()),
        "outer_state_prep_persistent_qubits": outer_persistent,
        "outer_state_prep_temporary_qubits": outer_temporary,
        "outer_state_prep_qubits": outer_persistent + outer_temporary,
        "inner_state_prep_persistent_qubits": inner_persistent,
        "inner_state_prep_temporary_qubits": inner_temporary,
        "inner_state_prep_qubits": inner_persistent + inner_temporary,
    }
    costs["persistent_qubits"] = outer_persistent + inner_persistent
    costs["temporary_qubits"] = max(outer_temporary, inner_temporary)
    costs["toffoli"] = (
        costs["outer_state_prep_toffoli"]
        + costs["inner_state_prep_toffoli"]
        + costs["givens_rotations_toffoli"]
    )
    costs["qubits"] = costs["persistent_qubits"] + costs["temporary_qubits"]
    return costs


def resource_costs(
    inputs: SubmissionInputs,
    circuits: Iterable[Any],
    brot: int,
    angles_toffoli: float,
    keep_bits: int = KEEP_BITS,
) -> dict[str, int | float | bool]:
    """``T_total`` and ``Q_total`` of the score (see the module docstring).

    ``angles_toffoli`` is the simulator's Toffoli count of the ANGLES family
    (``recover_words_from_circuits``).

    Returns the ANGLES and heuristic parts, the shared-workspace qubit breakdown
    (``persistent_qubits`` and ``temporary_qubits``), their totals ``toffoli``
    and ``qubits``, and ``angles_within_cap`` (``angles_qubits <= QUBIT_CAP``).
    """
    rank_r, bases_b, ensemble_c = inputs.w.shape
    angles = angles_costs(circuits, angles_toffoli)
    heuristic = heuristic_costs(
        inputs.num_orbitals, rank_r, bases_b, ensemble_c, brot, keep_bits
    )
    toffoli = angles["toffoli"] + heuristic["toffoli"]
    persistent_qubits = heuristic["persistent_qubits"] + angles["persistent_qubits"]
    temporary_qubits = max(heuristic["temporary_qubits"], angles["temporary_qubits"])
    qubits = persistent_qubits + temporary_qubits
    return {
        "brot": int(brot),
        "keep_bits": int(keep_bits),
        "angles_toffoli_per_pass": angles["toffoli_per_pass"],
        "angles_toffoli": angles["toffoli"],
        "angles_qubits": angles["qubits"],
        "angles_selection_qubits": angles["selection_qubits"],
        "angles_persistent_qubits": angles["persistent_qubits"],
        "angles_temporary_qubits": angles["temporary_qubits"],
        **{
            name: value
            for name, value in heuristic.items()
            if name.startswith(("outer", "inner", "givens"))
        },
        "heuristic_persistent_qubits": heuristic["persistent_qubits"],
        "heuristic_temporary_qubits": heuristic["temporary_qubits"],
        "heuristic_toffoli": heuristic["toffoli"],
        "heuristic_qubits": heuristic["qubits"],
        "persistent_qubits": persistent_qubits,
        "temporary_qubits": temporary_qubits,
        "toffoli": toffoli,
        "qubits": qubits,
        "angles_within_cap": angles["qubits"] <= QUBIT_CAP,
    }


def rejection_reason(
    metrics: Mapping[str, Any], resources: Mapping[str, Any] | None = None
) -> str | None:
    """Why ``score_from_metrics`` returns ``inf``, or ``None`` if it does not."""
    reasons = []
    if not metrics["error_1"] <= ERROR_1_THRESHOLD:
        reasons.append(
            f"error_1 = {1e3 * metrics['error_1']:.4f} mHa > {1e3 * ERROR_1_THRESHOLD:g} mHa"
        )
    if not metrics["error_2"] <= ERROR_2_THRESHOLD:
        reasons.append(
            f"error_2 = {1e3 * metrics['error_2']:.4f} mHa > {1e3 * ERROR_2_THRESHOLD:g} mHa"
        )
    if not metrics["gap_in_domain"] or not np.isfinite(metrics["lambda_eff"]):
        reasons.append("the SOS gap E_gap lies outside (0, 2 * Lambda) or its calculation is invalid")
    reasons.extend(_floor_rejection_reasons(metrics))
    if resources is not None and int(resources["angles_qubits"]) > QUBIT_CAP:
        reasons.append(
            f"an ANGLE_k circuit uses {resources['angles_qubits']} qubits > {QUBIT_CAP}"
        )
    return "; ".join(reasons) or None


def score_from_metrics(
    metrics: Mapping[str, Any], resources: Mapping[str, Any] | None = None
) -> float:
    """``inf`` if rejected, else ``lambda_eff * toffoli * qubits``.

    Rejected means: an accuracy gate fails (``passes_accuracy``), or the
    ANGLES circuits use more than ``QUBIT_CAP`` qubits. Without ``resources``
    (no circuits) only the ``lambda_eff`` factor is returned, with the same
    accuracy gates. ``rejection_reason`` says why a result is ``inf``.
    """
    if not passes_accuracy(metrics):
        return np.inf
    lambda_eff = float(metrics["lambda_eff"])
    if resources is None:
        return lambda_eff
    if int(resources["angles_qubits"]) > QUBIT_CAP:
        return np.inf
    return lambda_eff * float(resources["toffoli"]) * int(resources["qubits"])


def evaluate_circuits(
    inputs: SubmissionInputs,
    circuits: Any,
    tree_edges: npt.ArrayLike,
    keep_bits: int = KEEP_BITS,
    *,
    format: str = "auto",
    shots_per_address: int = SHOTS_PER_ADDRESS,
    seed: int | None = None,
    salt: str = "",
) -> dict[str, Any]:
    """Simulate an ANGLE_k family, evaluate the words it loads and score it.

    ``circuits`` is anything ``verifier.circuits.load_circuits`` accepts: a
    directory holding ``angle_k_0.kmx``, ``angle_k_1.kmx``, ... (or ``.kmb``), a
    single circuit file, or an ordered sequence of circuit paths or
    ``kickmix.Circuit`` objects. ``recover_words_from_circuits`` checks the
    register contract and recovers the words; ``brot`` and ``J`` are the width
    and the number of the ``word_j`` registers. Raises
    ``verifier.circuits.InvalidCircuits`` if the circuits break the contract.
    The simulator seed is ``seed``, or ``initial_seed(submission_sha256(...), salt)``.

    Returns ``angle_words``, ``brot``, ``batch_size``, ``metrics``,
    ``resources``, ``score``, ``submission_sha256`` and ``verified_shots``.
    """
    circuits = load_circuits(circuits, format=format)
    digest = submission_sha256(circuits, inputs.factorization_file or inputs.params, tree_edges)
    rank_r, bases_b, _ = inputs.w.shape
    num_orbitals = inputs.num_orbitals
    num_vectors = num_orbitals + rank_r * bases_b
    words, layout, toffoli = recover_words_from_circuits(
        circuits,
        num_vectors=num_vectors,
        num_angles=num_orbitals - 1,
        shots_per_address=shots_per_address,
        seed=initial_seed(digest, salt) if seed is None else seed,
    )
    brot = layout.brot
    metrics = evaluate_inputs(inputs, words, tree_edges, brot)
    resources = resource_costs(inputs, circuits, brot, toffoli, keep_bits)
    return {
        "angle_words": words,
        "brot": brot,
        "batch_size": layout.batch_size,
        "metrics": metrics,
        "resources": resources,
        "score": score_from_metrics(metrics, resources),
        "submission_sha256": digest,
        "verified_shots": num_vectors * shots_per_address,
    }


def score(
    system: str,
    factorization: str | Path | Mapping[str, npt.ArrayLike],
    angle_words: npt.ArrayLike | None = None,
    tree_edges: npt.ArrayLike | None = None,
    brot: int | None = None,
    *,
    data_dir: str | Path | None = None,
) -> float:
    """Score a Task 1 or Task 2 result: the ``lambda_eff`` factor, or ``inf``.

    * Task 1: ``score(system, factorization)`` scores the unrounded
      factorization (an ``.npz`` path or a mapping of arrays).
    * Task 2: ``score(system, factorization, angle_words, tree_edges, brot)``
      scores the angle words decoded on ``tree_edges`` at ``brot`` bits.

    ``inf`` if ``error_1`` or ``error_2`` exceeds 20 mHa or the gap is out of
    range. For Task 3 (circuits) use ``verifier.verify.verify``.
    """
    inputs = load_inputs(system, factorization, data_dir=data_dir)
    return score_from_metrics(evaluate_inputs(inputs, angle_words, tree_edges, brot))


__all__ = [
    "ACCEPTANCE_FLOORS",
    "ERROR_1_THRESHOLD",
    "ERROR_2_THRESHOLD",
    "KEEP_BITS",
    "QUBIT_CAP",
    "SubmissionInputs",
    "angles_costs",
    "evaluate_circuits",
    "evaluate_inputs",
    "evaluate_submission",
    "heuristic_costs",
    "load_inputs",
    "native_inputs",
    "passes_accuracy",
    "rejection_reason",
    "resource_costs",
    "score",
    "score_from_metrics",
    "symmetric_lower",
]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Circuit-free score (lambda_eff factor) of a factorization. "
        "For .kmx circuits use verifier/verify.py."
    )
    parser.add_argument("system", help="system name from systems.toml")
    parser.add_argument("factorization", type=Path, help="factorization .npz")
    parser.add_argument(
        "--data-dir", type=Path,
        help="directory with systems.toml and integrals/ "
        f"(default: ${DATA_DIR_ENVIRONMENT_VARIABLE}, else <repository>/data)",
    )
    args = parser.parse_args()
    known = load_systems(args.data_dir)
    if args.system not in known:
        parser.error(f"unknown system {args.system!r}; choose from {', '.join(sorted(known))}")
    value = score(args.system, args.factorization, data_dir=args.data_dir)
    print(f"{args.system}: {args.factorization.name}: lambda_eff factor = {value}")
