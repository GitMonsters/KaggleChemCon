"""Part 3 baseline: translate Qualtran ANGLE_k QROMs into kickmix circuits.

The Part 3 task is to implement, for each batch ``k``, the Boolean function
that maps a vector address to that batch's angle bits::

    f_k : {0, ..., N + R*B - 1} -> {0, 1}^(J * brot),   f_k(a) = delta[k, a]

as a reversible XOR oracle ``|a>|z> -> |a>|z XOR f_k(a)>`` (see
``dfthc.build_angle_k`` for ``delta``). Any circuit with this action is
acceptable. This module shows one way to do it: build the baseline ordinary
QROMs with Qualtran (``build_angle_ks``), decompose them, and replay the gates
into a ``kickmix.CircuitBuilder``.

Register contract of the emitted circuits (kickmix format v2; the verifier's
contract, see ``verifier/circuits.py``):

* ``"selection"``: the address ``a``, little endian (``q0`` is the 1s bit),
  ``ceil(log2(N + R*B))`` qubits.
* ``"word_0"``, ..., ``"word_{J-1}"``: the ``J`` angle words of the batch, one
  ``brot``-qubit register per word, little endian. The verifier reads ``brot``
  and ``J`` from these registers.
* Any further qubits are workspace. They start and end in ``|0>``; measurement
  based uncomputation (``HMR`` plus a conditioned ``CZ``) leaves no phase.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
import sys

import cirq
import kickmix as km
import numpy as np
from qualtran.bloqs.basic_gates import CNOT, CZ, Toffoli, XGate, ZGate
from qualtran.bloqs.mcmt import And
from qualtran.cirq_interop import BloqAsCirqGate

# Development shim: the repository root holds dfthc/, verifier/ and starter/.
_REPOSITORY_ROOT = str(Path(__file__).resolve().parents[2])
if _REPOSITORY_ROOT not in sys.path:
    sys.path.insert(0, _REPOSITORY_ROOT)

from dfthc.build_angle_k import AngleK  # noqa: E402
from verifier.circuits import (  # noqa: E402
    SELECTION_REGISTER,
    kmx_path,
    read_kmx_family,
    word_register,
)


def _gate(operation: cirq.Operation):
    gate = operation.gate
    return gate.bloq if isinstance(gate, BloqAsCirqGate) else gate


def _instruction_name(gate) -> str | None:
    """Kickmix instruction for a decomposed gate, or None if unsupported."""
    for gate_type, name in (
        (XGate, "X"), (CNOT, "CX"), (Toffoli, "CCX"), (ZGate, "Z"), (CZ, "CZ"),
    ):
        if isinstance(gate, gate_type):
            return name
    for cirq_gate, name in (
        (cirq.X, "X"), (cirq.CNOT, "CX"), (cirq.TOFFOLI, "CCX"),
        (cirq.Z, "Z"), (cirq.CZ, "CZ"), (cirq.SWAP, "SWAP"),
    ):
        if gate == cirq_gate:
            return name
    return None


def _emit(builder: km.CircuitBuilder, gate, qubits: list[km.q]) -> None:
    """Append one decomposed Qualtran/Cirq gate to the kickmix builder."""
    if isinstance(gate, And):
        # Qualtran's And computes (or, with uncompute=True, erases) a clean
        # AND target; negative controls are conjugated by X.
        control_1, control_2, target = qubits
        negated = [
            q for q, value in zip((control_1, control_2), (gate.cv1, gate.cv2))
            if value == 0
        ]
        for q in negated:
            builder.x(q)
        if gate.uncompute:
            builder.del_and(control_1, control_2, target)  # HMR + CZ if b
        else:
            builder.init_and(control_1, control_2, target)  # R + CCX
        for q in negated:
            builder.x(q)
        return
    name = _instruction_name(gate)
    if name == "X":
        builder.x(qubits[0])
    elif name == "CX":
        builder.cx(qubits[0], qubits[1])
    elif name == "CCX":
        builder.ccx(qubits[0], qubits[1], qubits[2])
    elif name == "Z":
        builder.z(qubits[0])
    elif name == "CZ":
        builder.cz(qubits[0], qubits[1])
    elif name == "SWAP":
        builder.swap(qubits[0], qubits[1])
    else:
        raise ValueError(f"no kickmix translation for {gate!r}")


def angle_k_to_circuit(angle_k: AngleK) -> km.Circuit:
    """Translate one Qualtran ANGLE_k QROM into a kickmix ``Circuit``."""
    if not isinstance(angle_k, AngleK):
        raise TypeError("angle_k must be an instantiated AngleK")
    qrom = angle_k.qrom
    if qrom.num_controls != 0 or len(qrom.data) != 1:
        raise ValueError("ANGLE_k requires one uncontrolled QROM data table")
    data = np.asarray(qrom.data[0])
    if data.ndim != 2 or min(data.shape) == 0:
        raise ValueError("ANGLE_k data must have shape (num_vectors, J)")
    registers = {register.name: register for register in angle_k.signature}
    selection_bits = int(registers["a"].bitsize)
    brot = int(registers["z"].bitsize)
    _, batch_size = data.shape
    if brot < 1 or np.any(data < 0) or np.any(data >= 1 << brot):
        raise ValueError("QROM data must contain unsigned brot-bit words")

    # Decompose with Cirq line qubits standing in for the public registers.
    # Qualtran register arrays are big endian; kickmix registers are little
    # endian, so reverse the bit order of the selection and of each word.
    public = cirq.LineQubit.range(selection_bits + batch_size * brot)
    selection = np.asarray(public[:selection_bits][::-1], dtype=object)
    output = np.asarray(public[selection_bits:], dtype=object).reshape(
        batch_size, brot
    )[:, ::-1]
    context = cirq.DecompositionContext(cirq.SimpleQubitManager(prefix="work"))
    circuit, output_registers = angle_k.decompose_bloq().to_cirq_circuit_and_quregs(
        qubit_manager=context.qubit_manager, a=selection, z=output
    )
    if not (
        np.array_equal(output_registers["a"], selection)
        and np.array_equal(output_registers["z"], output)
    ):
        raise ValueError("ANGLE_k decomposition changed its public qubit layout")
    operations = cirq.decompose(
        circuit,
        context=context,
        keep=lambda op: isinstance(_gate(op), And)
        or _instruction_name(_gate(op)) is not None,
        on_stuck_raise=lambda op: ValueError(
            f"ANGLE_k contains an operation unsupported by kickmix: {op!r}"
        ),
    )

    # Replay into kickmix: named public registers first (q0, q1, ...), then
    # workspace qubits allocated by the builder. Each word gets its own
    # brot-qubit register, which is how the circuits state their precision.
    builder = km.CircuitBuilder()
    km_selection = builder.create_quantum_register(selection_bits, SELECTION_REGISTER)
    km_words = [
        builder.create_quantum_register(brot, word_register(j)) for j in range(batch_size)
    ]
    work = cirq.QubitOrder.DEFAULT.order_for(
        {q for op in operations for q in op.qubits} - set(public)
    )
    km_work = builder.alloc_qubits(len(work)) if work else []
    qubit_map = {
        **{public[i]: km_selection[i] for i in range(selection_bits)},
        **{
            public[selection_bits + i]: km_words[i // brot][i % brot]
            for i in range(batch_size * brot)
        },
        **{q: km_work[i] for i, q in enumerate(work)},
    }
    for operation in operations:
        _emit(builder, _gate(operation), [qubit_map[q] for q in operation.qubits])
    return builder.finish_circuit()


def angle_ks_to_circuits(angle_ks: Sequence[AngleK]) -> tuple[km.Circuit, ...]:
    """Translate a family ANGLE_0, ANGLE_1, ... into kickmix circuits."""
    angle_ks = tuple(angle_ks)
    if not angle_ks or tuple(oracle.k for oracle in angle_ks) != tuple(
        range(len(angle_ks))
    ):
        raise ValueError("angle_ks must be a nonempty family ordered by k=0,1,...")
    if any(oracle.signature != angle_ks[0].signature for oracle in angle_ks):
        raise ValueError("all ANGLE_k oracles must have the same register layout")
    return tuple(angle_k_to_circuit(oracle) for oracle in angle_ks)


def write_kmx_family(
    circuits: Sequence[km.Circuit],
    directory: str | Path,
    *,
    header: Sequence[str] = (),
    overwrite: bool = False,
) -> tuple[Path, ...]:
    """Write ``angle_k_0.kmx``, ``angle_k_1.kmx``, ... and return their paths.

    Each file is ``str(circuit)`` (kickmix text format) preceded by ``header``
    comment lines. Apply the files in order to stream the batches into the
    word registers; applying them in reverse order clears them. The file names
    are those ``verifier.circuits.read_kmx_family`` reads.
    """
    circuits = tuple(circuits)
    if not circuits:
        raise ValueError("circuits must be nonempty")
    directory = Path(directory)
    paths = tuple(kmx_path(directory, k) for k in range(len(circuits)))
    if not overwrite:
        existing = [path for path in paths if path.exists()]
        if existing:
            raise FileExistsError(f"refusing to overwrite {existing[0]}")
    directory.mkdir(parents=True, exist_ok=True)
    for k, (path, circuit) in enumerate(zip(paths, circuits)):
        comment = "".join(
            f"# {line}\n" if line else "#\n" for line in (f"ANGLE_{k}", *header)
        )
        path.write_text(comment + str(circuit).rstrip("\n") + "\n", encoding="utf-8")
    return paths


__all__ = [
    "angle_k_to_circuit",
    "angle_ks_to_circuits",
    "read_kmx_family",
    "write_kmx_family",
]
