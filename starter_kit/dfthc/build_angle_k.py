"""Build the streamed ANGLE_k Boolean QROM oracles.

Each unit vector used by the Majorana basis transformation is represented by
N - 1 ordered Givens-rotation angles. The caller supplies the corresponding
brot-bit integer words and is responsible for the angle encoding and rounding
policy.

For DFTHC/SOSSA verification, the required angle-oracle address space
contains

    A = N + R * B

vectors: N one-body eigenvectors and R * B two-body vectors u_b^(r).
The verifier assumes that layout: the one-body vectors occupy addresses 0
through N - 1, followed by the two-body vectors in (r, b) order.

The complete classical angle table therefore contains

    A * (N - 1) * brot

bits, although an all-at-once QROM query for one selected vector address
outputs only (N - 1) * brot bits.

The SOSSA implementation loads all N - 1 angle words simultaneously. To
trade angle-register width for additional lookup calls, choose a batch size

    1 <= J <= N - 1

and define

    K = ceil((N - 1) / J).

The reusable output register contains J angle words, or J * brot qubits.
There are K separately compiled ANGLE_k oracles. The batch index k is
hard-wired into each oracle and is not a quantum input.

For vector address a, define the zero-padded raw batch

    batch[k, a] = (
        angle[a, k*J],
        ...,
        angle[a, k*J + J - 1],
    ).

ANGLE_0 XORs the first raw batch into the output register. For k > 0,
ANGLE_k XORs the difference between adjacent batches:

    delta[0, a] = batch[0, a]
    delta[k, a] = batch[k, a] XOR batch[k - 1, a].

Thus

    ANGLE_k: |a>|z> -> |a>|z XOR delta[k, a]>.

When z initially contains batch[k - 1, a], ANGLE_k changes it directly into
batch[k, a], avoiding a separate uncompute of the previous batch.

One forward traversal uses K ANGLE calls. In the standard
forward-Givens / MAJ / inverse-Givens schedule, every ANGLE_k is called once
forward and once through its explicit Qualtran adjoint. The underlying XOR
QROM is self-inverse, so both have the same Boolean action, but Qualtran keeps
the adjoint circuit distinct. The reverse traversal ends with ANGLE_0's
adjoint, which clears the reusable angle register.

The batch size J partitions the N - 1 angle words; it does not partition the
N + R * B vector addresses. Each ANGLE_k below uses an ordinary Qualtran QROM.
"""

from __future__ import annotations

from functools import cached_property
from math import ceil

import attrs
import numpy as np
import numpy.typing as npt
from qualtran import Bloq, BloqBuilder, Register, Side, Signature, SoquetT
from qualtran.bloqs.data_loading import QROM


@attrs.frozen
class AngleK(Bloq):
    """One streamed-angle XOR oracle backed by an ordinary QROM."""

    k: int
    qrom: QROM

    @cached_property
    def signature(self) -> Signature:
        selection = self.qrom.selection_registers[0]
        target = self.qrom.target_registers[0]
        return Signature(
            [
                Register("a", selection.dtype, side=Side.THRU),
                Register("z", target.dtype, shape=target.shape, side=Side.THRU),
            ]
        )

    def build_composite_bloq(
        self,
        bb: BloqBuilder,
        *,
        a: SoquetT,
        z: SoquetT,
    ) -> dict[str, SoquetT]:
        outputs = bb.add_d(self.qrom, selection=a, target0_=z)
        return {"a": outputs["selection"], "z": outputs["target0_"]}

    def __str__(self) -> str:
        return f"ANGLE_{self.k}"


def build_angle_ks(
    angle_words: npt.NDArray[np.integer],
    *,
    brot: int,
    batch_size: int,
) -> tuple[AngleK, ...]:
    """Build ANGLE_k QROMs from caller-encoded integer angle words.

    This function does not convert floating-point angles or choose a rounding
    policy.
    """
    words = np.asarray(angle_words)
    if not np.issubdtype(words.dtype, np.integer):
        raise TypeError("angle_words must already contain encoded integers")

    num_vectors, num_angles = words.shape
    num_batches = ceil(num_angles / batch_size)

    padded = np.zeros((num_vectors, num_batches * batch_size), dtype=words.dtype)
    padded[:, :num_angles] = words
    batches = padded.reshape(num_vectors, num_batches, batch_size).transpose(1, 0, 2)

    deltas = batches.copy()
    if num_batches > 1:
        deltas[1:] = np.bitwise_xor(batches[1:], batches[:-1])

    return tuple(
        AngleK(
            k=k,
            qrom=QROM.build_from_data(
                deltas[k],
                target_bitsizes=(brot,),
                target_shapes=((batch_size,),),
            ),
        )
        for k in range(num_batches)
    )


__all__ = ["AngleK", "build_angle_ks"]
