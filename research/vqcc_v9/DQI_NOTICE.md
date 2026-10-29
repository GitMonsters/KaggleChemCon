# DQI circuit `dqi.kmb.xz` (notebook version 9)

`dqi.kmb` (xz-compressed here, SHA-256 of the unpacked file in `dqi_verify.json`) is SomeJoint's fused confluent
Hermite decoder, narrowed from 889 to 877 qubits.

## Upstream

The decoder is by Agniv Banerjee and the SomeJoint team (notebook
[Quantum Circuits | 21.34 Local GM: Fused DQI](https://www.kaggle.com/code/somejoint/quantum-circuits-21-34-local-gm-fused-dqi)).
Its full source is embedded in that notebook. We rebuilt their file from it byte for byte (SHA-256 `7beea291…`)
before changing anything.

Their design starts from alex chilton's 883-qubit Schur decoder, which is based on jday96314's QuantumAI decoder
([GitHub](https://github.com/jday96314/QuantumAI); the repository states no license). It also uses the round17 mask
projection (mrmeows, after manwithacat) and Ahmed Berat Özer's version-8 work. Earlier links in the chain are Dmitrii
Gluzdov's round12, SomeJoint / Agniv Banerjee's Bézout–Hankel solve, ViberTeam and Phuc Thai.

## What we changed

The 889-qubit peak came from two places, each with 49 workspace qubits:

- the jets, where z = 1/u, the product v·z² and a zero flag are alive together;
- the outer fold of the undo-synthesis.

The changes:

1. **Forward jet.** The old u is measured as soon as z exists, and z moves into u's register. It still costs 290
   Toffolis, now with 37 workspace qubits.
2. **Reverse jet.** It keeps the norm and inverts u in place. The inverse phase drops from 106 Toffolis to 34, but the
   jet as a whole goes from 396 to 418 Toffolis.
3. **Small kernels.** The outer division now costs 334 Toffolis (was 324), and the local in-place inverse 224 (was 209).
4. **Undo-synthesis inner folds.** These fall back to the older 24-scratch division, +46 Toffolis per fold. The reason
   is the data width: 852 data qubits are live there (the `r` word and the border word are extra), so the
   retained-norm division would set the peak at 889 again.
5. **Conversion.** The conversion kept a 12-qubit history (lead) word, a copy of the root inverse. Most of its uses now
   read the root's own inverse register instead. On root-gap charts the value the gap swap needs is already in the
   child's last word, so the child swaps with that word, and two extra controlled swaps keep the multiplier and eraser
   unchanged. This costs +24 Toffolis per direction and takes the conversion peak from 879 to 867. Outputs are identical
   to the original on 160 cases each of random, no-gap, one-gap, two-gap, root-gap, double root-gap and child-gap
   charts. Only the unsupported degree-3 root chart differs (about 6·10⁻⁸ per shot), and it stays reversible.

**Result:**

- Q goes from 889 to 877. Extraction, synthesis and their undo now all peak at 877, the common floor of their 12-bit
  inverse and scaling kernels.
- Mean Toffolis go from 474,308 to 482,543 (+1.74 %).
- Q²·T falls by 1.0 %.

On identical simulated shots (same `--seed`: 20261027, 777 and 4242), the error counts are the same as the original's
and the ratio is 1.0 % higher on each seed (for example 55.28× against 54.73× at seed 20261027). The build scripts are in `src/dqi_narrow/`.
