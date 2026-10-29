# VQCC v9 components: narrow fused DQI and R10 chemistry with J5 lookups

These are the circuits for version 9 of the notebook
[Quantum AI - DF-THC Group Compression](https://www.kaggle.com/code/ahmedberatozer/quantum-ai-df-thc-group-compression),
for the Google Quantum AI Verifiable Quantum Circuit Compilation competition. The notebook finds every file by its
SHA-256 and runs the unmodified official verifiers on them.

## `invsqrt.kmb`

mrmeows' 63-qubit build, unchanged.

- Source: [qcc-1375-open-circuits](https://www.kaggle.com/datasets/mrmeows/qcc-1375-open-circuits) version 3,
  Apache-2.0. The notice is in `invsqrt_NOTICE.txt`.
- Mean Toffolis: 1,348.5.
- Cost 103,546.25, ratio 20.0565.

## `dqi.kmb.xz`

SomeJoint's fused confluent Hermite decoder, narrowed by us to 877 qubits. The notebook unpacks it with Python's
`lzma`.

- Mean Toffolis: 482,543.
- Official verifier, salt `notebook-verification`: PASS, 0 value errors and 0 phase errors in 102,400
  shots, R = 1.1913.
- Cost 4.4212e+11, ratio 55.28.
- Circuit SHA-256 `f3d335c3f693aaab93bcea8750605f206dd2ba6d04ad5bf9a6de1de1ac5ec3fb`.

`DQI_NOTICE.md` gives the full provenance, what changed and the identical-shot comparison.

## `dfthcsos/`

Michael Rustad's (R, B, C) = (10, 129, 46) 8-bit factorization, re-quantized and refit by us. 1366 angle rows,
15 ANGLE circuits with J = 5 words each, compiled with our five-word compiler.

Official verifier:

- PASS.
- λ_eff 47.553.
- error₁ 14.66 mHa, error₂ 19.998 mHa.
- T 17,330, Q 431.
- Cost 3.5519e+08, ratio **9.447**.

### How it was made

1. **Very fine progressive quantization.** We refit Rustad's continuous factors to error₂ 13.08 mHa. Then we quantized
   to 8 bits, freezing the most sensitive words first in steps of 0.05 %, 0.1 %, 0.25 %, … of all words. After each
   step an Adam re-fit of the free angles and weights absorbs the rounding. The 8-bit error₂ comes out at 19.41 mHa,
   against 19.97 for the public quantization. H1_I, Hsym_lower and tree_edges are byte-identical to Rustad's file.
2. **λ loop.** An offset-only optimizer, then three rounds of joint refit, column gauge and offset optimization. The
   one-body words come from mrmeows' `refresh_one_body8.py`.
3. **Five-word ANGLE circuits.** Our own compiler (`src/chem/j5_native.py`, `j5layout.py`, `j5_emit.py`) implements the
   design heliosli described publicly:
   - 128 qubits: 11 selection, 40 word and 77 high-cache qubits.
   - The five word bit-7 qubits serve as scratch. The sign gauge makes bit 7 equal to bit 6, so one CNOT clears it;
     it is restored at the end.
   - A three-phase low cache holds at most 4 qubits.
   - A planner confines relations and pairing planes to one phase's live set. It is built on mrmeows' kernel-relation
     and quotient-pairing code.
   - A greedy relation search (blocks with at most one address vertex, the address 4-block, structured kernel candidates).
   - Per circuit: 57–61 relations and about 321 Toffolis per pass.
   - Every circuit is checked on all addresses, forward and reverse.

`plans.json` holds the per-circuit plans. The scripts in `src/chem/` are research code, kept as used.

## Credits

- **invsqrt:** mrmeows, on the line of busyaprime, Phuc Thai (DePie), Dmitrii Gluzdov and marsyas.
- **DQI:** Agniv Banerjee and the SomeJoint team; alex chilton; jday96314; mrmeows; manwithacat; Dmitrii Gluzdov;
  ViberTeam; Phuc Thai. Our part is the narrowing.
- **Chemistry:**
  - Michael Rustad: the R10/B129/C46 factorization.
  - mrmeows: the compiler foundations, kernel planners and one-body tool.
  - heliosli: the J5 design ideas.
  - lightningv08: the five-word schedule.
  - marculera: the four-address loaders.
  - kozykappa: the joint refine code.
  - Ours: the quantization, λ loop and J5 implementation.

Our own parts are released under Apache-2.0.
