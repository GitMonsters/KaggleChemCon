# 33 | QCC V8 members: the 18 files of submission sub_012a

Inputs of the busyaprime notebook for the Google Quantum AI Verifiable Quantum Circuit Compilation competition. These
are the 18 files of our submission `sub_012a` (SHA-256 of the zip `b28c1e4f...`), byte for byte. Sub-folders are
flattened: `dfthcsos__angle_k_0.kmb` is `dfthcsos/angle_k_0.kmb` in the zip. The notebook finds each file by name,
accepts it only by its SHA-256 (`SHA256SUMS`), and writes the zip again.

| file here | path in submission.zip | bytes | SHA-256 | author | licence |
|---|---|---:|---|---|---|
| `dfthcsos__submission.npz` | `dfthcsos/submission.npz` | 1,277,102 | `a60e21cb...` | **busyaprime** (Akmal Xodarev): refit and column gauge of Ahmed Berat Ozer's V9 factorization at his fixed words | Apache 2.0 |
| `dqi.kmb` | `dqi.kmb` | 530,569,266 | `f3d335c3...` | Ahmed Berat Ozer (narrowing) on Agniv Banerjee and the SomeJoint team's fused decoder | Apache 2.0 |
| `invsqrt.kmb` | `invsqrt.kmb` | 459,891 | `0766a111...` | mrmeows (team selenium) | Apache 2.0 |
| `dfthcsos__angle_k_0.kmb` to `dfthcsos__angle_k_14.kmb` | `dfthcsos/angle_k_0.kmb to dfthcsos/angle_k_14.kmb` | about 0.44 MB each | 15 files, see `SHA256SUMS` | Ahmed Berat Ozer (his J5 compiler, V9) | Apache 2.0 |

## What is ours

`dfthcsos__submission.npz`: our refit of the chemistry factorization's free weights plus a Gram-preserving column gauge
(the gauge idea is alexchilton's), at Ahmed Berat Ozer's fixed angle words, so his 15 angle circuits are used unchanged.
Started from his V9 `submission.npz` (SHA-256 `5c352f0a...`). The official verifier on x86-64:
PASS, cost 355,065,066.08, lambda_eff 47.537037, T 17,330, Q 431,
error_1 19.99925 mHa and error_2 19.99950 mHa (limit 20 mHa each). That is
-0.0343% against Ahmed Berat Ozer's V9 chemistry (355,186,805.37) and -0.0322% against Evgen Dvorkin's refined
npz (355,179,539.02), at the same circuits.

## Credits

- **Ahmed Berat Ozer**: the 15 angle circuits (his five-word compiler) and the 877-qubit `dqi.kmb` (his narrowing), from
  his notebook [Quantum AI - DF-THC Group Compression](https://www.kaggle.com/code/ahmedberatozer/quantum-ai-df-thc-group-compression) version 9 and dataset
  [vqcc-v9-components](https://www.kaggle.com/datasets/ahmedberatozer/vqcc-v9-components) version 1, Apache 2.0 ("Our own parts are released under Apache-2.0"). His credits,
  as his page gives them:
  - invsqrt: mrmeows, on the line of busyaprime, Phuc Thai (DePie), Dmitrii Gluzdov and marsyas.
  - DQI: Agniv Banerjee and the SomeJoint team; alex chilton; jday96314; mrmeows; manwithacat; Dmitrii Gluzdov;
    ViberTeam; Phuc Thai. His part is the narrowing.
  - Chemistry: Michael Rustad (the R10/B129/C46 factorization); mrmeows (compiler foundations, kernel planners,
    one-body tool); heliosli (the J5 design ideas); lightningv08 (five-word schedule); marculera (four-address loaders);
    kozykappa (joint refine code). His: the quantization, the lambda loop and the J5 implementation.
- **mrmeows** (team selenium): `invsqrt.kmb`, his 63-qubit build, unchanged, from [qcc-1375-open-circuits](https://www.kaggle.com/datasets/mrmeows/qcc-1375-open-circuits),
  Apache 2.0.
- **Agniv Banerjee and the SomeJoint team**: the fused confluent Hermite decoder under `dqi.kmb`
  ([notebook](https://www.kaggle.com/code/somejoint/quantum-circuits-21-34-local-gm-fused-dqi)).
- **Evgen Dvorkin** (evgendvorkin): his public notebook output ([Google Quantum AI, Deep EDA with Analogies](https://www.kaggle.com/code/evgendvorkin/google-quantum-ai-deep-eda-with-analogies),
  Apache 2.0) is the zip this one is made from: the same 17 other members (byte-equal to Ahmed Berat Ozer's V9 files),
  stamps and order; only `dfthcsos/submission.npz` differs (his refined npz is not in this dataset).
- **Google Quantum AI** (the hosts): the verifiers and kickmix that check these files, competition data.

Licences: every file here is Apache 2.0 work (`LICENSE`, `NOTICE.md`); each author keeps their copyright.
