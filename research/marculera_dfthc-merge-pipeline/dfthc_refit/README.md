# dfthc-refit: cheaper dfthcsos block encodings by merging DF-THC groups

A pipeline for the chemistry benchmark (`dfthcsos`, FeMoCo 76 orbitals) of the Google Verifiable
Quantum Circuit Compilation competition: train a DF-THC factorization, round its Givens angles to
10 bits, refine the identity offsets, refit the weights, build four-address ANGLE circuits and score
them with the official verifier. One recipe file describes a whole run.

**Best verified: 10 × 120 × 54, 3.97× the reference**: 1,276 rows, 25,434
Toffolis, 690 qubits, λ_eff 48.17, error_2 19.68 mHa (the published notebook shows the earlier 3.83×,
before the ANF circuits and the orthogonal gauge search).

- Notebook: [DF-THC: Merge Groups Before Dropping Them](https://www.kaggle.com/code/marculera/df-thc-merge-groups-before-dropping-them)
  (`notebooks/dfthc-merge`) replays the result on a CPU, explaining each step and calling these
  modules.
- Dataset: [dfthc-merge-pipeline](https://www.kaggle.com/datasets/marculera/dfthc-merge-pipeline)
  holds this pipeline as `dfthc_refit/` and the factorization files `B10c54.npz` (trained),
  `B10c54_q10s.npz` (rounded) and `B10c54_pol.npz` (polished).

By default the notebook loads the rounding and the polish from the dataset: the polish takes ~3
min on a fast laptop but over an hour on Kaggle's CPU. `POLISH = True` reruns it.

## The idea

The chemistry cost `λ_eff × Toffolis × qubits` is driven by the `76 + R·B` rows the ANGLE circuits
load. Public pipelines shrink `R·B` with many small groups (26 × 60). This one **merges** groups
first: two groups combine exactly into one with twice the vectors and copies (block-diagonal
weights), it retrains so every copy can use all the merged vectors, and only then drops groups.

- At equal rows and equal weights (28 × 60 × 60 against 14 × 120 × 60, three seeds) merged groups
  reach 16.2 mHa at 10 bits against 19.5.
- From the public (26, 60, 30) factorization with the same budget per stage, small groups fail the
  20 mHa gate even at 24 groups (1,516 rows), while merged groups pass at 10 (1,276 rows). Nine
  merged groups fail (17.8 mHa before rounding, 22.1 at 10 bits).

## Status of the claim

Merging is obvious once someone thinks to try it; writing it down does not establish the result.
What had to be tested, and where it stands:

1. **It trains better at the same parameter budget:** yes, at equal rows and equal weights
   (28 × 60 × 60 against 14 × 120 × 60, three seeds: 19.5 against 16.2 mHa at 10 bits).
2. **The gain survives the extra costs** (at equal rows ≈4% more Toffolis, 34 more qubits, ≈2%
   more λ_eff): yes, because merged groups reach 22% fewer rows than the seed.
3. **It rounds and compiles within the verifier's gates:** yes, verified at 3.83×.
4. **It persists across independent initializations and other molecules:** open. The three seeds
   differ only by a small jitter, and there is one molecule (FeMoCo, 76 orbitals).

Better final solution or faster convergence? The equal-weights runs are 41,000 steps and the
equal-budget runs 150,000 steps per stage. Merged groups stay ahead at the end of those schedules,
but neither shape was trained to full convergence.

Next experiments, before any paper: three independent initializations per shape (different seeds
or factorizations, not jitter), trained to convergence; then the best comparison on another system
from the starter kit (Fe2S2, Fe4S4, FeMoCo-54). The gap-floor finding is secondary: the weight
refit now recovers it (~2% of λ_eff).

Assessment: a small but legitimate contribution, a specific structural choice tested under this
benchmark's constraints, not a new method.

## Quick start

Requirements: Python ≥ 3.11, `numpy`, `scipy`, the competition's `kickmix`, and for training `jax`
and `optax` (a GPU helps: a step takes ~16 ms on a Kaggle T4, ~200 ms on a laptop CPU).

Data: the competition data (the directory with `verifiers/`), the hosts' starter kit (training
only) and a public seed factorization. In this repository they live under `data/`; elsewhere set
`DFTHC_DATA` and `DFTHC_STARTER` (see `paths.py`) and put the seed under
`$DFTHC_DATA/public/group26/dfthc_group_v2_26_60_30.npz`
([ahmedberatozer/df-thc-group-compression-factorization](https://www.kaggle.com/datasets/ahmedberatozer/df-thc-group-compression-factorization)).
On Kaggle, attach the competition, the starter kit and that dataset; paths are found automatically.

```
python notebooks/dfthc-refit/run.py recipes/merged_10x120x54.toml                   # the published result (~3.5 h on a T4)
python notebooks/dfthc-refit/run.py recipes/merged_10x120x54.toml --scale 0.0002    # a 2-minute smoke test of the whole chain
python -m pytest notebooks/dfthc-refit/tests                                        # exactness and circuit checks (< 1 s)
```

A run writes every step into `runs/recipes/<recipe>/` (or `--out`), ending with `submission/` (a
complete `dfthcsos` folder) and `result.json` (the verifier's verdict). A step whose output exists
is skipped, so rerunning resumes an interrupted run.

On Kaggle GPUs, `make_kernel.py` packs the modules and recipes into one private script kernel:

```
python notebooks/dfthc-refit/make_kernel.py --id YOU/your-kernel --job "run.py recipes/merged_10x120x54.toml"
kaggle kernels push -p notebooks/dfthc-refit/kaggle
```

Several `--job`s run side by side, one per GPU.

## Recipes

```toml
seed = "group26"          # public seed: group26 (26, 60, 30), group32 (32, 60, 30) or public (65, 60, 30)
merge = 2                 # merge groups this many at a time (1: no merging)
shape = [10, 120, 60]     # final (R, B, C); B and C are fixed after merging

[[train]]                 # one fresh optimizer per entry
groups = 12               # drop to this many groups first (fade, then fit), or keep the count
fade = 2000
fit = 150000
lr_final = 1e-5           # any other middle.py option: lr, warmup, kappa, jitter, rng, ...

[lbfgs]                   # optional: lbfgs_fit.py options (before rounding)
iters = 3000
kappa = 5e-5

[round]                   # quantize.py options
brot = 10                 # 8: see recipes/merged_*_b8.toml
stage_steps = 3000
final_steps = 8000        # also free = "1.0,0.9995,..." and fit_steps

[polish]                  # optional: polish_words.py options
sweeps = 4

[refit]                   # optional: refit_weights.py options
target = 19.92            # error_2 kept below this (mHa)

[build]
qrom = "four"             # four-address circuits (10-bit words), "tensor" (five 8-bit words), or "reference"
shots = 64
```

`recipes/merged_10x120x54.toml` is the published result (`merged_10x120x60.toml`, without the copy
cut, gives 3.80×); `recipes/small_24x60x30.toml` is the
many-small-groups comparison. Natural next recipes: gap-aware training (`rho_lam`), other merge
factors (9 merged groups fail, see above).

## Pipeline

| Step | Module | Notes |
|---|---|---|
| Recipe | `run.py` | chains the steps below, resumable; `--scale` for smoke tests, `--dry-run` to print the commands |
| Seed | `public_seed.py` | public factorizations, decoded from their 10-bit words |
| Merge | `structure.py` | exact merging, group importance, the represented tensor (numpy only) |
| Train | `middle.py` | staged drops with fades, Adam on ε_fro + κ·D(w); `--init` continues; a larger B grows every group (new random bases, 1e-4 weights) |
| Fit | `lbfgs_fit.py` | optional: float64 L-BFGS on the same objective before rounding (15.35 against Adam's 17.99 mHa on a (4, 239, 128) structure) |
| Round | `quantize.py` | progressive rounding in the signed gauge, highest-impact angles first; `--free` sets the schedule (a leading 1.0 is a continuous refit), angles are not clipped and rows are moved into the signed gauge exactly at the end |
| Polish | `polish_words.py` | exact coordinate descent on the rounded words (±1 per angle, exact error_2 change), then w refit; 19.95 → 19.57 mHa in 4 sweeps on the 54-copy structure (−0.5 mHa on 9 groups) |
| Offsets | `refine.py` | orthogonal gauge of each group's copy columns (exact pair sweeps, then descent on the orthogonal group with basin hopping) alternated with L-BFGS on λ_eff (E_gap is convex in the offsets) |
| Weights | `refit_weights.py` | refit (w, W) at fixed words under an error_2 target, with an exact error_2 formula; recovers the gap-floor tax (λ_eff −2%) |
| Circuits | `fourqrom.py` | four-address ANGLE circuits in ≤ 128 qubits: an ANF load (one Toffoli per address monomial of degree ≥ 2, the don't-care rows cancel the rest), paired combine (41 Toffolis instead of 81) and a measurement-based erase whose phase lookup is also ANF (rows/4 + ~57 Toffolis per pass, 376 at 1,276 rows) |
| Circuits, 8 bits | `tensorqrom.py` | five 8-bit words per circuit with no bank: every address monomial cached (low 4 bits and high bits), mixed terms covered two per Toffoli by a GF(2) decomposition, erase by measurement (~293 Toffolis per pass at 1,032 rows, 123 qubits); `--qrom tensor` |
| One-body rows | `onebody.py` | exact ±1/±2 word moves plus random kicks on the one-body rows (error_1 54 → 13 mHa at 8 bits); exact signed-gauge conversion; used by `refine.py` when brot ≤ 9 |
| Score | `build.py` | writes a `dfthcsos` folder and runs the official verifier |

Support: `paths.py` (data locations), `encode.py` (Givens angles ↔ words, verifier metrics),
`landscape.py` / `landscape4.py` (cost models), `make_kernel.py` (Kaggle GPU kernels).
`experiments/` keeps the superseded first pipeline and the dead ends below, as a record.

## Findings

- Hard pruning wrecks the fit (B 60 → 50: 855 mHa); fade removed groups out and refit in stages.
- Weights are ~1e-4 in scale: a fresh optimizer on a converged structure needs a warm-up.
- Long fits matter: 150k steps per stage, and 10 groups needed 150k more to pass.
- Merged structures are rounding-sensitive (at 10 groups: rounding to nearest +36 mHa,
  progressive +3.9 mHa); the damage sits in a few high-curvature angles.
- Merged groups need their copies: cutting 60 → 30 fails at every stage (24.8 / 29.7 mHa at 12 /
  11 groups). The 640-vector dictionary cannot lose copies either (384 → 256: 116 mHa).
- Training w on ε + D raises the convex floor of E_gap at fixed w (3.91 → 4.08 Ha), about 2% of
  λ_eff; `refit_weights.py` recovers it after rounding.
- No effect: Givens-tree relabeling (±1 mHa), AdaRound, a second-order rounding direction,
  training through the rounded vectors, reordering rows to zero four-address bank bits.
- **8 bits.** The cost model's inner state preparation drops from 576 to 304 workspace qubits at
  brot 8, only for B ≥ 129; it also adds 514 Toffolis at every B = 16k (so B = 16k − 1).
  Clipping angles to the signed range cost up to two rounding steps on 3% of the angles; with
  free angles, an L-BFGS fit and κ = 5e-5, a (4, 239, 128) structure rounds to 8 bits at 18.8 mHa.
  Merged (6, 159, 80) and (7, 143, 64) from the (26, 60, 30) seed stop at 19.7 and 20.4 mHa before
  rounding. L-BFGS refits between freezing stages converge too slowly: Adam stays.

## Credits

Built on public Kaggle work, all Apache-2.0:

- **ahmedberatozer, DF-THC Group Compression:** the D(w) penalty, staged group dropping,
  sensitivity-ordered quantization and long refits; its factorizations are the seeds.
- **kozykappa, Complete-Domain Affine Compilation** (on the banked QROM of Kozykorpesh and Sumit
  Chongder): the four-address lookup with difference banks and the signed gauge.
- **kozykappa (sparse SOS factors) and Artem1410 (anazemcev):** the gauge rotation of the copy
  columns and the effective-norm refinement. `refine.py` is adapted from the group-compression
  notebook's implementation of both (`chemistry_refine.py`); `quantize.py` and `fourqrom.py` are
  our own code, written after reading its `dfthc_train.py` and `gauge_qrom.py`.
- **phucthaiv02, QCC Verified Portfolio + DF-THC Retraining Study:** the paired combine (reconstructed
  in `fourqrom.py` from its description), refitting w and W after rounding (`refit_weights.py`, our
  own implementation), exact word-level coordinate descent (our `polish_words.py`), and a per-group
  orthogonal copy gauge, an idea it credits to haideptry (our own search in `refine.py`).
- **ANF loading and erasure** (algebraic normal form of the table, with the never-addressed rows as
  don't-cares) appear in the public banked-QROM code (`banked_lab.py` in the group-compression
  notebook, after Kozykorpesh and Sumit Chongder) and in phucthaiv02's pruned ANF erase; the ANF load
  and erase in `fourqrom.py` are our own construction for the four-address circuit.
- **somejoint, Toffoli Diet:** the 640-vector dictionary.
- **mrmeows, QCC open circuits and sources:** the exact lookup emitter's design that `tensorqrom.py`
  follows (cached address monomials, paired affine products, measurement erasure; our decomposition
  and code) and `refresh_one_body8.py`'s ±1 one-body polish (our `onebody.py` adds kicks, ±2 moves
  and the exact gauge conversion).
- **lightningv08:** five words per ANGLE circuit. **michaelrustad:** the B ≥ 129 cliff at 8 bits and
  L-BFGS continuous refits. **ahmedberatozer (v8 components):** the fine start of the freezing
  schedule.
- **k256net and Dmitrii Gluzdov:** the public 10-bit (65, 60, 30) factorization.
- **The hosts:** the verifier, kickmix (whose `del_lookup` the truncated erase follows) and the
  starter kit (the DF-THC residual used in training).

Our own: merging as a warm start and the merged schedules, the equal-weights and equal-budget
comparisons, the convexity of the gap floor, the exact error_2 formula for the refit, the
the four-address ANF load and erase construction, the Gray-ordered top level and truncated erase of
the unary variant, and the recipe runner.
