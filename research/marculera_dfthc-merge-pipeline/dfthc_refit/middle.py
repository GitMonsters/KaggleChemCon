"""Train a DF-THC structure: merge groups exactly, drop groups in stages, refit (U, w).

1. **Seed** (`--seed`) or continue (`--init`): a public factorization, or an earlier result.
2. **Merge** groups k at a time (`--merge`, exact; `structure.merge`).
3. **Drop** the least important groups through `--stages`: each drop fades the dropped groups'
   weights from 1 to 0 over `--fade` steps while fitting, slices them away, and refits.
4. **Fit**: Adam on (U, w) for `epsilon_fro + kappa * D(w)`, the group-compression objective.
   `D(w) = 1/2 sum_rc [(sum_b |w|)^2 - (sum_b w)^2]` is zero when every square is semidefinite.
   The identity offsets do not enter epsilon_fro; `refine.py` fits them for the final words.

Options kept for the record of the experiments: `--copies` (fade copies in each group's SVD
basis; it failed for merged groups), `--rho-lam` (gap-aware training) and `--jitter` (seeds).

    python notebooks/dfthc-refit/middle.py R B C [--seed group26 --merge 2 | --init F.npz]
        [--stages 12 11 10] [--fade 2000] [--fit 150000] [--lr 1e-3 --lr-final 1e-5] [--out F.npz]

`run.py` chains these steps from a recipe file.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import FCIDUMP, SHIFTS, UPPER_ENERGY, WORK as RUNS, load  # noqa: E402  (puts the starter kit on sys.path)
import encode  # noqa: E402
import landscape4  # noqa: E402
from public_seed import convert  # noqa: E402
from refine import Problem, optimize_offsets  # noqa: E402
from structure import grow_bases, importance, merge  # noqa: E402
from dfthcsos.scoreinfra.reference import check_fcidump, load_reference  # noqa: E402
from dfthcsos.verify import native_inputs  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import optax  # noqa: E402
from dfthc import df_thc_sga  # noqa: E402
from dfthc.train_df_thc_sga import Initialize  # noqa: E402

jax.config.update('jax_default_matmul_precision', 'highest')


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('R', type=int)
    ap.add_argument('B', type=int)
    ap.add_argument('C', type=int)
    ap.add_argument('--merge', type=int, default=1)
    ap.add_argument('--stages', type=int, nargs='+', help='group counts to drop through, ending at R')
    ap.add_argument('--stage-fit', type=int, default=3000, help='fit steps after each intermediate drop')
    ap.add_argument('--fade', type=int, default=3000, help='fade steps per drop')
    ap.add_argument('--fit', type=int, default=9000)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--lr-final', type=float, default=1e-4)
    ap.add_argument('--warmup', type=int, default=0, help='steps of linear learning-rate warmup per fit')
    ap.add_argument('--kappa', type=float, default=2e-5)
    ap.add_argument('--seed', choices=('group32', 'group26', 'group24', 'public'), default='group32')
    ap.add_argument('--init', type=Path, help='continue from this structure (U, W, H2_I) instead of a seed')
    ap.add_argument('--copies', type=int, help='first truncate each group to this many copies (SVD of w)')
    ap.add_argument('--rho-lam', type=float, default=0.0,
                    help='weight of lambda_eff at the optimal offsets in the fit (gap-aware training)')
    ap.add_argument('--lam-every', type=int, default=200, help='steps between solves of the optimal offsets')
    ap.add_argument('--gap-margin', type=float, default=0.002)
    ap.add_argument('--jitter', type=float, default=0.0,
                    help='relative noise on the starting U and w, for independent seeds (e.g. 0.01)')
    ap.add_argument('--rng', type=int, default=0, help='seed of the --jitter noise')
    ap.add_argument('--print-every', type=int, default=500)
    ap.add_argument('--out', type=Path)
    a = ap.parse_args()
    out = a.out or RUNS / 'middle' / f'{a.R}x{a.B}x{a.C}_m{a.merge}.npz'
    out.parent.mkdir(parents=True, exist_ok=True)

    ref = Initialize(str(FCIDUMP), UPPER_ENERGY)
    with np.load(SHIFTS) as z:
        frozen = {k: jnp.asarray(z[k], dtype=df_thc_sga.dtype) for k in ('Hsym_lower', 'H1_I')}
    if a.init:  # keep training an earlier result; with --stages omitted nothing is dropped
        p = load(a.init)
    else:
        p = merge(load(convert(out.with_name(f'{out.stem}_seed.npz'), a.seed)), a.merge, a.C)
    if a.jitter > 0:
        # Multiplicative noise: a merged group's block-diagonal zeros stay zero at the start.
        rng = np.random.default_rng(a.rng)
        p = {**p, 'U': p['U'] * (1 + a.jitter * rng.normal(size=p['U'].shape)),
             'W': p['W'] * (1 + a.jitter * rng.normal(size=p['W'].shape))}
        print(f'jitter {a.jitter} (rng {a.rng})', flush=True)
    if p['W'].shape[1] < a.B:  # more bases than the seed or the last step: grow every group
        p = grow_bases(p, a.B, seed=a.rng)
        print(f'grew to {a.B} bases per group', flush=True)
    reduce_copies = bool(a.copies) and a.copies < p['W'].shape[2]
    if reduce_copies:
        # Rotate each group's copies into the SVD basis of w: exact, since the two-body tensor
        # only sees w w^T (sum_c X_c (x) X_c = sum_bb' (w w^T)_bb' P_b (x) P_b'). The weakest
        # components are then faded out below, like dropped groups.
        p = {**p, 'W': np.stack([wr @ np.linalg.svd(wr, full_matrices=False)[2].T for wr in p['W']])}
        if a.copies != a.C:
            sys.exit('--copies must equal C')
    R1, B1, C1 = p['W'].shape
    if reduce_copies:
        C1 = a.copies
    if B1 != a.B or C1 != a.C or R1 < a.R:
        sys.exit(f'merged shape {(R1, B1, C1)} does not reach {(a.R, a.B, a.C)}')
    stages = a.stages or ([a.R] if R1 > a.R else [])  # nothing to drop when continuing at the target shape
    if stages and (stages[-1] != a.R or any(x <= y for x, y in zip([R1] + stages, stages))):
        sys.exit(f'--stages must decrease from {R1} and end at R={a.R}')
    mask = jnp.ones((R1, 1, 1), dtype=df_thc_sga.dtype)  # 1 = kept, faded from 1 to 0 otherwise
    print(f'seed {a.seed}: merged to {(R1, B1, C1)}, dropping groups in stages {stages}', flush=True)

    # Gap-aware training: lambda_eff at the offsets W* that minimize it for the current structure.
    # min_W lambda_eff is not smooth to differentiate through, but W* is re-solved every
    # --lam-every steps (convex in W, refine.optimize_offsets) and held fixed in between, which
    # by the envelope theorem gives the right gradient for (U, w) at W*.
    ref_np = load_reference(check_fcidump())
    eta = int(ref_np['electrons'])
    with np.load(SHIFTS) as z:
        _, _, S_np, beta = native_inputs({'U': p['U'], 'W': p['W'], 'H2_I': p['H2_I'], 'H1_I': z['H1_I'],
                                          'Hsym_lower': z['Hsym_lower']})
    h1_base = jnp.asarray(ref_np['h1_thc'] - 0.5 * eta * S_np + beta * np.eye(S_np.shape[0]), dtype=df_thc_sga.dtype)
    energy = float(UPPER_ENERGY - ref_np['ecore'] + eta * beta)

    def lam_eff(U, w, W):
        u = U / jnp.linalg.norm(U, axis=2, keepdims=True)
        k = jnp.einsum('rc,rbc->rb', W, w)
        e = jnp.linalg.eigvalsh(h1_base - jnp.einsum('rb,rbp,rbq->pq', k, u, u))
        gap = energy + jnp.sum(jnp.abs(e)) - jnp.sum(e) + 0.5 * jnp.sum(W ** 2)
        lam = jnp.sum(jnp.abs(e)) + 0.25 * jnp.sum((jnp.sum(jnp.abs(w), axis=1) + jnp.abs(W + jnp.sum(w, axis=1))) ** 2)
        return jnp.sqrt(jnp.maximum(gap * (2.0 * lam - gap), 1e-12))

    def metrics(theta, s, mask, W_off, rho):
        w = theta['w'] * (mask + s * (1 - mask))
        U = theta['U']
        H2_sqrt = df_thc_sga.Compute_H2_sqrt(U, w)
        resid = df_thc_sga.Compute_H2_residual_after_symmetry_shift(frozen, ref, H2_sqrt)
        eps = df_thc_sga.Compute_H2_norm(resid, ref['tril_diag'])
        D = 0.5 * jnp.sum(jnp.sum(jnp.abs(w), axis=1) ** 2 - jnp.sum(w, axis=1) ** 2)
        lam = lam_eff(U, w, W_off)
        return {'eps': eps, 'D': D, 'lam': lam, 'cost': eps + a.kappa * D + rho * lam}

    def offsets_for(theta, W_off):
        U = np.asarray(theta['U'], np.float64)
        u = U / np.linalg.norm(U, axis=2, keepdims=True)
        problem = Problem(u, np.asarray(theta['w'], np.float64), ref_np, S_np, beta)
        return optimize_offsets(problem, W_off, a.gap_margin, maxiter=200)

    def fit(theta, steps: int, s_of_step, label: str):
        # A fresh Adam takes ~lr-sized steps in every coordinate at once: ramp it up when
        # continuing from a converged structure (--warmup), or it jumps far off the optimum.
        sched = optax.warmup_exponential_decay_schedule(
            0.0 if a.warmup else a.lr, a.lr, max(a.warmup, 1), max(steps, 1), a.lr_final / a.lr)
        opt = optax.adam(sched)
        state = opt.init(theta)
        vg = jax.value_and_grad(lambda th, s, mask, W_off, rho: metrics(th, s, mask, W_off, rho)['cost'])

        @jax.jit
        def step(theta, state, s, mask, W_off, rho):
            v, g = vg(theta, s, mask, W_off, rho)
            upd, state = opt.update(g, state, theta)
            return optax.apply_updates(theta, upd), state, v

        report = jax.jit(lambda th, s, mask, W_off, rho: metrics(th, s, mask, W_off, rho))
        gap_aware = a.rho_lam > 0 and label.startswith('fit')
        rho = jnp.asarray(a.rho_lam if gap_aware else 0.0, df_thc_sga.dtype)
        W_off = np.zeros((theta['w'].shape[0], theta['w'].shape[2]))
        W_j = jnp.asarray(W_off, df_thc_sga.dtype)
        best, best_cost, t = theta, np.inf, time.time()
        for i in range(steps + 1):
            s = s_of_step(i)
            if gap_aware and i % a.lam_every == 0:
                W_off = offsets_for(theta, W_off)
                W_j = jnp.asarray(W_off, df_thc_sga.dtype)
                best_cost = np.inf  # costs at different W* are not comparable
            if i % a.print_every == 0 or i == steps:
                o = report(theta, s, mask, W_j, rho)
                print(f"{label} {i:6d} s={s:.3f} eps {1e3 * float(o['eps']):7.2f} mHa  D {float(o['D']):8.3f}"
                      + (f"  lambda_eff {float(o['lam']):7.3f}" if gap_aware else '')
                      + f"  ({time.time() - t:.0f} s)", flush=True)
            if i < steps:
                previous = theta
                theta, state, cost = step(theta, state, jnp.asarray(s, df_thc_sga.dtype), mask, W_j, rho)
                if s == 0.0 and float(cost) < best_cost:  # best only once the dropped groups are gone
                    best, best_cost = previous, float(cost)
        return theta if best_cost == np.inf else best

    f32 = lambda x: jnp.asarray(x, dtype=df_thc_sga.dtype)  # noqa: E731
    theta = {'U': f32(p['U']), 'w': f32(p['W'])}
    if reduce_copies:
        m = np.ones((R1, 1, p['W'].shape[2]))
        m[:, :, a.copies:] = 0.0
        mask = jnp.asarray(m, dtype=df_thc_sga.dtype)
        theta = fit(theta, a.fade, lambda i: max(0.0, 1.0 - i / max(a.fade, 1)), f'fadeC{a.copies:<2d}')
        theta = {'U': theta['U'], 'w': theta['w'][:, :, :a.copies]}
        mask = jnp.ones((R1, 1, 1), dtype=df_thc_sga.dtype)
        if stages:
            theta = fit(theta, a.stage_fit, lambda i: 0.0, f'fitC{a.copies:<3d}')
    alive = np.arange(R1)  # merged groups still present
    for n, target in enumerate(stages):
        # Drop a slice at a time (the public pipeline: 65 -> 55 -> ... -> 32), refitting after
        # each: dropping half at once leaves the fit stuck far above 20 mHa.
        imp = importance(np.asarray(theta['U'], np.float64), np.asarray(theta['w'], np.float64))
        keep = np.sort(np.argsort(-imp)[:target])
        m = np.zeros((len(alive), 1, 1))
        m[keep] = 1.0
        mask = jnp.asarray(m, dtype=df_thc_sga.dtype)
        theta = fit(theta, a.fade, lambda i: max(0.0, 1.0 - i / max(a.fade, 1)), f'fade{target:<3d}')
        theta = {'U': theta['U'][keep], 'w': theta['w'][keep]}
        alive = alive[keep]
        mask = jnp.ones((target, 1, 1), dtype=df_thc_sga.dtype)
        last = n == len(stages) - 1
        theta = fit(theta, a.fit if last else a.stage_fit, lambda i: 0.0, f'fit{target:<4d}')
    if not stages:
        theta = fit(theta, a.fit, lambda i: 0.0, 'fit ')

    U, w = np.asarray(theta['U'], np.float64), np.asarray(theta['w'], np.float64)
    # Zero identity offsets as placeholders: refine.py refits them for the final words.
    np.savez(out, U=U, W=w, H2_I=w.sum(axis=1), H1_I=np.asarray(frozen['H1_I']),
             Hsym_lower=np.asarray(frozen['Hsym_lower']))
    m = encode.metrics(out, encode.log_depth_tree(), 20)
    loss49 = landscape4.loss(a.R, a.B, a.C, 49.0)
    result = {'shape': [a.R, a.B, a.C], 'merge': a.merge, 'rows': 76 + a.R * a.B,
              'error_2_mHa': 1e3 * m['error_2'], 'Lambda': m['Lambda'], 'E_gap': m['E_gap'],
              'lambda_eff_before_offset_refit': m['lambda_eff'],
              'x_ref_at_lambda49': landscape4.REFERENCE / loss49[0], 'T': loss49[1], 'Q': loss49[2]}
    print('RESULT', json.dumps(result), flush=True)
    print('saved', out)


if __name__ == '__main__':
    main()
