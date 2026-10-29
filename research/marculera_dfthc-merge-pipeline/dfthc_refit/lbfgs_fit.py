"""Continuous fit of a structure by L-BFGS in float64 (Adam in float32 stalls higher).

    python notebooks/dfthc-refit/lbfgs_fit.py STRUCTURE.npz OUT.npz [--iters 3000] [--kappa 2e-5]

Minimizes the group-compression objective epsilon_fro + kappa * D(w) of middle.py over (U, w), with
exact JAX gradients and scipy's L-BFGS-B. michaelrustad reports L-BFGS continuous refits near
10 mHa where our Adam fits stop at 15-16. On a (4, 239, 128) structure it reaches 15.35 mHa where
8,000 Adam steps stop at 17.99; a larger kappa (5e-5) keeps the accuracy and halves D(w).
OUT.npz has U, W (and H2_I = sum_b w, offsets are refit later) like middle.py output.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import FCIDUMP, SHIFTS, UPPER_ENERGY  # noqa: E402  (puts the starter kit on sys.path)

import jax  # noqa: E402

jax.config.update('jax_enable_x64', True)
jax.config.update('jax_default_matmul_precision', 'highest')
import jax.numpy as jnp  # noqa: E402
from dfthc import df_thc_sga  # noqa: E402

df_thc_sga.dtype = jnp.float64
from dfthc.train_df_thc_sga import Initialize  # noqa: E402
from scipy.optimize import minimize  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('structure', type=Path)
    ap.add_argument('out', type=Path)
    ap.add_argument('--iters', type=int, default=3000)
    ap.add_argument('--kappa', type=float, default=2e-5)
    a = ap.parse_args()
    ref = Initialize(str(FCIDUMP), UPPER_ENERGY)
    ref = {k: (jnp.asarray(v, jnp.float64) if hasattr(v, 'dtype') and jnp.issubdtype(v.dtype, jnp.floating) else v)
           for k, v in ref.items()}
    with np.load(SHIFTS) as z:
        frozen = {k: jnp.asarray(z[k], jnp.float64) for k in ('Hsym_lower', 'H1_I')}
    with np.load(a.structure) as z:
        U0, w0 = np.asarray(z['U'], np.float64), np.asarray(z['W'], np.float64)
    shapes = (U0.shape, w0.shape)
    n_u = U0.size

    def parts(x):
        return x[:n_u].reshape(shapes[0]), x[n_u:].reshape(shapes[1])

    def objective(x):
        U, w = parts(x)
        resid = df_thc_sga.Compute_H2_residual_after_symmetry_shift(frozen, ref, df_thc_sga.Compute_H2_sqrt(U, w))
        eps = df_thc_sga.Compute_H2_norm(resid, ref['tril_diag'])
        D = 0.5 * jnp.sum(jnp.sum(jnp.abs(w), axis=1) ** 2 - jnp.sum(w, axis=1) ** 2)
        return eps + a.kappa * D, (eps, D)

    vg = jax.jit(jax.value_and_grad(objective, has_aux=True))
    t0, it = time.time(), [0]

    def fun(x):
        (v, (eps, D)), g = vg(jnp.asarray(x))
        it[0] += 1
        if it[0] % 200 == 0:
            print(f'eval {it[0]:5d}: error_2 {1e3 * float(eps):8.3f} mHa  D {float(D):7.2f}  ({time.time() - t0:.0f} s)',
                  flush=True)
        return float(v), np.asarray(g, np.float64)

    x0 = np.concatenate([U0.ravel(), w0.ravel()])
    res = minimize(fun, x0, jac=True, method='L-BFGS-B',
                   options={'maxiter': a.iters, 'maxfun': 2 * a.iters, 'maxcor': 30, 'ftol': 1e-15, 'gtol': 1e-12})
    U, w = parts(res.x)
    (_, (eps, D)), _ = vg(jnp.asarray(res.x))
    print(f'done after {res.nit} iterations: error_2 {1e3 * float(eps):.3f} mHa, D {float(D):.2f}', flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(a.out, U=U, W=w, H2_I=w.sum(axis=1))
    print('saved', a.out)


if __name__ == '__main__':
    main()
