"""Lower error_2 after rounding by exact coordinate descent on the two-body words.

Each move changes one 10-bit Givens word by +-1 (staying in the signed gauge), i.e. one vector u_b.
With w fixed, error_2^2 = F - 2 L + ||T||^2 (see refit_weights.py) changes exactly by

    dF = 4 sum_d delta_d G_db + 2 (M_bb delta^T M delta + (M_b . delta)^2)       G = M K M
    dL = 2 sum_{d != b} M_bd (q'_d - q_d) + M_bb (q'_bb - q_bb)                    q' = <P_b', T P_d>

where M = blockdiag(w_r w_r^T), K_bd = (u_b . u_d)^2 and delta = K'_b - K_b. All 150 moves of one
vector are scored with one product T [P'_1 ... P'_150]; the best improving move is applied, K, Q
and G are updated, and the sweep continues. After each sweep w is refit on error_2 alone.
(phucthaiv02 reports exact word-level coordinate descent after rounding; this is our own version.)

    python notebooks/dfthc-refit/polish_words.py ROUNDED.npz OUT.npz [--sweeps 3]

ROUNDED.npz is quantize.py output (`words` (R*B, N-1), `w` (R, B, C)); OUT.npz has the same keys.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parent))
import encode  # noqa: E402  (puts the verifiers on sys.path)
from paths import SHIFTS  # noqa: E402
from refit_weights import error_terms  # noqa: E402
from dfthcsos.scoreinfra.reference import check_fcidump, load_reference  # noqa: E402
from dfthcsos.verify import native_inputs  # noqa: E402



class Polisher:
    def __init__(self, words: np.ndarray, w: np.ndarray, T: np.ndarray, edges: np.ndarray, steps=(-1, 1), brot: int = 10):
        self.R, self.B, self.C = w.shape
        # signed-gauge words as integers: theta/pi * 2**(brot-1) in [-2**(brot-2), 2**(brot-2))
        self.brot, self.lo, self.hi = brot, -(1 << (brot - 2)), (1 << (brot - 2)) - 1
        self.steps = steps
        self.edges, self.T = edges, T
        self.k = np.where(words > self.hi, words - (1 << brot), words).astype(np.int64)   # signed integers
        self.u = self.decode(self.k)
        self.set_weights(w)

    def decode(self, k: np.ndarray) -> np.ndarray:
        v = encode.decode(k % (1 << self.brot), self.edges, self.brot)
        return v / np.linalg.norm(v, axis=1, keepdims=True)

    def set_weights(self, w: np.ndarray) -> None:
        R, B = self.R, self.B
        self.w = w
        self.M = np.zeros((R * B, R * B))
        for r in range(R):
            self.M[r * B:(r + 1) * B, r * B:(r + 1) * B] = w[r] @ w[r].T
        self.K = (self.u @ self.u.T) ** 2
        P = np.einsum('mp,mq->mpq', self.u, self.u).reshape(R * B, -1)
        TP = self.T @ P.T
        self.Q = np.zeros((R * B, R * B))
        for r in range(R):
            s = slice(r * B, (r + 1) * B)
            self.Q[s, s] = P[s] @ TP[:, s]
        self.G = self.M @ self.K @ self.M
        self.T2 = float(np.sum(self.T ** 2))

    def error2(self) -> float:
        F = np.sum(self.M * (self.K @ self.M @ self.K))       # tr(M K M K)
        return float(F - 2 * np.sum(self.M * self.Q) + self.T2)

    def best_move(self, b: int):
        """The best +-1 move of vector b's words: (change of error_2^2, angle, step), or None."""
        B = self.B
        r = b // B
        s = slice(r * B, (r + 1) * B)
        cands, moves = [], []
        for t in range(self.k.shape[1]):
            for step in self.steps:
                v = self.k[b, t] + step
                if self.lo <= v <= self.hi:
                    row = self.k[b].copy()
                    row[t] = v
                    cands.append(row)
                    moves.append((t, step))
        un = self.decode(np.array(cands))                        # (n, N)
        delta = (un @ self.u.T) ** 2 - self.K[b]                 # (n, R B), new K row minus old
        delta[:, b] = 0.0                                        # K_bb = 1 stays
        Mb = self.M[b]
        dMd = np.einsum('nd,nd->n', delta @ self.M, delta)      # delta^T M delta, over all groups
        dF = 4 * delta @ self.G[:, b] + 2 * (Mb[b] * dMd + (delta @ Mb) ** 2)
        Pn = np.einsum('np,nq->npq', un, un).reshape(len(un), -1)
        Y = self.T @ Pn.T                                        # (N^2, n)
        Pg = np.einsum('mp,mq->mpq', self.u[s], self.u[s]).reshape(B, -1)
        qn = (Pg @ Y).T                                          # <P_d, T P'_n> for d in the group
        qnn = np.einsum('nk,kn->n', Pn, Y)                       # <P'_n, T P'_n>
        j = b - r * B
        off = np.arange(B) != j
        dL = 2 * (qn[:, off] - self.Q[b, s][off]) @ Mb[s][off] + Mb[b] * (qnn - self.Q[b, b])
        gain = dF - 2 * dL
        i = int(np.argmin(gain))
        return (gain[i], *moves[i], un[i], qn[i], qnn[i]) if gain[i] < -1e-15 else None

    def apply(self, b: int, t: int, step: int, un, qn, qnn) -> None:
        B = self.B
        r = b // B
        s = slice(r * B, (r + 1) * B)
        self.k[b, t] += step
        new = (un @ self.u.T) ** 2
        new[b] = 1.0
        delta = new - self.K[b]
        self.u[b] = un
        self.K[b], self.K[:, b] = new, new
        q = self.Q[b, s].copy()
        q[:] = qn
        q[b - r * B] = qnn
        self.Q[b, s], self.Q[s, b] = q, q
        # K' = K + e_b delta^T + delta e_b^T (delta_b = 0), so G' = G + m (M delta)^T + (M delta) m^T, m = M e_b.
        Md = self.M @ delta
        self.G += np.outer(self.M[:, b], Md) + np.outer(Md, self.M[:, b])


def refit_w(p: Polisher, iters: int = 2000) -> np.ndarray:
    """Minimize error_2 over w alone (L-BFGS on the exact formula of refit_weights.py)."""
    R, B, C = p.R, p.B, p.C
    K4 = p.K.reshape(R, B, R, B)
    Qb = np.stack([p.Q[r * B:(r + 1) * B, r * B:(r + 1) * B] for r in range(R)])
    sc = np.std(p.w)

    def f(y):
        w = y.reshape(R, B, C) * sc
        A = np.einsum('rbc,rbsd,sdk->rcsk', w, K4, w, optimize=True)
        E = np.sum(A ** 2) - 2 * np.einsum('rbc,rbd,rdc->', w, Qb, w, optimize=True) + p.T2
        g = 4 * np.einsum('rbsd,sdk,rcsk->rbc', K4, w, A, optimize=True) - 4 * np.einsum('rbd,rdc->rbc', Qb, w)
        return E, (g * sc).ravel()

    res = minimize(f, (p.w / sc).ravel(), jac=True, method='L-BFGS-B', options={'maxiter': iters, 'ftol': 1e-16, 'gtol': 1e-14})
    return res.x.reshape(R, B, C) * sc


def polish(words: np.ndarray, w: np.ndarray, sweeps: int = 4, log=print, steps=(-1, 1), brot: int = 10,
           S: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Sweeps of exact word moves (by `steps`), each followed by a w refit; returns the new (words, w).

    `S` is the one-body shift of the target (default: our own, from `paths.SHIFTS`)."""
    R, B, C = w.shape
    edges = encode.log_depth_tree()
    N = edges.shape[0] + 1
    ref = load_reference(check_fcidump())
    u0 = encode.decode(words, edges, brot).reshape(R, B, N)
    if S is None:
        with np.load(SHIFTS) as z:
            _, _, S, _ = native_inputs({'U': u0, 'W': w, 'H2_I': w.sum(axis=1), 'H1_I': z['H1_I'],
                                        'Hsym_lower': z['Hsym_lower']})
    T = np.asarray(ref['tei'], np.float64).reshape(N * N, N * N).copy()
    eye = np.eye(N).ravel()
    T += 0.5 * (np.outer(S.ravel(), eye) + np.outer(eye, S.ravel()))
    p = Polisher(words, w, T, edges, steps, brot)
    _, _, T2check = error_terms(u0 / np.linalg.norm(u0, axis=2, keepdims=True), np.asarray(ref['tei'], np.float64), S)
    assert abs(T2check - p.T2) < 1e-9 * T2check
    log(f'start: error_2 {1e3 * np.sqrt(p.error2()):.4f} mHa')
    for sweep in range(sweeps):
        t0, moved = time.time(), 0
        for b in np.random.default_rng(sweep).permutation(R * B):
            m = p.best_move(int(b))
            if m is not None:
                p.apply(int(b), *m[1:])
                moved += 1
        e_moves = np.sqrt(p.error2())
        p.set_weights(refit_w(p))
        log(f'sweep {sweep}: {moved} moves, error_2 {1e3 * e_moves:.4f} mHa, after the w refit '
            f'{1e3 * np.sqrt(p.error2()):.4f} mHa ({time.time() - t0:.0f} s)')
        if moved == 0:
            break
    return p.k % (1 << brot), p.w


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('rounded', type=Path)
    ap.add_argument('out', type=Path)
    ap.add_argument('--sweeps', type=int, default=3)
    ap.add_argument('--steps', type=int, nargs='+', default=[-1, 1], help='word moves to try (e.g. -2 -1 1 2)')
    ap.add_argument('--brot', type=int, default=10)
    a = ap.parse_args()
    with np.load(a.rounded) as z:
        words, w = np.asarray(z['words'], np.int64), np.asarray(z['w'], np.float64)
    words, w = polish(words, w, a.sweeps, log=lambda *x: print(*x, flush=True), steps=tuple(a.steps), brot=a.brot)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(a.out, words=words, w=w, brot=a.brot)
    print('saved', a.out)


if __name__ == '__main__':
    main()
