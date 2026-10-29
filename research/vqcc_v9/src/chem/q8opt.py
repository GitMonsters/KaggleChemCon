"""8-bit DF-THC quantization research: continuous refits, exact per-group K refits, discrete word search.

State: integer words K (R, B, 75) in signed form (theta/pi = k / 128 at brot 8) and weights W (R, B, C).
The two-body error depends on W only through G_r = W_r W_r^T, so between discrete sweeps every group gets
the exact least-squares PSD rank-C update; the column gauge of W (which sets lambda_eff) is chosen later.
"""
from __future__ import annotations

import math
import time

import numpy as np
import torch

import q8lib as L

N = 76
P = N * (N + 1) // 2


class Problem:
    def __init__(self, edges, brot=8, dev='cuda'):
        self.dev = dev
        self.brot = brot
        self.half = 1 << (brot - 1)
        self.tb64 = L.TwoBody(dev, torch.float64)
        self.T32 = self.tb64.T.float()
        self.pair64 = self.tb64.pair
        self.pair32 = L.Pair(dev, torch.float32)
        self.tree64 = L.Tree(edges, dev, torch.float64)
        self.tree32 = L.Tree(edges, dev, torch.float32)
        self.edges = np.asarray(edges)

    # ---------------------------------------------------------------- exact evaluation
    def vectors(self, K, dtype=torch.float64):
        tree = self.tree64 if dtype == torch.float64 else self.tree32
        th = torch.as_tensor(K, device=self.dev).to(dtype) / self.half
        return tree(th)

    def err2(self, U, W):
        """Official error_2 (float64)."""
        return self.tb64.err(U.double(), W.double()).item()

    # ---------------------------------------------------------------- K (= W W^T) refit
    def refit_groups(self, U, W, C=None, sweeps=2, ridge=0.0):
        """Exact block coordinate least squares: for each group r, G_r = argmin ||sum_s P_s G_s P_s^T - T||
        over PSD rank-C; returns W with columns sqrt(eig) * eigvec (gauge fixed later)."""
        R, B, _ = U.shape
        C = W.shape[2] if C is None else C
        Pm = self.pair64(U.double())                                  # (R, B, P)
        Ws = [W[r].double() for r in range(R)]
        H = [Pm[r].T @ Ws[r] for r in range(R)]                     # x vectors (P, C)
        for _ in range(sweeps):
            for r in range(R):
                Y = self.tb64.T.clone()
                for s in range(R):
                    if s != r:
                        Y -= H[s] @ H[s].T
                Q, Rt = torch.linalg.qr(Pm[r].T)                    # P_r^T = Q Rt, Q (P, B)
                M = Q.T @ Y @ Q
                M = 0.5 * (M + M.T)
                ev, V = torch.linalg.eigh(M)
                ev = ev[-C:].clamp(min=0.0)
                V = V[:, -C:]
                Kp = V * ev.sqrt()                                   # (B, C): Rt G Rt^T = Kp Kp^T
                if ridge:
                    Rt = Rt + ridge * torch.eye(B, device=self.dev, dtype=Rt.dtype)
                Wr = torch.linalg.solve_triangular(Rt, Kp, upper=True)   # G = Wr Wr^T
                Ws[r] = Wr
                H[r] = Pm[r].T @ Wr
        return torch.stack(Ws)

    def min_d_gauge(self, W, steps=3000, lr=2e-3, eps=1e-8):
        """Column gauge W_r -> W_r O_r (O_r orthogonal) minimizing the sign-mixing penalty D (K unchanged)."""
        R, B, C = W.shape
        Wf = W.float()
        A = torch.zeros(R, C, C, device=self.dev, requires_grad=True)
        opt = torch.optim.Adam([A], lr=lr)
        best = (float('inf'), W)
        for s in range(steps + 1):
            O = torch.linalg.matrix_exp(A - A.transpose(1, 2))
            Wr = torch.bmm(Wf, O)
            a = torch.sqrt(Wr * Wr + eps).sum(1)
            sm = Wr.sum(1)
            D = 0.5 * (a * a - sm * sm).sum()
            if s % 100 == 0 and D.item() < best[0]:
                best = (D.item(), Wr.detach().double())
            if s == steps:
                break
            opt.zero_grad(set_to_none=True)
            D.backward()
            opt.step()
        return best[1]

    def fit_w(self, U, W, steps=300, lr=5e-5, kappa=2e-5):
        """lambda-aware weight refit: Adam on W (vectors fixed), objective err + kappa * D(W)."""
        v = self.pair32(U.float()).detach()
        w = W.detach().float().clone().requires_grad_(True)
        opt = torch.optim.Adam([w], lr=lr)
        best = (float('inf'), w.detach().clone())
        for s in range(steps + 1):
            a = torch.einsum('rbm,rbc->rcm', v, w).reshape(-1, P)
            e = a.T @ a - self.T32
            err = torch.sqrt((e * e).sum())
            tot = err + kappa * L.dpen(w)
            if s % 20 == 0 and tot.item() < best[0]:
                best = (tot.item(), w.detach().clone())
            if s == steps:
                break
            opt.zero_grad(set_to_none=True)
            tot.backward()
            opt.step()
        return best[1].double()

    # ---------------------------------------------------------------- continuous refit
    def fit_continuous(self, theta, W, steps, lr_th=3e-4, lr_w=2e-4, kappa=1e-4, log=print, every=2000):
        """Adam on continuous angles (units of pi) and weights; objective err + kappa * D."""
        th = theta.detach().float().clone().requires_grad_(True)
        w = W.detach().float().clone().requires_grad_(True)
        opt = torch.optim.Adam([{'params': [th], 'lr': lr_th}, {'params': [w], 'lr': lr_w}])
        R, B, C = w.shape
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps, eta_min=0.02 * lr_w)
        best = (float('inf'), None, None)
        t0 = time.time()
        for s in range(steps + 1):
            U = self.tree32(th)
            v = self.pair32(U)
            a = torch.einsum('rbm,rbc->rcm', v, w).reshape(-1, P)
            e = a.T @ a - self.T32
            err = torch.sqrt((e * e).sum())
            dp = L.dpen(w)
            tot = err + kappa * dp
            if s % 100 == 0 and tot.item() < best[0]:
                best = (tot.item(), th.detach().clone(), w.detach().clone())
            if s % every == 0 or s == steps:
                log(f'  cont step {s:6d} err {1e3 * err.item():8.3f} mHa D {dp.item():8.2f} ({time.time() - t0:.0f}s)')
            if s == steps:
                break
            opt.zero_grad(set_to_none=True)
            tot.backward()
            opt.step()
            sched.step()
        return best[1], best[2]

    # ---------------------------------------------------------------- annealed quantization-aware fit
    def anneal(self, theta, W, steps, mu0=1e-6, mu1=1e-2, kappa=2e-5, lr_th=1e-4, lr_w=5e-5, log=print, every=2000,
               weight_by_sensitivity=True):
        """Adam on (theta, W) with a growing periodic penalty mu(t) * mean s_k sin^2(pi * half * theta_k)
        pulling angles to the 8-bit grid; s_k (optional) = |du/dtheta_k|^2 so cheap angles snap first."""
        half = float(self.half)
        th = theta.detach().float().clone().requires_grad_(True)
        w = W.detach().float().clone().requires_grad_(True)
        opt = torch.optim.Adam([{'params': [th], 'lr': lr_th}, {'params': [w], 'lr': lr_w}])
        t0 = time.time()
        for s in range(steps + 1):
            frac = s / max(1, steps)
            mu = mu0 * (mu1 / mu0) ** frac
            U = self.tree32(th)
            v = self.pair32(U)
            a = torch.einsum('rbm,rbc->rcm', v, w).reshape(-1, P)
            e = a.T @ a - self.T32
            err = torch.sqrt((e * e).sum())
            pen = torch.sin(math.pi * half * th) ** 2
            qpen = pen.mean()
            tot = err + kappa * L.dpen(w) + mu * qpen
            if s % every == 0 or s == steps:
                dist = (th * half - torch.round(th * half)).abs()
                log(f'  anneal step {s:6d} err {1e3 * err.item():8.3f} mHa D {L.dpen(w).item():7.2f} mu {mu:.2e} '
                    f'grid dist mean {dist.mean().item():.3f} max {dist.max().item():.3f} ({time.time() - t0:.0f}s)')
            if s == steps:
                break
            opt.zero_grad(set_to_none=True)
            tot.backward()
            opt.step()
            with torch.no_grad():
                th.clamp_(-0.5, 0.5 - 1.0 / half)
        return th.detach(), w.detach()

    # ---------------------------------------------------------------- progressive quantization
    def progressive(self, theta, W, fracs, steps=1500, final_steps=3000, kappa=2e-5, probes=8,
                    lr_th=1e-4, lr_w=1e-4, log=print, high_first=True, ramp=0.0):
        """Freeze angles to the 8-bit grid in sensitivity order (highest impact first); between stages
        re-optimize the free angles and all weights (Adam), so the free part compensates."""
        half = float(self.half)
        lo, hi = -0.5, 0.5 - 1.0 / half
        theta = theta.detach().float().clone()
        W = W.detach().float().clone()
        frozen = torch.zeros_like(theta, dtype=torch.bool)
        qv = torch.zeros_like(theta)

        def qround(t):
            return torch.clamp(torch.round(t * half) / half, lo, hi)

        def loss(th, w):
            U = self.tree32(th)
            v = self.pair32(U)
            a = torch.einsum('rbm,rbc->rcm', v, w).reshape(-1, P)
            e = a.T @ a - self.T32
            return torch.sqrt((e * e).sum()), e

        for stage, frac in enumerate(list(fracs) + [None]):
            if frac is not None:
                need = int(frac * theta.numel()) - int(frozen.sum())
                if need > 0:
                    th = torch.where(frozen, qv, theta).clone().requires_grad_(True)
                    _, e = loss(th, W)
                    acc = torch.zeros_like(th)
                    for _ in range(probes):
                        zz = torch.randn_like(e)
                        g, = torch.autograd.grad((e * (zz + zz.T)).sum(), th, retain_graph=True)
                        acc += g * g
                    imp = acc * (qround(theta) - theta) ** 2
                    imp[frozen] = -1.0 if high_first else float('inf')
                    idx = torch.argsort(imp.flatten(), descending=high_first)[:need]
                    fl, q = frozen.flatten().clone(), qv.flatten().clone()
                    nw = torch.zeros_like(fl)
                    nw[idx] = True
                    fl[idx] = True
                    q[idx] = qround(theta).flatten()[idx]
                    frozen, qv = fl.view_as(frozen), q.view_as(qv)
                    newly = nw.view_as(frozen)
                    start = theta.clone()
                else:
                    newly = torch.zeros_like(frozen)
                    start = theta.clone()
            else:
                newly = torch.zeros_like(frozen)
                start = theta.clone()
            n = steps if frac is not None else final_steps
            thp = theta.clone().requires_grad_(True)
            wp = W.clone().requires_grad_(True)
            groups = [{'params': [wp], 'lr': lr_w}] + ([] if bool(frozen.all()) else [{'params': [thp], 'lr': lr_th}])
            opt = torch.optim.Adam(groups)
            ff = frozen.float()
            best = (float('inf'), theta, W, float('inf'))
            nn = newly.float()
            ramp_steps = int(ramp * n) if bool(newly.any()) else 0
            for s in range(n + 1):
                alpha = min(1.0, s / ramp_steps) if ramp_steps else 1.0
                target = qv * (1 - nn) + nn * (start + alpha * (qv - start))
                err, _ = loss(ff * target + (1 - ff) * thp, wp)
                tot = err + kappa * L.dpen(wp)
                if s % 25 == 0 and alpha >= 1.0 and tot.item() < best[0]:
                    best = (tot.item(), thp.detach().clone(), wp.detach().clone(), err.item())
                if s == n:
                    break
                opt.zero_grad(set_to_none=True)
                tot.backward()
                opt.step()
                with torch.no_grad():
                    thp.clamp_(lo, hi)
                for gr in opt.param_groups:
                    gr['lr'] *= 0.2 ** (1.0 / n)
            theta, W = best[1], best[2]
            log(f'  PQ stage {stage:2d} frozen {float(frozen.float().mean()):.3f} error_2 {1e3 * best[3]:8.3f} mHa D {L.dpen(W).item():.1f}')
        final = torch.where(frozen, qv, qround(theta))
        K = torch.round(final * half).long()
        return K, W.double()

    # ---------------------------------------------------------------- pruning
    def removal_cost(self, K, W):
        """Exact fixed-W Delta F of deleting each vector (its projector leaves every x_rc)."""
        R, B, C = W.shape
        Wf, p, X, E = self.state(K, W)
        Y = torch.einsum('rbc,rcm->rbm', Wf, X)
        om = (Wf * Wf).sum(-1)
        yy = (Y * Y).sum(-1)
        G = (Y.reshape(-1, P) @ E).reshape(R, B, P)
        d = -p
        t1 = 4.0 * (d * G).sum(-1)
        n2 = (d * d).sum(-1)
        dy = (d * Y).sum(-1)
        Ed = (d.reshape(-1, P) @ E).reshape(R, B, P)
        t2 = 2.0 * om * (Ed * d).sum(-1)
        return t1 + t2 + 2.0 * n2 * yy + 2.0 * dy * dy + 4.0 * om * dy * n2 + om * om * n2 * n2

    def prune_vectors(self, K, W, m):
        """Remove the m cheapest vectors from every group (by exact fixed-W removal cost)."""
        cost = self.removal_cost(K, W)
        R, B, C = W.shape
        keep = torch.argsort(cost, dim=1)[:, m:].sort(dim=1).values      # drop the m smallest costs
        K2 = torch.gather(K, 1, keep[:, :, None].expand(R, B - m, K.shape[2]))
        W2 = torch.gather(W, 1, keep[:, :, None].expand(R, B - m, C))
        return K2.contiguous(), W2.contiguous()

    def prune_columns(self, W, m, H2=None):
        """Drop the m columns with the smallest weight energy in every group (current gauge)."""
        R, B, C = W.shape
        energy = (W * W).sum(1)                                           # (R, C)
        keep = torch.argsort(energy, dim=1)[:, m:].sort(dim=1).values
        W2 = torch.gather(W, 2, keep[:, None, :].expand(R, B, C - m))
        H22 = None if H2 is None else torch.gather(H2, 1, keep)
        return W2.contiguous(), H22

    # ---------------------------------------------------------------- parallel discrete steps
    def state(self, K, W):
        Wf = W.float()
        U = self.vectors(K, torch.float32)
        p = self.pair32(U)
        X = torch.einsum('rbm,rbc->rcm', p, Wf)
        Xf = X.reshape(-1, P)
        E = Xf.T @ Xf - self.T32
        return Wf, p, X, E

    def best_single_moves(self, K, W, steps=(-1, 1), exclude=None):
        """For every vector, the best single-word move and its predicted Delta F (fixed W, without the
        small E-curvature term).  Returns (gain (R,B), word index (R,B), delta (R,B))."""
        R, B, C = W.shape
        Wf, p, X, E = self.state(K, W)
        Y = torch.einsum('rbc,rcm->rbm', Wf, X)                     # (R, B, P)
        om = (Wf * Wf).sum(-1)                                       # (R, B)
        yy = (Y * Y).sum(-1)
        G = (Y.reshape(-1, P) @ E).reshape(R, B, P)
        lo, hi = -self.half // 2, self.half // 2 - 1
        gains = torch.full((R, B), float('inf'), device=self.dev)
        widx = torch.zeros((R, B), dtype=torch.long, device=self.dev)
        dsel = torch.zeros((R, B), dtype=torch.long, device=self.dev)
        nw = N - 1
        for r in range(R):
            for d in steps:
                cand = K[r][:, None, :].repeat(1, nw, 1)              # (B, 75, 75)
                ar = torch.arange(nw, device=self.dev)
                cand[:, ar, ar] += d
                ok = ((cand >= lo) & (cand <= hi)).all(-1)            # (B, 75)
                Uc = self.vectors(cand.reshape(-1, nw), torch.float32).reshape(B, nw, N)
                dP = self.pair32(Uc) - p[r][:, None, :]               # (B, 75, P)
                t1 = 4.0 * torch.einsum('bkm,bm->bk', dP, G[r])
                n2 = (dP * dP).sum(-1)
                dy = torch.einsum('bkm,bm->bk', dP, Y[r])
                o = om[r][:, None]
                dF = t1 + 2.0 * n2 * yy[r][:, None] + 2.0 * dy * dy + 4.0 * o * dy * n2 + o * o * n2 * n2
                dF = torch.where(ok, dF, torch.full_like(dF, float('inf')))
                if exclude is not None:
                    dF = torch.where(exclude[r], torch.full_like(dF, float('inf')), dF)
                m, j = dF.min(dim=1)
                better = m < gains[r]
                gains[r] = torch.where(better, m, gains[r])
                widx[r] = torch.where(better, j, widx[r])
                dsel[r] = torch.where(better, torch.full_like(j, d), dsel[r])
        return gains, widx, dsel

    def parallel_descent(self, K, W, iters, frac=0.25, refit_every=1, log=print, tag=''):
        """Steepest-descent style: apply the best single move of the top `frac` vectors at once,
        keep the step if the exact error (after K refit) drops, else halve frac."""
        K = K.clone()
        U = self.vectors(K)
        W = self.refit_groups(U, W, sweeps=2)
        e = self.err2(U, W)
        best = (e, K.clone(), W.clone())
        for it in range(iters):
            gains, widx, dsel = self.best_single_moves(K, W)
            flat = gains.flatten()
            neg = int((flat < 0).sum())
            if neg == 0:
                log(f'{tag} iter {it}: no predicted improvement')
                break
            k = max(1, int(frac * neg))
            sel = torch.topk(-flat, k).indices
            Kn = K.clone()
            R, B, _ = K.shape
            rr, bb = sel // B, sel % B
            Kn[rr, bb, widx.flatten()[sel]] += dsel.flatten()[sel]
            Un = self.vectors(Kn)
            Wn = self.refit_groups(Un, W, sweeps=1) if it % refit_every == 0 else W
            en = self.err2(Un, Wn)
            if en < e:
                K, W, e = Kn, Wn, en
                frac = min(1.0, frac * 1.25)
                if e < best[0]:
                    best = (e, K.clone(), W.clone())
            else:
                frac *= 0.5
            if it % 10 == 0:
                log(f'{tag} iter {it}: candidates {neg}, applied {k}, error_2 {1e3 * e:.4f} mHa, frac {frac:.3f}')
        return best

    # ---------------------------------------------------------------- discrete search
    def sweep(self, K, W, order=None, max_step=1, model_moves=True, temp=0.0, rng=None, log=None, inner=1):
        """One Gauss-Seidel pass over all two-body vectors with exact Delta F evaluation of
        all single-word moves (+-1..max_step); up to `inner` greedy moves per vector visit."""
        R, B, C = W.shape
        dev = self.dev
        Kt = torch.as_tensor(K, device=dev).clone()
        Wf = W.float()
        U = self.vectors(Kt, torch.float32)                          # (R, B, N)
        p = self.pair32(U)                                           # (R, B, P)
        X = torch.einsum('rbm,rbc->rcm', p, Wf)                      # (R, C, P)
        Xf = X.reshape(-1, P)
        E = Xf.T @ Xf - self.T32
        F0 = float((E.double() ** 2).sum())
        lo, hi = -self.half // 2, self.half // 2 - 1                # signed half-open gauge [-1/2, 1/2)
        steps = [d for d in range(-max_step, max_step + 1) if d]
        accepted = 0
        gain = 0.0
        if order is None:
            order = [(r, b) for r in range(R) for b in range(B)]
        eye = torch.eye(N - 1, device=dev, dtype=torch.int64)
        for (r, b) in order:
            wb = Wf[r, b]                                            # (C,)
            om = float((wb * wb).sum())
            for _ in range(inner):
                y = wb @ X[r]                                        # (P,)
                g = E @ y
                yy = float(y @ y)
                kb = Kt[r, b]
                cands = [kb[None, :] + d * eye for d in steps]
                cand = torch.cat(cands, 0)
                ok = ((cand >= lo) & (cand <= hi)).all(1)
                cand = cand[ok]
                Uc = self.vectors(cand, torch.float32)
                dP = self.pair32(Uc) - p[r, b][None, :]             # (K, P)
                t1 = 4.0 * (dP @ g)
                n2 = (dP * dP).sum(1)
                dy = dP @ y
                dF = t1 + 2.0 * n2 * yy + 2.0 * dy * dy + 4.0 * om * dy * n2 + om * om * n2 * n2
                top = torch.topk(-dF, min(4, dF.numel())).indices   # exact E-curvature term on a shortlist
                Ed = dP[top] @ E
                dF_top = dF[top] + 2.0 * om * (Ed * dP[top]).sum(1)
                jj = int(torch.argmin(dF_top))
                j = int(top[jj])
                best = float(dF_top[jj])
                take = best < 0
                if not take and temp > 0 and rng is not None:
                    take = rng.random() < math.exp(-best / temp)
                if not take:
                    break
                d = dP[j]
                Kt[r, b] = cand[j]
                p[r, b] += d
                X[r] += wb[:, None] * d[None, :]
                E += torch.outer(d, y) + torch.outer(y, d) + om * torch.outer(d, d)
                accepted += 1
                gain += best
        Xf = X.reshape(-1, P)
        E = Xf.T @ Xf - self.T32
        F1 = float((E.double() ** 2).sum())
        return Kt, accepted, math.sqrt(max(F0, 0)), math.sqrt(max(F1, 0))
