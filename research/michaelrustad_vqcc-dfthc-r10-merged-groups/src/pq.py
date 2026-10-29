"""Progressive quantization of the two-body angles at a target brot, with continuous compensation (error_2 only).

Stage s freezes (rounds) the not-yet-frozen angles with the largest rounding impact, estimated as |g * d| + 0.5 * h * d^2
with g the gradient and h a Hutchinson estimate of the Hessian diagonal of error_2^2 (d = rounding offset). Then the
remaining continuous angles and w are re-optimized with Adam. Output: words2 + w, ready for flip.py / finish.py.

  python src/pq.py IN.npz OUT.npz --brot 7 --stages 12 --steps 400
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dft_eval as de  # noqa: E402
from qat import Problem, decode, dev, DT, N  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inp")
    ap.add_argument("out")
    ap.add_argument("--brot", type=int, default=7)
    ap.add_argument("--stages", type=int, default=12)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--lr-th", type=float, default=3e-4)
    ap.add_argument("--lr-w", type=float, default=3e-5)
    ap.add_argument("--probes", type=int, default=4)
    ap.add_argument("--fp32", action="store_true")
    ap.add_argument("--final-w", type=int, default=2000, help="Adam steps on w only after the last stage")
    ap.add_argument("--lbfgs", type=int, default=0, help="if >0: L-BFGS outer iterations per stage instead of Adam")
    ap.add_argument("--resume", default="", help="pq_lb state npz (frozen, qv, th, w) to continue from")
    ap.add_argument("--adaptive", action="store_true", help="freeze to floor or ceil by the local quadratic model")
    a = ap.parse_args()
    z = np.load(a.inp, allow_pickle=False)
    params = {k: np.array(z[k]) for k in z.files}
    edges = params["tree_edges"].astype(np.int64)
    w0, Wo0 = de.native(params)
    R, B, C = w0.shape
    P = Problem(edges)
    dt = torch.float32 if a.fp32 else DT
    Tp, cwt = P.Tp.to(dt), P.cw.to(dt)
    scale = float(1 << (a.brot - 1))
    th = torch.tensor(de.tree_angles(params["U"].reshape(-1, N), edges), dtype=dt, device=dev, requires_grad=True)
    w = torch.tensor(w0, dtype=dt, device=dev, requires_grad=True)
    frozen = torch.zeros_like(th, dtype=torch.bool)
    qv = torch.zeros_like(th)  # frozen grid values
    if a.resume:
        zs = np.load(a.resume)
        with torch.no_grad():
            frozen.copy_(torch.tensor(zs["frozen"], device=dev))
            qv.copy_(torch.tensor(zs["qv"], dtype=dt, device=dev))
            th.copy_(torch.tensor(zs["th"], dtype=dt, device=dev))
            w.copy_(torch.tensor(zs["w"], dtype=dt, device=dev))
    start_frozen = int(frozen.sum())

    def eff(t):
        return torch.where(frozen, qv, t)

    def e2fn(t, w_):
        v2 = decode(t, P.edges).reshape(R, B, N)
        M = (v2[:, :, P.il_t[0]] * v2[:, :, P.il_t[1]]) * cwt
        A = torch.einsum("rbc,rbp->rcp", w_, M).reshape(R * C, -1)
        return torch.linalg.norm(A.T @ A - Tp)

    total = th.numel()
    t0 = time.time()
    for s in range(1, a.stages + 1):
        target_frozen = start_frozen + int(round((total - start_frozen) * s / a.stages))
        n_new = target_frozen - int(frozen.sum())
        # impact estimate on the unfrozen angles
        t_eff = eff(th).detach().requires_grad_(True)
        f = e2fn(t_eff, w.detach()) ** 2
        g, = torch.autograd.grad(f, t_eff, create_graph=True)
        hd = torch.zeros_like(t_eff)
        for _ in range(a.probes):
            v = torch.randint(0, 2, t_eff.shape, device=dev).to(dt) * 2 - 1
            hv, = torch.autograd.grad(g, t_eff, grad_outputs=v, retain_graph=True)
            hd += v * hv / a.probes
        with torch.no_grad():
            qn = torch.floor(th * scale + 0.5) / scale
            d = qn - th
            impact = (g * d).abs() + 0.5 * hd.clamp(min=0) * d ** 2
            impact[frozen] = -1
            idx = torch.topk(impact.flatten(), n_new).indices
            frozen.view(-1)[idx] = True
            qsel = qn.view(-1)[idx]
            if a.adaptive:
                lo = torch.floor(th * scale).view(-1)[idx] / scale
                hi = lo + 1.0 / scale
                tv, gv, hv_ = th.view(-1)[idx], g.view(-1)[idx], hd.clamp(min=0).view(-1)[idx]
                m_lo = gv * (lo - tv) + 0.5 * hv_ * (lo - tv) ** 2
                m_hi = gv * (hi - tv) + 0.5 * hv_ * (hi - tv) ** 2
                qsel = torch.where(m_lo <= m_hi, lo, hi)
                print(f"  adaptive: {int((qsel != qn.view(-1)[idx]).sum())}/{len(idx)} away from nearest", flush=True)
            qv.view(-1)[idx] = qsel
        del g, hd, f
        if a.lbfgs:
            lb = torch.optim.LBFGS([th, w], lr=1, max_iter=20, history_size=50, line_search_fn="strong_wolfe")

            def closure():
                lb.zero_grad()
                e = e2fn(eff(th), w)
                e.backward()
                th.grad[frozen] = 0
                return e
            for _ in range(a.lbfgs if s < a.stages else 0):
                lb.step(closure)
        else:
            opt = torch.optim.Adam([{"params": [th], "lr": a.lr_th}, {"params": [w], "lr": a.lr_w}])
            for _ in range(a.steps if s < a.stages else 0):
                e = e2fn(eff(th), w)
                opt.zero_grad()
                e.backward()
                th.grad[frozen] = 0
                opt.step()
        with torch.no_grad():
            e = e2fn(eff(th), w).item()
        print(f"stage {s}/{a.stages}: frozen {int(frozen.sum())}/{total} e2 {1e3 * e:.3f} mHa {time.time() - t0:.0f}s",
              flush=True)
    if a.lbfgs:
        lbw = torch.optim.LBFGS([w], lr=1, max_iter=20, history_size=50, line_search_fn="strong_wolfe")

        def closure_w():
            lbw.zero_grad()
            e = e2fn(eff(th), w)
            e.backward()
            return e
        for i in range(max(1, a.final_w // 100)):
            e = lbw.step(closure_w)
        print(f"final w L-BFGS: e2 {1e3 * e2fn(eff(th), w).item():.3f} mHa", flush=True)
    else:
        optw = torch.optim.Adam([w], lr=a.lr_w)
        best_e, best_w = None, None
        for i in range(a.final_w):
            e = e2fn(eff(th), w)
            if best_e is None or e.item() < best_e:
                best_e, best_w = e.item(), w.detach().clone()
            optw.zero_grad()
            e.backward()
            optw.step()
            if i % 500 == 0:
                print(f"final w refit {i}: e2 {1e3 * e.item():.3f} mHa", flush=True)
        if best_w is not None:
            with torch.no_grad():
                w.copy_(best_w)
            print(f"kept best w: e2 {1e3 * best_e:.3f} mHa", flush=True)
    words2 = de.to_words(eff(th).detach().cpu().numpy().astype(np.float64), a.brot)
    wn = w.detach().cpu().numpy().astype(np.float64)
    out = dict(params)
    out.update(U=de.decode(words2, edges, a.brot).reshape(R, B, N), W=wn, H2_I=Wo0 + wn.sum(axis=1), words2=words2,
               brot=np.array(a.brot))
    np.savez(a.out, **out)
    print("saved", a.out)


if __name__ == "__main__":
    main()
