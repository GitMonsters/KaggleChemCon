"""Narrow-width jet pair and outer division for SomeJoint's fused DQI decoder (Q 889 -> 879 target).

forward_canonical_new: same map J(u,v)=(inv0 u, v*(inv0 u+[u=0])^2) and same 290T as the original,
  but old u is measured (deferred) right after z=inv0(u) exists and z is swapped into u's register
  before the v product, so the 24-qubit product target never coexists with z (49 -> 37 private).
  The old-v measurement phase is deferred too (token mv), repaid by the reverse once v is restored.
jet_reverse_new: closes everything without a 24-bit inverse phase. Norm N=N(u') is kept, v' is first
  multiplied by (conj u'+F)^2 (99T, 24 scratch while only N,F live), then scaled by t^2 (t=1/N, 2x42),
  its deferred phase repaid bilinearly while u' is still present, u' is inverted in place by t (2x42),
  and N is erased with the 34T classical-mask inverse phase on t. 418T, 37 private (original 396T, 49).
divide_zero_narrow: identity-at-zero division of the outer fold with the zero flag taken from an
  11T ladder on the norm, so the 24-qubit multiply scratch runs before the inverse (334T/37 vs 324T/49).
"""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from exp.jet24.kernel import inverse24_initialized_zero191, safe_square
from exp.mul99.kernels import mul_xor99_zero, phase_product_quantum99
from exp.mul99.schedules import multiply_forward99
from exp.noscratch.schedules import norm_xor_noscratch
from exp.noscratch.kernels import mul_xor_noscratch
from exp.inverse12_escape.joint import inverse_initialized
from exp.fold_cached_norm.phase34 import phase_inverse34
from exp.arithmetic.division import phase_product24
from exp.arithmetic import field as ar
from exp.arithmetic.field import erase_product, linear_inplace, inverse_columns, square_columns, tower, extension_columns
from exp.decoder.common import zero_hold, zero_free


def zero_flag12(b, x):
    """F=[x==0] for a 12-qubit word: 11T, ladder erased by measurement."""
    lad = zero_hold(b, x); F = b.alloc_qubits(1)[0]; b.cx(lad[-1], F); zero_free(b, x, lad)
    return F


def forward_canonical_new(b, u, v):
    z, F = inverse24_initialized_zero191(b, u)
    mu = b.alloc_bits(24); b.hmr(u, mu); b.swap(u, z); b.free(z)
    out = b.alloc_qubits(24)
    safe_square(b, u, F); mul_xor99_zero(b, v, u, out); safe_square(b, u, F, True)
    mv = b.alloc_bits(24); b.hmr(v, mv); b.swap(v, out); b.free(out)
    fz = b.alloc_bits(1); b.hmr(F, fz[0]); b.free(F)
    return mu, mv, fz


def jet_forward_new(b, u, v, K=None):
    cols, frm, _ = extension_columns(K); linear_inplace(b, u, cols); linear_inplace(b, v, cols)
    token = forward_canonical_new(b, u, v)
    linear_inplace(b, u, frm); linear_inplace(b, v, frm)
    return token


def _halves_scale(b, x, t, N):
    """x (tower coords) *= t in place; old halves erased with N=1/t (bilinear phase, 0T)."""
    for off in (0, 12):
        half = x[off:off + 12]; out = b.alloc_qubits(12)
        mul_xor_noscratch(b, half, t, out); erase_product(b, out, N, half)
        b.swap(half, out); b.free(out)


def jet_reverse_new(b, u, v, token, K=None):
    cols, frm, _ = extension_columns(K); linear_inplace(b, u, cols); linear_inplace(b, v, cols)
    mu, mv, fz = token
    to24, frm24, Kt = tower(24)
    sq24 = square_columns(24); isq24 = inverse_columns(sq24)
    sq12 = square_columns(12); isq12 = inverse_columns(sq12)
    # 1. N = N(u'), F = [u' = 0]
    linear_inplace(b, u, to24)
    N = b.alloc_qubits(12); norm_xor_noscratch(b, u, N, Kt)
    F = zero_flag12(b, N)
    # 2. v' <- v' * (conj u' + F)^2, old v' measured; phase repaid in step 5
    b.cx(u[12:], u[:12]); b.cx(F, u[0]); linear_inplace(b, u, frm24); linear_inplace(b, u, sq24)
    z = b.alloc_qubits(24); mul_xor99_zero(b, v, u, z)
    old = b.alloc_bits(24); b.hmr(v, old); b.swap(v, z); b.free(z)
    linear_inplace(b, u, isq24); linear_inplace(b, u, to24); b.cx(F, u[0]); b.cx(u[12:], u[:12])
    # 3. t = 1 / N_safe, N kept
    b.cx(F, N[0]); t = inverse_initialized(b, N)
    # 4. v *= t^2 (tower halves), erase with N^2
    linear_inplace(b, v, to24)
    linear_inplace(b, t, sq12); linear_inplace(b, N, sq12)
    _halves_scale(b, v, t, N)
    linear_inplace(b, t, isq12); linear_inplace(b, N, isq12)
    linear_inplace(b, v, frm24)
    # 5. repay old: v'_orig = v_final * (u'_safe)^2  (standard basis), u' still present
    b.cx(F, u[0]); linear_inplace(b, u, frm24); linear_inplace(b, u, sq24)
    phase_product_quantum99(b, v, u, old); b.free(old)
    linear_inplace(b, u, isq24); linear_inplace(b, u, to24); b.cx(F, u[0])
    # 6. u' -> conj(u'_safe) * t in place = inv0(u') after unsafe
    b.cx(F, u[0]); b.cx(u[12:], u[:12])
    _halves_scale(b, u, t, N)
    b.cx(F, u[0])
    # 7. erase N (= inv0 t_raw) and F (= [t_raw = 0]) with the 34T inverse phase; t_raw = N(u_new)
    b.cx(F, N[0]); b.cx(F, t[0])
    um = b.alloc_bits(12); b.hmr(N, um); b.free(N)
    fm = b.alloc_bits(1); b.hmr(F, fm[0]); b.free(F); b.cx(fz[0], fm[0]); b.free(fz)
    phase_inverse34(b, t, um, fm[0]); b.free(um); b.free(fm)
    ar._erase_norm(b, u, t, Kt); b.free(t)
    linear_inplace(b, u, frm24)
    # 8. forward's deferred phases, now linear in the restored u and v
    b.cz(u, mu); b.free(mu); b.cz(v, mv); b.free(mv)
    linear_inplace(b, u, frm); linear_inplace(b, v, frm)


def divide_zero_narrow(b, x, y, K=None, extra_zero_mask=None, after=None):
    """x /= y (y preserved), identity at y = 0; all HMR phases closed. 334T, 37 private."""
    assert K is not None
    N = b.alloc_qubits(12); norm_xor_noscratch(b, y, N, K)
    F = zero_flag12(b, N)
    b.cx(y[12:], y[:12]); b.cx(F, y[0]); old = multiply_forward99(b, x, y, K); b.cx(F, y[0]); b.cx(y[12:], y[:12])
    b.cx(F, N[0]); u = inverse_initialized(b, N)
    _halves_scale(b, x, u, N)
    b.cx(F, N[0]); b.cx(F, u[0])
    um = b.alloc_bits(12); b.hmr(u, um); b.free(u)
    b.cx(F, y[0]); phase_product24(b, x, y, old, K); b.cx(F, y[0]); b.free(old)
    if after is not None:
        after(F)
    fm = b.alloc_bits(1); b.hmr(F, fm[0]); b.free(F)
    if extra_zero_mask is not None:
        b.cx(extra_zero_mask, fm[0])
    phase_inverse34(b, N, um, fm[0]); b.free(um); b.free(fm)
    ar._erase_norm(b, y, N, K); b.free(N)


# ---------------------------------------------------------------- local in-place inverse, reverse
from exp.inverse12_escape.joint import inverse12_keep_inverse_norm64, erase_norm_via_inverse_cache19


def inverse24_reverse_narrow(b, x, mask, K=None):
    """inverse24_reverse209 with the 6-qubit norm(t) cache erased before the 12-qubit product scratch
    and recomputed (one GF64 product, 15T) for the final cleanup: 224T, 36 private (was 209T, 42)."""
    cols, back, _ = extension_columns(K); linear_inplace(b, x, cols); to, frm, K24 = tower(24); linear_inplace(b, x, to)
    N = b.alloc_qubits(12); norm_xor_noscratch(b, x, N, K24); t, u = inverse12_keep_inverse_norm64(b, N)
    to12, frm12, K12 = tower(12)
    linear_inplace(b, t, to12); ar._erase_norm(b, t, u, K12); b.free(u); linear_inplace(b, t, frm12)
    out = b.alloc_qubits(12)
    mul_xor_noscratch(b, x[12:], t, out); erase_product(b, out, N, x[12:]); mul_xor_noscratch(b, x[:12], t, x[12:]); b.cx(out, x[12:])
    b.cx(out, x[12:]); erase_product(b, x[12:], N, x[:12]); b.cx(out, x[12:]); b.swap(x[:12], x[12:]); b.swap(x[12:], out); b.free(out)
    u = b.alloc_qubits(6); linear_inplace(b, t, to12); ar._norm_xor(b, t, u, K12); linear_inplace(b, t, frm12)
    erase_norm_via_inverse_cache19(b, t, N, u); ar._erase_norm(b, x, t, K24); b.free(t); linear_inplace(b, x, frm); b.cz(x, mask); b.free(mask); linear_inplace(b, x, back)
