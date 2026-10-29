"""Monic-root conversion without the 12-qubit lead (history) word.

In SomeJoint's short-Hankel monic-root conversion the lead word only ever holds q1 = ai, the clipped
root inverse, which the root's pre/post actions already have in their own ai register; it is used
(a) as an operand in the root's 'm'/'c' ops -> use ai directly; (b) as the substitute pivot p in the
root child's q_post gap swap. That swap can only fire when the root itself was a gap chart (otherwise
the child's q1 = a_root * ai_child != 0); there p is replaced by the constant 1 (a cx on bit 0).
Same Toffoli count; the conversion's peak loses the 12 history qubits.
"""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from exp.convert_short_hankel import phase34 as P34

old = P34._private            # cloned stable converter namespace (34T phase), shared globals P34._scope
S = P34._scope
_orig_q_post = S['q_post']


def q_post(b, q, payloads, H, m, parent, reverse=False, phase=None):
    if parent is None or not isinstance(parent[1], tuple):
        return _orig_q_post(b, q, payloads, H, m, parent, reverse, phase)
    # Root child: the original swaps in p = lead = ai_root when q == 0 (root-gap chart only), then scales by
    # qi = 1/p = a_rot. a_rot already sits in the child's g[-1]: swap it into q instead, take qi = 1/a_rot
    # = ai_root, and swap (q, qi) around the scalings so the factor is again a_rot and the eraser ai_root.
    tops, (_, gl), c = parent
    F = S['fresh_zero_flag'](b, q)
    b.cswap(F, q, gl)
    if not reverse:
        for v, t in zip(payloads, tops): b.cswap(F, v[-1], t)
    with S['q_inverse'](b, q, m, reverse, phase) as qi:
        b.cswap(F, q, qi)
        if not reverse:
            for v in payloads: S['scale'](b, v[-1], q, qi, True)
            for v, t in zip(payloads, tops): S['masked_product'](b, F, v[-1], c, t)
        else:
            for v, t in zip(payloads, tops): S['masked_product'](b, F, v[-1], c, t)
            for v in payloads: S['scale'](b, v[-1], q, qi, False)
        b.cswap(F, q, qi)
    if reverse:
        for v, t in zip(payloads, tops): b.cswap(F, v[-1], t)
    b.cswap(F, q, gl)
    S['clear_flag'](b, q, F)


S['q_post'] = q_post         # cloned post_look/base resolve q_post through this namespace


def _pre_root(b, f, g, payloads, undo=False, incoming=None, defer=False):
    m = len(f) - 1; n = m - 1; a = g[-1]; q0 = f[-2]
    def action(ai):
        ops = [('s', v[-1], True) for v in payloads]
        ops += [('m', g[i], v[-1], v[i]) for v in payloads for i in range(n)]
        ops += [('m', ai, g[-2], q0), ('s', q0, True), ('c', g[:-1], [q0, ai], f[:-2])]
        old.runops(b, ops, a, ai, undo)
    return old.inverse_apply(b, a, action, incoming, defer)


def _post_root(b, f, g, payloads, reverse=False, incoming=None, defer=False):
    n = len(g) - 1; a = g[-1]; q0 = f[-2]
    def action(ai):
        ops = [('s', v[-1], False) for v in payloads]
        ops += [('m', g[i], v[i], v[-1]) for v in payloads for i in range(n)]
        ops += [('s', v[-1], True) for v in payloads]
        ops += [('c', g[:-1], [q0, ai], f[:-2]), ('s', q0, False), ('m', ai, g[-2], q0)]
        old.runops(b, ops, a, ai, reverse)
    return old.inverse_apply(b, a, action, incoming, defer)


def _look_root(b, f, g, payloads, reverse, phase):
    m = len(f) - 1; F = old.fresh_zero_flag(b, g[-1]); old.rotate(b, g, F)
    if not reverse: carry = _pre_root(b, f, g, payloads, defer=True)
    else: carry = _post_root(b, f, g, payloads, True, phase['a'].pop(m), True)
    old.canonical_permute(b, f, g, F); old.clear_flag(b, g[-1], F)
    parent = ([v[-1] for v in payloads], ('g', f[-3]), f[-2])
    old.look_solve(b, g, f[:-2], [v[:-1] for v in payloads], None, parent, reverse, phase)
    F = old.fresh_zero_flag(b, g[-1]); old.canonical_permute(b, f, g, F, True)
    if not reverse: phase['a'][m] = _post_root(b, f, g, payloads, False, carry, True)
    else: _pre_root(b, f, g, payloads, True, carry)
    old.rotate(b, g, F, True); old.clear_flag(b, g[-1], F)


def _bez_monic(b, O, E, payloads, reverse, phase):
    def reduce():
        for i in range(len(O)): old.ar.mul_xor0(b, E[-1], O[i], E[i])
    reduce(); _look_root(b, O + [None], E[:-1], payloads, reverse, phase); reduce()


def convert_safe(b, A, B, reverse=False, history_scratch12=None, phase_token=None, *, monic_root=True):
    assert len(A) == len(B) == 35 and history_scratch12 is None
    phase = {'a': {}, 'q': {}} if phase_token is None else phase_token
    assert not reverse or phase_token is not None
    old.frob(b, B, 2048); old.frob(b, A, 2048)
    E = B[::2]; O = B[1::2]; u = A[1::2]; w = A[:34:2]; top = A[-1]
    if not reverse:
        for i in range(17): old.ar.mul_xor0(b, top, O[i], w[i])
        _bez_monic(b, O, E, [u, w], False, phase)
        for i in range(17): old.ar.mul_xor0(b, E[i + 1], w[i], top)
        for i in range(17): b.swap(A[2 * i], A[2 * i + 1])
        old.hankel(b, B[1:], A[:34])
        for i in reversed(range(34)): b.swap(A[i], A[i + 1])
    else:
        for i in range(34): b.swap(A[i], A[i + 1])
        old.hankel(b, B[1:], A[:34], True)
        for i in range(17): b.swap(A[2 * i], A[2 * i + 1])
        for i in range(17): old.ar.mul_xor0(b, E[i + 1], w[i], top)
        _bez_monic(b, O, E, [u, w], True, phase)
        for i in range(17): old.ar.mul_xor0(b, top, O[i], w[i])
    old.frob(b, A, 2); old.frob(b, B, 2)
    if reverse: assert not phase['a'] and not phase['q']
    return phase if not reverse else None
