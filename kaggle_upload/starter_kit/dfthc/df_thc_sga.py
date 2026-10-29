import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike
from jax.random import normal
from jax.nn import relu

dtype = jnp.float32


# DF-THC tensor primitives.

def Compute_H1_lambda(H1: ArrayLike):
    H1_eigs =  jnp.linalg.eigvalsh(H1)
    return jnp.sum(jnp.abs(H1_eigs))

def Compute_H2_sqrt(U: ArrayLike, W: ArrayLike) -> Array:
    U_normalized = U / jnp.linalg.norm(U, ord = 2, axis = 2)[:,:,None]
    return jnp.einsum("tjp,tjq,tjl->tlpq", U_normalized, U_normalized, W)

def Compute_H2_lower(H2_sqrt: ArrayLike, ref) -> Array:
    H2_sqrt_l = H2_sqrt[:,:,ref['tril'][0],ref['tril'][1]]
    H2_l = jnp.einsum("tlp,tlq->pq", H2_sqrt_l, H2_sqrt_l)
    return H2_l

def Compute_H2_norm(H2: ArrayLike, tril_diag: ArrayLike) -> Array:
    """Return the full four-index Frobenius norm from packed pair indices."""
    delta = jnp.square(H2)
    return jnp.sqrt(4.0 * jnp.sum(delta) - 4.0 * jnp.sum(delta[tril_diag, :]) + jnp.sum(delta[:,tril_diag][tril_diag,:]))


def Compute_H2_norm_1(H2: ArrayLike, tril_diag: ArrayLike) -> Array:
    delta = jnp.abs(H2)
    return (4.0 * jnp.sum(delta) - 4.0 * jnp.sum(delta[tril_diag, :]) + jnp.sum(delta[:,tril_diag][tril_diag,:]))


# This compute a spin-free representation.
# Unlike the other SOS that work in the regular fermion representation, 
# this computes an SOS in the Majorana representation
# with generators G2spinfree = {I, G_pq = gamma_{p00} gamma_{q01}+gamma_{p10} gamma_{q11}}.
# So the two-body terms are sum_{rc} (d^{(rc)} + b^{(rc)}_pq G_pq)^2 = (d^{(rc)} - Tr[b^{(rc)}] + b^{(rc)}_pq E_pq)^2,
# where params['H2_I'][r,c] = d^{(rc)}
# where b^{(rc)}_pq = sum_b U^{(rb)}_{p} W^{(rbc)} U^{(rb)}_q
# where params['U'][r,b,p] = U^{(rb)}_{p}
# where params['W'][r,b,c] = W^{(rbc)}
#
# G2spinfree generates one-body terms \sum_{rc} 2(d^{(rc)} - Tr[b^{(rc)}]) b^{(rc)}_pq E_pq)
# In the exact limit, -2Tr[b^{(rc)}] b^{(rc)}_pq E_pq = -h_{rrpq} E_pq.
# Hence, the one-body term from G2spinfree is 2d^{(rc)}b^{(rc)}_pq E_pq-h_{rrpq} E_pq.


def H1_from_lower(H1_lower: ArrayLike, ref) -> Array:
    H1 = jnp.zeros((ref['orbitals'], ref['orbitals'])).astype(dtype)
    H1 = H1.at[ref['tril'][0],ref['tril'][1]].set(H1_lower)
    H1 = H1 + jnp.transpose(H1,(1,0)) - jnp.diag(jnp.diag(H1))
    return H1


def Compute_H2_residual_after_symmetry_shift(params, ref, H2_sqrt: ArrayLike) -> Array:
    """Return the Eq. (40) residual against the input two-electron tensor.

    For FCIDUMP training, ``ref['H2_lower']`` contains the original FCIDUMP
    integrals.  The DFTHC tensor approximates the symmetry-shifted tensor in
    Eq. (32), so subtract the shift before comparing in the original integral
    frame.  The returned residual is the negative of the one displayed in
    Eq. (40), which has the same Frobenius norm.
    """
    h2_reconstructed_lower = Compute_H2_lower(H2_sqrt, ref)
    minus_half_S_lower = -0.5 * params['Hsym_lower']
    h2_reconstructed_lower = h2_reconstructed_lower.at[
        ref['tril_diag'], :
    ].add(minus_half_S_lower[None, :])
    h2_reconstructed_lower = h2_reconstructed_lower.at[
        :, ref['tril_diag']
    ].add(minus_half_S_lower[:, None])
    return h2_reconstructed_lower - ref['H2_lower']


def Compute_H0_shifted_by_Hsym_lower(params, ref) -> Array:
    H0 = (ref['ecore_df']
          + (ref['orbitals'] - ref['electrons']) * params['H1_I']
          - 0.5 * jnp.sum(jnp.square(params['H2_I'])))
    return H0


def Compute_df_thc_sga_lambda_H2(params) -> float:
    return 0.25 * jnp.sum(
        jnp.square(
            jnp.linalg.norm(params['W'], ord=1, axis=1)
            + jnp.abs(params['H2_I'])
        )
    )


def Compute_qaggle_df_thc_cost_function(
    params,
    ref,
    rho: float,
    method: str,
    **kwargs,
):
    """Revised Qaggle Eqs. (32)--(48), using the archive coordinates.

    ``Generate_params`` stores lowercase ``w`` as ``params['W']`` and stores
    ``w_B`` as ``params['H2_I']``.  The conversion below recovers the paper's
    identity offset ``W`` and scalar ``beta`` without changing the established
    parameter format used by the training and archive code.

    Finite-bit one-body error is deliberately evaluated after ANGLE/QROM
    decoding in ``dfthc.verify_angle_k``. This differentiable objective
    has no encoded angle words or decoded one-body vectors; rebuilding its
    analytically defined ``h1_prime`` from its own eigendecomposition would
    only add an identically zero round-trip term.
    """
    del method
    epsilon_dfthc = kwargs.get('epsilon_dfthc', 0.020)

    w_b_rho_c = params['W']
    H2_sqrt = Compute_H2_sqrt(params['U'], w_b_rho_c)

    # Eq. (40): undo the symmetry shift, compare with the original FCIDUMP
    # two-electron integrals, and restore the multiplicities hidden by packed
    # pair indices to obtain the full four-index Frobenius norm.
    h2_residual_lower = Compute_H2_residual_after_symmetry_shift(
        params, ref, H2_sqrt
    )
    epsilon_fro = Compute_H2_norm(
        h2_residual_lower, ref['tril_diag']
    )
    h2_error_1 = Compute_H2_norm_1(
        h2_residual_lower, ref['tril_diag']
    )

    # Convert the established archive coordinates to the notation in the
    # revised specification: w_B = W + sum_b w_b and
    # beta = H1_I - Tr(S) / 2.
    S = H1_from_lower(params['Hsym_lower'], ref)
    W_rho_c = params['H2_I'] - jnp.sum(w_b_rho_c, axis=1)
    beta = params['H1_I'] - 0.5 * jnp.trace(S)

    # Eq. (38): both symmetry-shift variables enter h1_prime.
    h1_prime = (
        ref['H1_THC']
        - 0.5 * ref['electrons'] * S
        + beta * jnp.eye(ref['orbitals'], dtype=dtype)
        - jnp.einsum('rc,rcpq->pq', W_rho_c, H2_sqrt)
    )

    # Eqs. (41)--(42).
    lambda_H1 = Compute_H1_lambda(h1_prime)
    lambda_H2 = Compute_df_thc_sga_lambda_H2(params)
    Lambda = lambda_H1 + lambda_H2

    # Eqs. (43)--(45) must likewise use beta and W directly.
    # Compute_H0_shifted_by_Hsym_lower is equivalent only at zero residual.
    # Include the FCIDUMP scalar because energy_upper is a total energy,
    # whereas the paper suppresses that scalar in Eq. (31).
    E_SOS = (
        ref['ecore']
        - ref['electrons'] * beta
        - (lambda_H1 - jnp.trace(h1_prime))
        - 0.5 * jnp.sum(jnp.square(W_rho_c))
    )
    SOS_gap = ref['energy_upper'] - E_SOS

    # Eq. (45) requires 0 <= E_gap <= 2 Lambda.  The supplied upper energy
    # does not guarantee that interval while the factorization error is still
    # large, so extend Eq. (47) outside its physical domain with an explicit
    # distance-to-domain penalty.  Clipping is used only inside the square
    # root; the penalty supplies the gradient that returns an invalid iterate
    # to the physical interval.
    gap_below_domain = relu(-SOS_gap)
    gap_above_domain = relu(SOS_gap - 2.0 * Lambda)
    gap_domain_violation = gap_below_domain + gap_above_domain
    gap_for_amplification = jnp.clip(
        SOS_gap, min=0.0, max=2.0 * Lambda
    )
    lambda_eff_squared = gap_for_amplification * (
        2.0 * Lambda - gap_for_amplification
    )
    lambda_eff = jnp.sqrt(jnp.maximum(lambda_eff_squared, 1.0e-12))

    # Eq. (48), in the exact-penalty form already used by the shared trainer.
    # The optional floors are the competition verifier's acceptance floors on
    # raw Lambda and the SOS gap (dfthcsos/verify.py). These soft penalties
    # guide optimization; the verifier enforces strict > at both boundaries.
    lambda_floor = kwargs.get('lambda_floor')
    sos_gap_floor = kwargs.get('sos_gap_floor')
    floor_violation = 0.0
    if lambda_floor is not None:
        floor_violation = floor_violation + relu(lambda_floor - Lambda)
    if sos_gap_floor is not None:
        floor_violation = floor_violation + relu(sos_gap_floor - SOS_gap)
    cost = (
        relu(epsilon_fro - epsilon_dfthc)
        + gap_domain_violation
        + floor_violation
        + rho * lambda_eff
    )

    return {
        'lambda_eff': lambda_eff,
        'lambda_H1': lambda_H1,
        'lambda_H2': lambda_H2,
        'Lambda': Lambda,
        'Spectral norm above zero': Lambda,
        'E_SOS': E_SOS,
        'epsilon_fro': epsilon_fro,
        'error_h2_ord=1': h2_error_1,
        'sos_gap': SOS_gap,
        'gap_domain_violation': gap_domain_violation,
        'floor_violation': floor_violation,
        'cost': cost,
    }


def Generate_params(rank: int, bases: int, copies: int, orbitals: int, seed: int = 0):
    key_final = jax.random.key(seed)
    params = {'W': 0.01 * normal(key_final,[rank, bases, copies] , dtype = dtype),
          'U': 1.0 * normal(key_final,[rank, bases, orbitals] , dtype = dtype),
          'Hsym_lower': 0.01 * normal(key_final, int(orbitals*(orbitals+1)/2), dtype = dtype),
          'H2_I': 0.01 * normal(key_final,[rank, copies] , dtype = dtype),   
          'H1_I': jnp.array(0.05, dtype = dtype)}
    return params
