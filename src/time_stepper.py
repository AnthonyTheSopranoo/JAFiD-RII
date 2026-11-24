import jax
import jax.numpy as jnp

from poisson_solver import solve_psi_jit
from RHS import RHS, RHS_phi

phi_min = 1e-9


@jax.jit
def apply_C_BC(C):
    C = C.at[:, 0].set(1.0)
    C = C.at[:, -1].set(0.0)
    return C


@jax.jit
def apply_phi_BC(phi, phi0):
    # phi = phi.at[:, 0].set(phi1)
    phi = phi.at[:, -1].set(phi0)
    return phi


@jax.jit
def rk2_step(
    m: jax.Array,
    phi: jax.Array,
    psi: jax.Array,
    H: float,
    W: float,
    S: float,
    phi0: float,
    dt: float,
    dx: float | jax.Array,
    dy: float | jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    C = m / jnp.maximum(phi, phi_min)
    C = apply_C_BC(C)

    psi, _, _ = solve_psi_jit(phi, psi, dx, dy, 7)
    k1_m, k1_phi = RHS(phi, psi, C, H, W, dx, dy), RHS_phi(C, phi, W, S, dx)

    m2 = m + 0.5 * dt * k1_m
    phi2 = jnp.maximum(phi + 0.5 * dt * k1_phi, phi_min)
    phi2 = apply_phi_BC(phi2, phi0)
    C2 = apply_C_BC(m2 / phi2)

    psi, err, iters = solve_psi_jit(phi2, psi, dx, dy, 7)
    k2_m, k2_phi = RHS(phi2, psi, C2, H, W, dx, dy), RHS_phi(C2, phi2, W, S, dx)

    m_next = m + dt * k2_m
    phi_next = jnp.maximum(phi + dt * k2_phi, phi_min)
    phi_next = apply_phi_BC(phi_next, phi0)
    return m_next, phi_next, psi, err, iters