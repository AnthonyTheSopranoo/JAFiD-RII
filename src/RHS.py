import jax
import jax.numpy as jnp
import typing
from jax.typing import Array


@jax.jit
def get_velocities(psi: Array):

    qx = 1 + ((jnp.roll(psi, -1, axis=0) - jnp.roll(psi, 1, axis=0)) / (2 * dy))

    qy = jnp.zeros_like(psi)
    qy = qy.at[:, 1:-1].set(-((psi[:, 2:] - psi[:, :-2]) / (2 * dx)))

    return qx, qy


@jax.jit
def RHS(
    phi: Array, psi: Array, C: Array, dx: float, dy: float, H: float, W: float
) -> Array:
    C_mid = C[:, 1:-1]

    C_left = C[:, :-2]
    C_right = C[:, 2:]

    C_up = jnp.roll(C, -1, axis=0)[:, 1:-1]
    C_down = jnp.roll(C, 1, axis=0)[:, 1:-1]

    phi_mid = phi[:, 1:-1]

    phi_left = phi[:, :-2]
    phi_right = phi[:, 2:]

    phi_up = jnp.roll(phi, -1, axis=0)[:, 1:-1]
    phi_down = jnp.roll(phi, 1, axis=0)[:, 1:-1]

    phi_half_x_pos = (phi_mid + phi_right) / 2
    phi_half_x_neg = (phi_mid + phi_left) / 2

    phi_half_y_pos = (phi_mid + phi_up) / 2
    phi_half_y_neg = (phi_mid + phi_down) / 2

    qx, qy = get_velocities(psi)

    qx_mid = qx[:, 1:-1]
    qx_left = qx[:, :-2]
    qx_right = qx[:, 2:]

    qx_half_pos_x = (qx_mid + qx_right) / 2
    qx_half_neg_x = (qx_mid + qx_left) / 2

    qy_mid = qy[:, 1:-1]
    qy_up = jnp.roll(qy, -1, axis=0)[:, 1:-1]
    qy_down = jnp.roll(qy, 1, axis=0)[:, 1:-1]

    qy_half_pos_y = (qy_mid + qy_up) / 2
    qy_half_neg_y = (qy_mid + qy_down) / 2

    diffusion_x = phi_half_x_pos * ((C_right - C_mid) / (dx**2)) - phi_half_x_neg * (
        (C_mid - C_left) / (dx**2)
    )
    diffusion_y = phi_half_y_pos * ((C_up - C_mid) / (dy**2)) - phi_half_y_neg * (
        (C_mid - C_down) / (dy**2)
    )

    diffusion = H * (diffusion_x + diffusion_y)

    qxC = qx_half_pos_x * ((C_right + C_mid) / (2 * dx)) - qx_half_neg_x * (
        (C_mid + C_left) / (2 * dx)
    )
    qyC = qy_half_pos_y * ((C_up + C_mid) / (2 * dy)) - qy_half_neg_y * (
        (C_mid + C_down) / (2 * dy)
    )

    phiCdx = phi_half_x_pos * ((C_right + C_mid) / (2 * dx)) - phi_half_x_neg * (
        (C_mid + C_left) / (2 * dx)
    )

    advection = (qxC + qyC) - W * phiCdx

    reaction = C_mid * (1 - phi[:, 1:-1])

    phiC = jnp.zeros_like(C)
    phiC = phiC.at[:, 1:-1].set(-advection + diffusion - reaction)

    return phiC


@jax.jit
def RHS_phi(C: Array, phi: Array, dx: float, dy: float, S: float, W: float) -> Array:
    phi_x = jnp.zeros_like(phi)

    phi_x = phi_x.at[:, :-2].set(
        (-3.0 * phi[:, :-2] + 4.0 * phi[:, 1:-1] - phi[:, 2:]) / (2.0 * dx)
    )
    phi_x = phi_x.at[:, -2].set((phi[:, -1] - phi[:, -2]) / dx)
    phi_x = phi_x.at[:, -1].set(0.0)

    reaction = S * C * (1.0 - phi)
    dphi = reaction + W * phi_x
    dphi = dphi.at[:, -1].set(0.0)
    return dphi
