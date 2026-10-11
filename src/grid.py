import jax
import jax.numpy as jnp
import nump as np
from jax.typing import Array

rng = np.random.default_rng(42)


def C_base_state(X, dphi: float):
    return 1 / (jnp.exp(dphi * X) + 1)


def phi_base_state(X, dphi, phi0: float):
    return phi0 + dphi * (1 / (jnp.exp(dphi * X) + 1))


def generate_grid(
    Ly: int, Ny: int, Lx: int
) -> tuple[Array, Array, float, float, float, int]:
    dy = Ly / (Ny)

    dx = dy

    Nx = int(round(2 * Lx / dx)) + 1
    Lx = (Nx - 1) * dx / 2.0
    x = np.linspace(-Lx, Lx, Nx)
    y = np.linspace(0, Ly, Ny, endpoint=False)
    X, Y = jnp.meshgrid(x, y)

    return X, Y, dx, dy, Lx, Nx


def generate_initial_conditions(
    X, amplitude, dphi: float, phi0: float, Ny: int, x0: float = 0
) -> tuple[Array, Array]:

    zet = x0 + amplitude * rng.standard_normal(Ny)

    C_base = C_base_state(X, dphi)

    phi_base = phi_base_state(X - zet[:, None], dphi, phi0)

    return C_base, phi_base
