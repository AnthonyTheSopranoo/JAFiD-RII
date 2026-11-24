import jax
import jax.numpy as jnp
import numpy as np


def C_base_state(X, dphi):
    return 1 / (jnp.exp(dphi * X) + 1)


def phi_base_state(X, phi0, dphi):
    return phi0 + dphi * (1 / (jnp.exp(dphi * X) + 1))


def generate_grid(
    Ly: float | jax.Array,
    Ny: int,
    Lx: float | jax.Array,
) -> tuple[
    jax.Array,
    jax.Array,
    float | jax.Array,
    float | jax.Array,
    float,
    int,
]:
    dy = Ly / (Ny)

    dx = dy

    Nx = int(round(2 * Lx / dx)) + 1
    Lx = (Nx - 1) * dx / 2.0
    x = np.linspace(-Lx, Lx, Nx)
    y = np.linspace(0, Ly, Ny, endpoint=False)
    X, Y = jnp.meshgrid(x, y)

    return X, Y, dx, dy, Lx, Nx


def generate_initial_conditions(
    X: jax.Array,
    amplitude: float,
    phi0: float,
    dphi: float,
    x0: float,
    Ny: int,
    rng: np.random.Generator = np.random.default_rng(42),
) -> tuple[jax.Array, jax.Array]:
    zet = x0 + amplitude * rng.standard_normal(Ny)

    C_base = C_base_state(X, dphi)

    phi_base = phi_base_state(X - zet[:, None], phi0, dphi)

    return C_base, phi_base