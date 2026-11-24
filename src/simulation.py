import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.70"   # before `import jax`


import time
import typing

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt

from grid import generate_grid, generate_initial_conditions
from poisson_solver import solve_psi_jit
from RHS import get_velocities
from time_stepper import rk2_step

jax.config.update("jax_num_cpu_devices", 20)
# jax.config.update("jax_platform_name", "cpu")

jax.config.update("jax_enable_x64", True)

print(f"JAX backend: {jax.default_backend()}")


def solver(
    H: float,
    S: float,
    W: float,
    phi0: float,
    Ly: float | jax.Array,
    Ny: int,
    Lx: float | jax.Array,
    dt: float,
    Nt: int,
    snapshot_interval: int,
    phi1: float = 1,
    phi_min: float = 1e-8,
    amp: float = 1e-3,
) -> tuple[
    list[npt.NDArray[typing.Any]],
    list[npt.NDArray[typing.Any]],
    list[npt.NDArray[typing.Any]],
    list[float],
]:
    dphi = phi1 - phi0

    X, Y, dx, dy, Lx, Nx = generate_grid(Ly, Ny, Lx)

    C_init, phi_init = generate_initial_conditions(X, amp, phi0, dphi, 0, Ny)

    phi = jnp.array(phi_init)
    m = jnp.array(C_init) * phi
    t0 = time.time()
    psi, psi_err, psi_iters = solve_psi_jit(phi, jnp.zeros((Ny, Nx)), dx, dy, 7)
    print(
        f"  psi: {int(psi_iters)} iters, error={float(psi_err):.2e}  "
        f"range=[{float(psi.min()):.3e}, {float(psi.max()):.3e}]"
    )
    _warm = rk2_step(m, phi, psi, H, W, S, phi0, dt, dx, dy)
    jax.block_until_ready(_warm[0])
    print(f"  Warm-up (compile) done in {time.time() - t0:.2f}s")
    cell = dx * dy
    M0 = float(jnp.sum(m) * cell)

    C_history, phi_history, t_history = [], [], []
    psi_history = []

    print(f"\nNx={Nx}  Ny={Ny}  dx={dx:.5f}  dy={dy:.5f}  dt={dt}  nt={Nt}")
    print(f"H={H}  S={S}  n_exp={2}")
    print("-" * 70)

    t0 = time.time()

    for step in range(Nt):
        m, phi, psi, psi_err, psi_iters = rk2_step(
            m, phi, psi, H, W, S, phi0, dt, dx, dy
        )

        if step % snapshot_interval == 0:
            C_snap = m / jnp.maximum(phi, phi_min)
            C_history.append(np.array(C_snap))
            phi_history.append(np.array(phi))
            psi_history.append(np.array(psi))
            t_history.append(step * dt)
        if step % 20000 == 0:
            C = m / jnp.maximum(phi, phi_min)
            u, v = get_velocities(psi, dx, dy)
            cfl = dt * float(jnp.max(jnp.abs(u) / dx + jnp.abs(v) / dy))
            Mtot = float(jnp.sum(m) * cell)
            print(
                f"step {step:6d} | t={step * dt:.6f} | "
                f"C=[{float(C.min()):.4e},{float(C.max()):.4e}] | "
                f"phi=[{float(phi.min()):.5f},{float(phi.max()):.5f}] | "
                f"M={Mtot:.6e} | CFL={cfl:.5f} | "
                f"psi({int(psi_iters)}it,{float(psi_err):.1e}) | "
                f"{time.time() - t0:.1f}s"
            )

    total = time.time() - t0
    print(f"\nDone.  {total:.1f}s  ({total / Nt * 1000:.2f} ms/step)")

    return C_history, phi_history, psi_history, t_history


if __name__ == "__main__":
    H = 1 / 10
    S = 0.07
    dphi = 0.7
    W = S / (S + dphi)
    phi0 = 0.1
    phi1 = 1.0
    Ly = jnp.pi
    Lx = 5 * Ly
    Ny = 128
    dt = 7e-4
    Nt = 100000
    snapshot_interval = 10000

    C_history, phi_history, psi_history, t_history = solver(
        H, S, W, phi0, Ly, Ny, Lx, dt, Nt, snapshot_interval, phi1=phi1
    )