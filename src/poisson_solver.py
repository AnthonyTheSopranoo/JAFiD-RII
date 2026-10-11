import jax
import jax.numpy as jnp


@jax.jit
def restriction_x(array):
    y_len, x_len = array.shape
    lenx_coarsened = ((x_len) - 1) // 2 + 1
    coarsened_in_x = jnp.zeros((y_len, lenx_coarsened))
    coarsened_in_x = coarsened_in_x.at[:, 1:-1].set(
        array[:, 2:-1:2] * 0.5 + array[:, 1:-2:2] * 0.25 + array[:, 3::2] * 0.25
    )

    return coarsened_in_x


@jax.jit
def restriction_y(array):
    y_len, x_len = array.shape
    coarsened_in_y = jnp.zeros((y_len // 2, x_len))

    coarse_y = (
        array * 0.5
        + jnp.roll(array, -1, axis=0) * 0.25
        + jnp.roll(array, 1, axis=0) * 0.25
    )[::2]
    coarsened_in_y = coarsened_in_y.at[:, 1:-1].set(coarse_y[:, 1:-1])

    return coarsened_in_y


@jax.jit
def prolongation_x(array):
    y_len, x_len = array.shape
    fine_grid_x = jnp.zeros((y_len, 2 * x_len - 1))

    fine_grid_x = fine_grid_x.at[:, ::2].set(array)

    fine_grid_x = fine_grid_x.at[:, 1:-1:2].set(
        0.5 * (fine_grid_x[:, :-1:2] + fine_grid_x[:, 2::2])
    )

    return fine_grid_x


@jax.jit
def prolongation_y(array):
    y_len, x_len = array.shape
    fine_grid_y = jnp.zeros((y_len * 2, x_len))
    fine_grid_y = fine_grid_y.at[::2, :].set(array)

    fine_grid_y = fine_grid_y.at[1::2, :].set(
        (
            jnp.roll(fine_grid_y, 1, axis=0)[1::2, :]
            + jnp.roll(fine_grid_y, -1, axis=0)[1::2, :]
        )
        / 2
    )

    return fine_grid_y


from functools import partial


@jax.jit
def source_psi(phi, dy):
    perm = 1 / jnp.power(phi, 2)
    perm_mid = perm[:, 1:-1]
    perm_up = jnp.roll(perm, -1, axis=0)[:, 1:-1]
    perm_down = jnp.roll(perm, 1, axis=0)[:, 1:-1]
    perm_pos_half_y = (perm_mid + perm_up) / 2
    perm_neg_half_y = (perm_mid + perm_down) / 2
    f = jnp.zeros_like(phi)
    return f.at[:, 1:-1].set((perm_pos_half_y - perm_neg_half_y) / dy)


@jax.jit
def jacobi_psi_step(psi, phi, f, dx, dy):
    psi_right = psi[:, 2:]
    psi_left = psi[:, :-2]
    psi_up = jnp.roll(psi, -1, axis=0)[:, 1:-1]
    psi_down = jnp.roll(psi, 1, axis=0)[:, 1:-1]

    perm = 1 / jnp.power(phi, 2)
    perm_right = perm[:, 2:]
    perm_left = perm[:, :-2]
    perm_mid = perm[:, 1:-1]
    perm_up = jnp.roll(perm, -1, axis=0)[:, 1:-1]
    perm_down = jnp.roll(perm, 1, axis=0)[:, 1:-1]

    perm_pos_half_x = (perm_mid + perm_right) / 2
    perm_neg_half_x = (perm_mid + perm_left) / 2
    perm_pos_half_y = (perm_mid + perm_up) / 2
    perm_neg_half_y = (perm_mid + perm_down) / 2

    aP = (perm_pos_half_x + perm_neg_half_x) / dx**2 + (
        perm_pos_half_y + perm_neg_half_y
    ) / dy**2

    psiij_1 = (perm_pos_half_x * psi_right + perm_neg_half_x * psi_left) / dx**2
    psiij_2 = (perm_pos_half_y * psi_up + perm_neg_half_y * psi_down) / dy**2
    psi_ij_3 = f[:, 1:-1]

    psiij = (psiij_1 + psiij_2 + psi_ij_3) / aP

    psi_array = jnp.zeros_like(phi)
    return psi_array.at[:, 1:-1].set(psiij)


@jax.jit
def residual_psi(psi, phi, f, dx, dy):
    k = 1.0 / phi**2
    kc = k[:, 1:-1]
    kE = (kc + k[:, 2:]) / 2
    kW = (kc + k[:, :-2]) / 2
    kN = (kc + jnp.roll(k, -1, axis=0)[:, 1:-1]) / 2
    kS = (kc + jnp.roll(k, 1, axis=0)[:, 1:-1]) / 2

    pc = psi[:, 1:-1]
    pE, pW = psi[:, 2:], psi[:, :-2]
    pN = jnp.roll(psi, -1, axis=0)[:, 1:-1]
    pS = jnp.roll(psi, 1, axis=0)[:, 1:-1]

    div_x = (kE * (pE - pc) - kW * (pc - pW)) / dx**2
    div_y = (kN * (pN - pc) - kS * (pc - pS)) / dy**2

    r = jnp.zeros_like(psi)
    return r.at[:, 1:-1].set(f[:, 1:-1] + div_x + div_y)


@partial(jax.jit, static_argnames="n_iter")
def smooth_psi(psi_init, phi, f, dx, dy, n_iter, omega=2 / 3):
    def body(_, psi):
        psi_jac = jacobi_psi_step(psi, phi, f, dx, dy)
        return (1 - omega) * psi + omega * psi_jac

    return jax.lax.fori_loop(0, n_iter, body, psi_init)


def V_cycle(psi, phi, f, dx, dy, num_levels, level=1):
    if num_levels == level:
        return smooth_psi(psi, phi, f, dx, dy, n_iter=50)  # coarse "solve"

    u_smoothed = smooth_psi(psi, phi, f, dx, dy, n_iter=1)

    res = residual_psi(u_smoothed, phi, f, dx, dy)

    restricted_x = restriction_x(res)
    restricted_residual = restriction_y(restricted_x)

    phi_coarse = phi[::2, ::2]  # injection

    e_c = V_cycle(
        jnp.zeros_like(restricted_residual),
        phi_coarse,
        restricted_residual,
        2 * dx,
        2 * dy,
        num_levels,
        level + 1,
    )

    u = u_smoothed + prolongation_y(prolongation_x(e_c))

    u = smooth_psi(u, phi, f, dx, dy, n_iter=1)  # was psi

    return u


@partial(jax.jit, static_argnames="num_levels")
def solve_psi_jit(phi, psi_init, dx, dy, num_levels, psi_tol=5e-8, psi_maxiter=50):
    f = source_psi(phi, dy)

    def cond(state):
        psi, psi_old, it = state
        err = jnp.max(jnp.abs(psi - psi_old))
        return (err > psi_tol) & (it < psi_maxiter)

    def body(state):
        psi, _, it = state
        return (V_cycle(psi, phi, f, dx, dy, num_levels), psi, it + 1)

    state0 = (psi_init, psi_init + 1.0, 0)
    psi, psi_old, it = jax.lax.while_loop(cond, body, state0)
    err = jnp.max(jnp.abs(psi - psi_old))
    return psi, err, it
