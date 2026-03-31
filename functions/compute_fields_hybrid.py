import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import jv, jvp, h2vp, hankel2

def compute_fields_hybrid(rho, phi, k1, n1, n_func, a, b, m_max, E0, Y1):
    """
    rho: observation point
    k1: wavenumber outside cylinder
    n1: refractive index outside cylinder
    n_func: a function n(r) that returns refractive index at normalized radius r
    a: outer radius
    b: PEC radius
    """
    Ez_total = np.zeros_like(phi, dtype=complex)
    r_obs = rho / a
    r_start = b / a
    eps = 1e-6  # for finite difference

    for m in range(-m_max, m_max + 1):
        abs_m = abs(m)
        f_init = [0j, 1.0 + 0j]

        def ode_system(r, f):
            nr = n_func(r)
            A11, A12 = -abs_m, -k1 * a
            A21 = (k1 * a * r**2 * (nr**2 / n1**2)) - (m**2 / (k1 * a))
            A22 = -abs_m
            return [(1/r)*(A11*f[0] + A12*f[1]),
                    (1/r)*(A21*f[0] + A22*f[1])]

        # Integrate to r=1 AND to r=1+eps to get dEz/dr numerically
        sol_1     = solve_ivp(ode_system, [r_start, 1.0],       f_init, rtol=1e-10, atol=1e-12, dense_output=True)
        sol_1_eps = solve_ivp(ode_system, [r_start, 1.0 + eps], f_init, rtol=1e-10, atol=1e-12, dense_output=True)

        f1_at_1     = sol_1.y[0, -1]
        f1_at_1_eps = sol_1_eps.y[0, -1]

        # Physical Ez at r=1:  Ez = -i * r^|m| * f1, at r=1 → -i * f1
        ez_at_1     = -1j * (1.0     ** abs_m) * f1_at_1
        ez_at_1_eps = -1j * ((1.0+eps) ** abs_m) * f1_at_1_eps

        # Physical dEz/d(physical_rho) at rho=a:
        # r = rho/a, so d/d(rho) = (1/a)*d/dr
        dez_drho = (ez_at_1_eps - ez_at_1) / (eps * a)

        # Physical Hphi from curl: Hphi = (1/iωμ) * dEz/drho = Y1/k1 * dEz/drho
        # (since Y1 = k1/(ωμ), so 1/(iωμ) = -i*Y1/k1)
        hphi_int_unit = (-1j * Y1 / k1) * dez_drho

        # --- Boundary matching (Ez and Hφ in physical units on both sides) ---
        phase_m = (-1j)**m
        Jm_ka = jv(m,      k1 * a)
        Jp_ka = jvp(m,     k1 * a)   # derivative w.r.t. argument k1*a
        Hm_ka = hankel2(m, k1 * a)
        Hp_ka = h2vp(m,    k1 * a)

        # Exterior physical Hphi = (1/iωμ)*dEz/drho = (-i*Y1/k1) * k1 * Bessel'
        #                        = -i * Y1 * Bessel'(k1*a)
        hphi_ext_J = -1j * Y1 * Jp_ka
        hphi_ext_H = -1j * Y1 * Hp_ka

        M = np.array([
            [-E0 * phase_m * Hm_ka,       ez_at_1      ],
            [-E0 * phase_m * hphi_ext_H,  hphi_int_unit]
        ])
        V = np.array([
             E0 * phase_m * Jm_ka,
             E0 * phase_m * hphi_ext_J
        ])

        am, Cm = np.linalg.solve(M, V)

        # --- Field reconstruction ---
        if rho < a:
            sol_rho = solve_ivp(ode_system, [r_start, r_obs], f_init, rtol=1e-10, atol=1e-12)
            f1_rho = sol_rho.y[0, -1]
            ez_m = Cm * (-1j * (r_obs**abs_m) * f1_rho)
        else:
            ez_m = E0 * phase_m * (jv(m, k1*rho) + am * hankel2(m, k1*rho))

        Ez_total += ez_m * np.exp(1j * m * phi)

    return Ez_total