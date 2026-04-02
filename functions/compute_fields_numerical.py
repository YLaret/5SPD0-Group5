import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import jv, jvp, h2vp, hankel2

def compute_fields_numerical(rho, phi, k1, n1, n2, nfunc, a, b, m_max, E0, Y1):
    """
    rho: point in radial direction
    k1: wavenumber outside cylinder
    n1: refractive index outside cylinder
    n_func: a function n(r) that returns refractive index at normalized radius r
    a: outer radius
    b: inner radius
    """
    Ez_total = 0j
    r_obs = rho/a  #normalized
    r_start = b/a
    eps = 1e-6  # for finite difference (derivative)
    for m in range(-m_max, m_max + 1):
        abs_m = abs(m)
        f_init = [0j, 1.0 + 0j]  # initial values for f
        def ode_system(r, f):
            nr = nfunc(n1, n2, r*a)
            A11, A12 = -abs_m, -k1*a
            A21 = (k1*a*r**2*(nr**2/n1**2))-(m**2/(k1*a))
            A22 = -abs_m
            return [(1/r)*(A11*f[0] + A12*f[1]), (1/r)*(A21*f[0] + A22*f[1])]

        # getting a solution for r=1 to get a derivative of Ez to r
        sol = solve_ivp(ode_system, [r_start, 1.0], f_init, rtol=1e-10, atol=1e-12, dense_output=True, method='RK45')
        sol_eps = solve_ivp(ode_system, [r_start, 1.0 + eps], f_init, rtol=1e-10, atol=1e-12, dense_output=True, method='RK45')
        f1 = sol.y[0, -1]
        f1_eps = sol_eps.y[0, -1]
        # unscaling
        Ez = -1j*(1**abs_m)*f1
        Ez_eps = -1j*((1+eps)**abs_m)*f1_eps
        # compute finite difference to get the derivative
        dEz_drho = (Ez_eps-Ez)/(eps*a)
        # calculate Hphi using derivative Ez to rho
        hphi_int_unit = (-1j*Y1/k1)*dEz_drho # Hphi = 1/(j*omega*mu)dEz/drho and 1/(j*omega*mu) = -j*Y1/k1

        # Bessel and Hankel on the boundaries to compute ratio
        jm = (-1j)**m
        Jm_ka = jv(m, k1*a)
        Jp_ka = jvp(m, k1*a)
        Hm_ka = hankel2(m, k1*a)
        Hp_ka = h2vp(m, k1*a)

        M = np.array([[-E0*jm*Hm_ka, Ez], [-E0*jm*-1j*Y1*Hp_ka, hphi_int_unit]])
        V = np.array([E0*jm*Jm_ka, E0*jm*-1j*Y1*Jp_ka])
        Bm, Cm = np.linalg.solve(M, V)
        
        if rho < a: # solve for inside outer radius, inside b is zero
            sol_rho = solve_ivp(ode_system, [r_start, r_obs], f_init, rtol=1e-10, atol=1e-12, method='RK45')
            f1_rho = sol_rho.y[0, -1]
            ez_m = Cm*(-1j*(r_obs**abs_m)*f1_rho) # unscale
        else: # outside is analytical solution
            ez_m = E0*jm*(jv(m, k1*rho)+Bm*hankel2(m, k1*rho))

        Ez_total += ez_m*np.exp(1j*m*phi)

    return Ez_total
