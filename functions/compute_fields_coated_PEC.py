import numpy as np
from scipy.integrate import solve_ivp

# def compute_fields_coated_PEC(rho, phi, k1, n1, n_func, a, b, m_max):
#     """
#     rho: observation point
#     k1: wavenumber outside cylinder
#     n1: refractive index outside cylinder
#     n_func: a function n(r) that returns refractive index at normalized radius r
#     a: outer radius
#     b: PEC radius
#     """
#     Ez = 0j
#     Hphi = 0j
    
#     # genormaliseerde begin en eind waarden, zodat je in de solver ook gescaled blijft 
#     r_start = b/a
#     r_end = rho/a
#     def scaled_system(r, f, m, k1, a, n1, n_func):
#         abs_m = abs(m)
#         n_r = n_func(r) 
#         # de matrix 
#         A11 = -abs_m
#         A12 = -k1*a
#         A21 = (k1*a*r**2*(n_r**2/n1**2))-(m**2/(k1*a))
#         A22 = -abs_m
#         # de oplossing voor de afgeleide naar geschaalde rho
#         df1 = (1/r)*(A11*f[0]+A12*f[1])
#         df2 = (1/r)*(A21*f[0]+A22*f[1])
#         return [df1, df2]

#     for m in range(-m_max, m_max + 1):
#         f_initial = [0j, 1+0j] # willekeurige Hphi
        
#         sol = solve_ivp(scaled_system, [r_start, r_end], f_initial, args=(m, k1, a, n1, n_func), method='RK45', rtol=1e-10, atol=1e-12)
#         f_final = sol.y[:, -1]

#         Ez_c = 1j*((rho/a)**abs(m))*f_final[0]
#         Hphi_c = ((rho/a)**(abs(m)-1))*f_final[1]
        
#         Ez += Ez_c*np.exp(1j*m*phi)
#         Hphi += Hphi_c*np.exp(1j*m*phi)
            
#     return Ez, Hphi, sol.y

import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import jv, jvp, hankel2, h2vp

# def compute_fields_coated_PEC(rho, phi, k1, n1, n_func, a, b, m_max, E0, Y1):
#     """
#     rho: observation radius
#     phi: observation angles (array)
#     k1: exterior wavenumber
#     n1: exterior refractive index
#     n_func: function n(r) for the dielectric
#     a: outer radius
#     b: inner PEC radius
#     m_max: number of harmonics
#     E0: incident wave amplitude
#     """
#     Ez_total = np.zeros_like(phi, dtype=complex)
#     r_obs = rho / a
#     r_start = b / a  # PEC boundary
#     eps = 1e-6

#     for m in range(-m_max, m_max + 1):
#         abs_m = abs(m)
        
#         # --- 1. NUMERICAL INTEGRATION (Unit Solution) ---
#         # We start at PEC with Ez=0 (f1=0) and Hphi=1 (f2=1)
#         f_init = [0j, 1.0 + 0j]
        
#         def ode_system(r, f):
#             nr = n_func(r)
#             A11, A12 = -abs_m, -k1*a
#             A21 = (k1*a * r**2 * (nr**2/n1**2)) - (m**2/(k1*a))
#             A22 = -abs_m
#             return [(1/r)*(A11*f[0] + A12*f[1]), (1/r)*(A21*f[0] + A22*f[1])]

#         # Integrate to the boundary r=1 to find the internal 'basis' values
#         sol_match = solve_ivp(ode_system, [r_start, 1.0], f_init, rtol=1e-10, atol = 1e-12)
#         f1_at_1 = sol_match.y[0, -1]
#         f2_at_1 = sol_match.y[1, -1]

#         # Convert scaled f to physical e_z and h_phi (at r=1)
#         # From Eq 6: f1 = j * r^-|m| * ez => ez = -j * f1
#         ez_int_unit = -1j * f1_at_1
#         hphi_int_unit = f2_at_1

#         # --- 2. BOUNDARY MATCHING (Analytical at r=1) ---
#         # Incident wave harmonics at the boundary rho = a
#         # Term: E0 * (-j)^m * Jm(k1*a)
#         phase_m = (-1j)**m
#         Jm_ka = jv(m, k1*a)
#         Jp_ka = jvp(m, k1*a)
#         Hm_ka = hankel2(m, k1*a)
#         Hp_ka = h2vp(m, k1*a)

#         # Solve 2x2: [Continuity of Ez, Continuity of Hphi]
#         # Cm * ez_int_unit = E0 * phase_m * (Jm_ka + am * Hm_ka)
#         # Cm * hphi_int_unit = E0 * phase_m * (Jp_ka + am * Hp_ka)
        
#         # Rearranging into Matrix: [ -E0*phase*Hm  ez_unit ] [ am ] = [ E0*phase*Jm ]
#         #                          [ -E0*phase*Hp  h_unit  ] [ Cm ]   [ E0*phase*Jp ]
        
#         M = np.array([[-E0 * phase_m * Hm_ka, ez_int_unit],
#                       [-E0 * phase_m * Hp_ka * Y1, hphi_int_unit]])
#         V = np.array([E0 * phase_m * Jm_ka, 
#                       E0 * phase_m * Jp_ka * Y1])
        
#         am, Cm = np.linalg.solve(M, V)

#         # --- 3. RECONSTRUCT FIELD AT OBSERVATION POINT ---
#         if rho < a:
#             # Inside: use scaled numerical solution multiplied by Cm
#             sol_rho = solve_ivp(ode_system, [r_start, r_obs], f_init, rtol=1e-10)
#             f1_rho = sol_rho.y[0, -1]
#             # Internal physical Ez_m = Cm * (-j * r^|m| * f1)
#             ez_m = Cm * (-1j * (r_obs**abs_m) * f1_rho)
#         else:
#             # Outside: use Analytical Incident + Scattered
#             ez_m = E0 * phase_m * (jv(m, k1*rho) + am * hankel2(m, k1*rho))

#         Ez_total += ez_m * np.exp(1j*m*phi)
        
#     return Ez_total

def compute_fields_coated_PEC(rho, phi, k1, n1, n_func, a, b, m_max, E0, Y1):
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