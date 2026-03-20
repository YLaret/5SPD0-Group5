# import numpy as np
# from scipy.special import jv, jvp, yv, hankel2, h2vp
# from scipy.integrate import solve_ivp

# def compute_fields_coated_PEC(rho, phi, eps0, mu0, epsr, E0, omega, a, b, m_max):
#     """
#         rho = distance
#         k1 = wavenumber in the homogeneous exterior domain
#         k2 = wavenumber in the coating domain
#         E0 = wave amplitude
#         a = outer radius
#         b = PEC radius
#     """
    
#     Ez = 0j
#     Hphi = 0j
    
#     def system(r, y, omega, mu0, eps0, epsr, m):
#         Ez_m, Hphi_m = y
    
#         dEz_m = 1j*omega*mu0*Hphi_m
#         dHphi_m = 1j*omega*eps0*epsr*Ez_m - 1j*m**2/(omega*mu0*r**2)*Ez_m - 1/r*Hphi_m
        
#         return [dEz_m, dHphi_m]

#     print('system:', system) 
    
#     for m in range(-m_max, m_max + 1):
#         # coated span
#         r_span = [b,a]

#         # boundary coating-vaccuum
#         r_boundary = a
        
#         # boundary PEC-coating (E=0, H=1 (PEC))
#         y_pec = [0j, 1+0j]
        
#         # evaluate at boundary coating-vaccuum
#         sol_boundary = solve_ivp(system, r_span, y_pec, t_eval=[r_boundary], args=[omega,mu0,eps0,epsr,m], method='BDF')
        
#         # compute @ rho
#         if rho <= b:
#             Ez += y_pec[0] * np.exp(1j * m * phi)
#             Hphi += y_pec[1] * np.exp(1j * m * phi)
#         elif rho <= a:
#             sol_rho = solve_ivp(system, r_span, y_pec, t_eval=[rho], args=[omega,mu0,eps0,epsr,m], method='BDF')
#             Ez += sol_rho.y[0, 0] * np.exp(1j * m * phi)
#             Hphi += sol_rho.y[1, 0] * np.exp(1j * m * phi)
#         else:
#             sol_rho = solve_ivp(system, [a, rho], sol_boundary.y[:,-1], t_eval=[rho], args=[omega,mu0,eps0,1,m], method='BDF')
#             Ez += sol_rho.y[0,0] * np.exp(1j * m * phi)
#             Hphi += sol_rho.y[1,0] * np.exp(1j * m * phi)
            
#     return Ez, Hphi

# import numpy as np
# from scipy.integrate import solve_ivp
# from scipy.special import jv, jvp, hankel2, h2vp

# def compute_fields_coated_PEC(rho, phi, eps0, mu0, epsr, E0, omega, a, b, m_max):
#     Ez_total = np.zeros(len(phi), dtype=complex)
#     Hphi_total = np.zeros(len(phi), dtype=complex)
    
#     k1 = omega * np.sqrt(mu0 * eps0)
#     Z1 = np.sqrt(mu0 / eps0)

#     def system(r, y, m, omega, mu0, eps0, epsr_val):
#         Ez_m, Hphi_m = y
#         dEz_dr = 1j * omega * mu0 * Hphi_m
#         dHphi_dr = (1j * omega * eps0 * epsr_val - 1j * m**2 / (omega * mu0 * r**2)) * Ez_m - (1/r) * Hphi_m
#         return [dEz_dr, dHphi_dr]

#     for m in range(-m_max, m_max + 1):
#         # 1. Get the "Mode Shape" by integrating from PEC to Boundary a
#         y0 = [0j, 1.0 + 0j] 
#         sol_to_a = solve_ivp(system, [b, a], y0, args=(m, omega, mu0, eps0, epsr), method='RK45')
        
#         # Values at boundary a
#         E_num_a = sol_to_a.y[0, -1]
#         H_num_a = sol_to_a.y[1, -1]

#         # 2. Boundary Matching to find the scaling constant C_m
#         # Goal: C*E_num(a) = E_inc(a) + E_scat(a)
#         #       C*H_num(a) = H_inc(a) + H_scat(a)
        
#         # Incident and Hankel terms at r=a
#         jm = 1j**(-m)
#         Ei_a = E0 * jm * jv(m, k1 * a)
#         Hi_a = E0 * jm * (1j / Z1) * jvp(m, k1 * a)
        
#         Es_a = hankel2(m, k1 * a)
#         Hs_a = (1j / Z1) * h2vp(m, k1 * a)

#         # Solve the 2x2 system for [C_m, A_m]
#         # [ E_num_a  -Es_a ] [ C_m ] = [ Ei_a ]
#         # [ H_num_a  -Hs_a ] [ A_m ] = [ Hi_a ]
#         M = np.array([[E_num_a, -Es_a], [H_num_a, -Hs_a]])
#         rhs = np.array([Ei_a, Hi_a])
#         Cm, Am = np.linalg.solve(M, rhs)

#         # 3. Compute Field at rho using the scaling Cm
#         if rho <= a:
#             # Re-integrate or evaluate internal field scaled by Cm
#             sol_rho = solve_ivp(system, [b, rho], y0, args=(m, omega, mu0, eps0, epsr), method='RK45', t_eval=[rho])
#             Ez_m_rho = Cm * sol_rho.y[0, 0]
#         else:
#             # Field is external: Incident + Scattered (using Am)
#             Ez_m_rho = E0 * jm * jv(m, k1 * rho) + Am * hankel2(m, k1 * rho)

#         Ez_total += Ez_m_rho * np.exp(1j * m * phi)

#     return Ez_total, Hphi_total

import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import jv, jvp, hankel2, h2vp

def compute_fields_coated_PEC(rho, phi, eps0, mu0, epsr, E0, omega, a, b, m_max):
    # Initialize as complex array matching the shape of phi
    Ez_total = np.zeros(len(phi), dtype=complex)
    
    k1 = omega * np.sqrt(mu0 * eps0)
    Z1 = np.sqrt(mu0 / eps0)

    def system(r, y, m, omega, mu0, eps0, epsr_val):
        Ez_m, Hphi_m = y
        dEz_dr = 1j * omega * mu0 * Hphi_m
        dHphi_dr = (1j * omega * eps0 * epsr_val - 1j * m**2 / (omega * mu0 * r**2)) * Ez_m - (1/r) * Hphi_m
        return [dEz_dr, dHphi_dr]

    for m in range(-m_max, m_max + 1):
        # 1. Integrate for the mode shape (PEC to Boundary)
        y0 = [0j, 1.0 + 0j] 
        sol_to_a = solve_ivp(system, [b, a], y0, args=(m, omega, mu0, eps0, epsr), 
                             method='RK45', rtol=1e-10, atol=1e-13)
        
        E_num_a = sol_to_a.y[0, -1]
        H_num_a = sol_to_a.y[1, -1]

        # 2. Match with external Vacuum field
        jm_term = E0 * (1j**-m)
        Ei_a = jm_term * jv(m, k1 * a)
        Hi_a = jm_term * (1j / Z1) * jvp(m, k1 * a)
        
        Es_a = hankel2(m, k1 * a)
        Hs_a = (1j / Z1) * h2vp(m, k1 * a)

        # Matrix solve for coefficients
        M = np.array([[E_num_a, -Es_a], [H_num_a, -Hs_a]])
        rhs = np.array([Ei_a, Hi_a])
        Cm, Am = np.linalg.solve(M, rhs)

        # 3. Calculate Ez_m at the specific rho requested
        if rho <= a:
            # Re-integrate to rho to get specific value
            sol_rho = solve_ivp(system, [b, rho], y0, args=(m, omega, mu0, eps0, epsr), 
                                method='RK45', t_eval=[rho], rtol=1e-10, atol=1e-13)
            Ez_m_rho = Cm * sol_rho.y[0, 0]
        else:
            # Standard incident + scattered formula
            Ez_m_rho = jm_term * jv(m, k1 * rho) + Am * hankel2(m, k1 * rho)

        # 4. Sum up the modes (Crucial: Multiply by the angular part)
        Ez_total += Ez_m_rho * np.exp(1j * m * phi)

    return Ez_total, None