import numpy as np
from scipy.special import jv, jvp, yv, hankel2, h2vp
from scipy.integrate import solve_ivp

def compute_fields_coated_PEC(rho, phi, eps0, mu0, epsr, E0, omega, a, b, m_max):
    """
        rho = distance
        k1 = wavenumber in the homogeneous exterior domain
        k2 = wavenumber in the coating domain
        E0 = wave amplitude
        a = outer radius
        b = PEC radius
    """
    
    Ez = 0j
    Hphi = 0j
    
    def system(r, y, omega, mu0, eps0, epsr, m):
        Ez_m, Hphi_m = y
    
        dEz_m = 1j*omega*mu0*Hphi_m
        dHphi_m = 1j*omega*eps0*epsr*Ez_m - 1j*m**2/(omega*mu0*r**2)*Ez_m - 1/r*Hphi_m
        
        return [dEz_m, dHphi_m]
        
    
    for m in range(-m_max, m_max):
        # coated span
        r_span = [b,a]

        # boundary coating-vaccuum
        r_boundary = a
        
        # boundary PEC-coating (E=0, H=1 (PEC))
        y_pec = [0j, 1+0j]
        
        # evaluate at boundary coating-vaccuum
        sol_boundary = solve_ivp(system, r_span, y_pec, t_eval=[r_boundary], args=[omega,mu0,eps0,epsr,m], method='BDF')
        
        # compute @ rho
        if rho <= b:
            Ez += y_pec[0] * np.exp(1j * m * phi)
            Hphi += y_pec[1] * np.exp(1j * m * phi)
        elif rho <= a:
            sol_rho = solve_ivp(system, r_span, y_pec, t_eval=[rho], args=[omega,mu0,eps0,epsr,m], method='BDF')
            Ez += sol_rho.y[0, 0] * np.exp(1j * m * phi)
            Hphi += sol_rho.y[1, 0] * np.exp(1j * m * phi)
        else:
            sol_rho = solve_ivp(system, [a, rho], sol_boundary.y[:,-1], t_eval=[rho], args=[omega,mu0,eps0,epsr,m], method='BDF')
            Ez += sol_rho.y[0,0] * np.exp(1j * m * phi)
            Hphi += sol_rho.y[1,0] * np.exp(1j * m * phi)
            
    return Ez, Hphi
