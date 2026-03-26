import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import jv, yv, jvp, yvp

def compute_fields_coated_PEC(rho, phi, omega, k1, n1, n_func, Y1, a, b, m_max, eps0, mu0):
    """
    rho: observation point
    k1: wavenumber outside cylinder
    n1: refractive index outside cylinder
    n_func: a function n(r) that returns refractive index at normalized radius r
    a: outer radius
    b: PEC radius
    """
    Ez = 0j
    Hphi = 0j
    
    r_start = b/a
    r = rho/a
    
    def scaled_system(r, f, m, k1, a, n_func, n1):
        f1, f2 = f
        a11 = -np.abs(m)
        a12 = -k1*a
        a21 = k1*a*r**2*n_func(r)**2/(n1**1)-m**2/(k1*a)
        a22 = -np.abs(m)
        
        df1 = 1/r * (a11*f1 + a12*f2)
        df2 = 1/r * (a21*f1 + a22*f2)
        
        return [df1, df2]

    for m in range(-m_max, m_max):
        f1 = 0
        f2 = 1 + 0j

        f_0 = [f1, f2]
        
        sol = solve_ivp(scaled_system, [r_start, r], f_0, args=(m, k1, a, n_func, n1), method='RK45', rtol=1e-10, atol=1e-12)
        f_sol = sol.y[:, -1]

        ez = r**np.abs(m)/1j*f_sol[0]
        hphi = r**(np.abs(m)-1)*f_sol[1]
        
        Ez += np.exp(1j*m*phi)*ez
        Hphi += np.exp(1j*m*phi)*Y1*hphi
 
    return Ez, Hphi
