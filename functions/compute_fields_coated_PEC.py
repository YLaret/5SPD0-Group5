import numpy as np
from scipy.integrate import solve_ivp

def compute_fields_coated_PEC(rho, phi, k1, n1, n_func, a, b, m_max, am_c, am_r):
    """
    rho: observation radius
    k1: wavenumber in exterior
    n1: refractive index in exterior
    n_func: a function n(r) that returns refractive index at normalized radius r
    a: outer radius
    b: PEC radius
    """
    Ez = 0j
    Hphi = 0j
    
    # Normalized start and end points
    r_start = b / a
    r_target = rho / a
    
    def scaled_system(r, f, m, k1, a, n1, n_func):
        abs_m = abs(m)
        n_r = n_func(r) # refractive index at this layer 
        
        # Matrix A elements from Eq. (14) 
        A11 = -abs_m
        A12 = -k1 * a
        A21 = (k1 * a * r**2 * (n_r**2 / n1**2)) - (m**2 / (k1 * a))
        A22 = -abs_m
        
        # df/dr = (1/r) * A * f 
        df1 = (1/r) * (A11 * f[0] + A12 * f[1])
        df2 = (1/r) * (A21 * f[0] + A22 * f[1])
        return [df1, df2]

    for m in range(-m_max, m_max):
        # 1. Boundary Condition at PEC (r = b/a)
        # At PEC, Ez = 0. Therefore f1 = 0.
        # We set f2 = 1 (arbitrary excitation constant).
        f_initial = [0j, 1+0j]
        
        # 2. Integrate from b/a to the target observation point
        # Note: we stay in the normalized 'r' domain for the solver
        sol = solve_ivp(
            scaled_system, 
            [r_start, r_target], 
            f_initial, 
            args=(m, k1, a, n1, n_func),
            method='RK45',
            rtol=1e-8
        )
        
        f_final = sol.y[:, -1]
        r_final = r_target
        
        # 3. Descale back to physical fields 
        # Ez = -j * r^|m| * f1
        # Hphi = r^(|m|-1) * f2 
        # (Note: Y1 factor from Eq 2 is usually handled in scaling or post-processing)
        curr_Ez = -1j * (r_final**abs(m)) * f_final[0]
        curr_Hphi = (r_final**(abs(m) - 1)) * f_final[1]
        
        if m+m_max > m_max:
            curr_Ez = curr_Ez*1/(am_r[m])*1/(am_c[m])
        else:
            curr_Ez = curr_Ez*1/(am_r[m+m_max])*1/(am_c[m_max+m])
        
        Ez += curr_Ez * np.exp(1j * m * phi)
        Hphi += curr_Hphi * np.exp(1j * m * phi)
            
    return Ez, Hphi