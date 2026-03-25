import numpy as np
from scipy.integrate import solve_ivp


def compute_fields_coated_PEC(rho, phi, k1, n1, n_func, a, b, m_max):
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
    
    # genormaliseerde begin en eind waarden, zodat je in de solver ook gescaled blijft 
    r_start = b/a
    r_end = rho/a
    def scaled_system(r, f, m, k1, a, n1, n_func):
        abs_m = abs(m)
        n_r = n_func(r) 
        # de matrix 
        A11 = -abs_m
        A12 = -k1*a
        A21 = (k1*a*r**2*(n_r**2/n1**2))-(m**2/(k1*a))
        A22 = -abs_m
        # de oplossing voor de afgeleide naar geschaalde rho
        df1 = (1/r)*(A11*f[0]+A12*f[1])
        df2 = (1/r)*(A21*f[0]+A22*f[1])
        return [df1, df2]

    for m in range(-m_max, m_max + 1):
        f_initial = [0j, 0.5+0j] # willekeurige Hphi
        
        sol = solve_ivp(scaled_system, [r_start, r_end], f_initial, args=(m, k1, a, n1, n_func), method='RK45', rtol=1e-10, atol=1e-12)
        f_final = sol.y[:, -1]

        Ez_c = 1j*((rho/a)**abs(m))*f_final[0]
        Hphi_c = ((rho/a)**(abs(m)-1))*f_final[1]
        
        Ez += Ez_c*np.exp(1j*m*phi)
        Hphi += Hphi_c*np.exp(1j*m*phi)
            
    return Ez, Hphi, sol.y