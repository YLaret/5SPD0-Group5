import numpy as np
from scipy.special import jv, jvp, yv, hankel2, h2vp

def compute_E_z_coated_PEC(rho, phi, k1, k2, am_r, am_c, E0, a, b):
    """
        rho = distance
        phi = angle
        k1 = wavenumber in the homogeneous exterior domain
        k2 = wavenumber in the homogeneous interior domain
        am_r, am_t = complex amplitudes
        E0 = wave amplitude
        a = outer radius
        b = PEC radius
    """
    
    Ez = 0j
    Escat = 0j
    Ein = 0j
    
    m_max = int(len(am_r)/2)
    
    for m in range(-m_max, m_max):
        # Angular harmonic coefficients
        Jm_k1rho = jv(m, k1*rho)
        H2m_k1rho = hankel2(m, k1*rho)

        # PEC coefficient
        beta = jv(m, k2*b) / yv(m, k2*b)
        
        # coating radial basis
        coating_basis = jv(m, k2*rho) - beta*yv(m, k2*rho)

        em_i = (-1j)**m * Jm_k1rho
        em_r = am_r[m_max+m] * (-1j)**m * H2m_k1rho
        em_c = am_c[m_max+m] * (-1j)**m * coating_basis
        
        if rho>=a:
            Ez += np.exp(1j*m*phi)*E0*(em_i+em_r)
            Escat += np.exp(1j*m*phi)*E0*em_r
            Ein += np.exp(1j*m*phi)*E0*em_i
        elif b < rho < a:
            Ez += np.exp(1j*m*phi)*E0*em_c
            Ein += np.exp(1j*m*phi)*E0*em_i
        else:
            Ez += 0
            
    return Ez, Escat, Ein
