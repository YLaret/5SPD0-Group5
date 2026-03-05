import numpy as np
from scipy.special import jv, jvp, hankel2, h2vp

def compute_E_z(rho, phi, k1, k2, am_r, am_t, E0, a):
    """
        rho = distance
        phi = angle
        k1 = wavenumber in the homogeneous exterior domain
        k2 = wavenumber in the homogeneous interior domain
        am_r, am_t = complex amplitudes
        E0 = wave amplitude
        a = boundary distance
    """
    Ez = 0j
    Escat = 0j
    m_max = int(len(am_r)/2)
    for m in range(-m_max, m_max):
        # Angular harmonic coefficients
        Jm_k1rho = jv(m, k1*rho)
        Jm_k2rho = jv(m, k2*rho)
        H2m_k1rho = hankel2(m, k1*rho)

        em_i = (-1j)**m*Jm_k1rho
        em_r = am_r[m_max+m]*(-1j)**m*H2m_k1rho
        em_t = am_t[m_max+m]*(-1j)**m*Jm_k2rho
        
        if rho>=a:
            Ez += np.exp(1j*m*phi)*E0*(em_i+em_r)
            Escat += np.exp(1j*m*phi)*E0*em_r
        elif rho<=a:
            Ez += np.exp(1j*m*phi)*E0*em_t
    return Ez, Escat
