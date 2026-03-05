import numpy as np
from scipy.special import jv, jvp, hankel2, h2vp

def compute_H_phi(rho, phi, k1, k2, am_r, am_t, E0, Y1, a):
    """
        rho = distance
        phi = angle
        k1 = wavenumber in the homogeneous exterior domain
        k2 = wavenumber in the homogeneous interior domain
        am_r, am_t = complex amplitudes
        E0 = wave amplitude
        Y1 = wave admittance
        a = boundary distance
    """
    H_phi = 0j
    m_max = int(len(am_r)/2)
    for m in range(-m_max, m_max):
        # Angular harmonic coefficients
        Jm_k1rho_p = jvp(m, k1*rho)
        Jm_k2rho_p = jvp(m, k2*rho)
        H2m_k1rho_p = h2vp(m, k1*rho)
        
        hm_i = (-1j)**m*(k1/(1j*k1))*Jm_k1rho_p                    #omega*mu*Y1=k1
        hm_r = am_r[m_max+m]*(-1j)**m*(k1/(1j*k1))*H2m_k1rho_p
        hm_t = am_t[m_max+m]*(-1j)**m*(k2/(1j*k1))*Jm_k2rho_p
        
        if rho>=a:
            H_phi += np.exp(1j*m*phi)*E0*Y1*(hm_i+hm_r)
        elif rho<=a:
            H_phi += np.exp(1j*m*phi)*E0*Y1*hm_t
    return H_phi

