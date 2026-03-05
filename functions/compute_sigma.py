import numpy as np
from scipy.special import jv, jvp, hankel2, h2vp

def compute_sigma(phi, k1, k2, a, m_max, E0):
    """
        phi = angle
        k1 = wavenumber in the homogeneous exterior domain
        k2 = wavenumber in the homogeneous interior domain
        am_r, am_t = complex amplitudes
        E0 = wave amplitude
        Y1 = wave admittance
        a = boundary distance
    """
    sigma = np.zeros(len(phi),dtype=np.complex128)
    for p in range(len(phi)):
        for m in range(-m_max, m_max):
            # Definitions of Bessel and Hankel functions
            Jm_k1a = jv(m, k1*a)
            Jm_k2a = jv(m, k2*a)
            H2m_k1a = hankel2(m, k1*a)

            Jm_k1a_p = jvp(m, k1*a)
            Jm_k2a_p = jvp(m, k2*a)
            H2m_k1a_p = h2vp(m, k1*a)

            Delta = 1j*Jm_k2a*H2m_k1a_p-1j*k2/k1*Jm_k2a_p*H2m_k1a

            am_r_tmp = 1/(1j*Delta)*(Jm_k2a*Jm_k1a_p-k2/k1*Jm_k2a_p*Jm_k1a)

            sigma[p] = sigma[p] + am_r_tmp*np.exp(1j*m*phi[p])
        sigma[p] = 4/(k1*E0**2)*abs(sigma[p])**2

    return sigma
