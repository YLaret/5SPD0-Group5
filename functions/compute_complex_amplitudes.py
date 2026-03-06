import numpy as np
from scipy.special import jv, jvp, hankel2, h2vp

def compute_complex_amplitudes(k1, k2, m_max, a):
    """
        k1 = wavenumber in the homogeneous exterior domain
        k2 = wavenumber in the homogeneous interior domain
        m_max = resolution
        a = boundary distance
    """
    am_r = []
    am_t = []
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
        am_t_tmp = 2/(np.pi*k1*a*Delta)

        am_r.append(am_r_tmp)
        am_t.append(am_t_tmp)
    
    am_r = np.array(am_r, dtype=complex)
    am_t = np.array(am_t, dtype=complex)

    return am_r, am_t
