import numpy as np
from scipy.special import jv, jvp, yv, yvp, hankel2, h2vp

def compute_complex_amplitudes_coated_PEC(k1, k2, m_max, a, b):
    """
        k1 = wavenumber in the homogeneous exterior domain
        k2 = wavenumber in the homogeneous interior domain
        m_max = resolution
        a = outer radius
        b = PEC radius
    """
    
    am_r = []
    am_c = []
    
    for m in range(-m_max, m_max):
        # Bessel values at a
        Jm_k1a = jv(m, k1*a)
        H2m_k1a = hankel2(m, k1*a)
        
        Jm_k2a = jv(m, k2*a)
        Ym_k2a = yv(m, k2*a)

        Jm_k1a_p = jvp(m, k1*a)
        H2m_k1a_p = h2vp(m, k1*a)
        
        Jm_k2a_p = jvp(m, k2*a)
        Ym_k2a_p = yvp(m, k2*a)
        
        # PEC condition coefficient
        beta = jv(m, k2*b) / yv(m, k2*b)
        
        F = Jm_k2a - beta*Ym_k2a
        F_p = Jm_k2a_p - beta*Ym_k2a_p
        
        # Linear system
        M = np.array([
            [H2m_k1a, -F],
            [k1*H2m_k1a_p, -k2*F_p]
        ], dtype=complex)
        
        rhs = np.array([
            -Jm_k1a,
            -k1*Jm_k1a_p
        ], dtype=complex)
        
        A_m, B_m = np.linalg.solve(M, rhs)
        
        am_r.append(A_m)
        am_c.append(B_m)
    
    am_r = np.array(am_r, dtype=complex)
    am_c = np.array(am_c, dtype=complex)

    return am_r, am_c
