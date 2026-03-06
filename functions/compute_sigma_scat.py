import numpy as np
from scipy.special import jv, jvp, hankel2, h2vp

def compute_sigma_scat(rho, phi, Escat, E0):
    """
        phi = angle
        k1 = wavenumber in the homogeneous exterior domain
        k2 = wavenumber in the homogeneous interior domain
        am_r, am_t = complex amplitudes
        E0 = wave amplitude
        Y1 = wave admittance
        a = boundary distance
    """
    sigma = 2 * np.pi * (np.abs(Escat))**2/E0**2 * rho # ik denk dat rho er nog wel inmoet want onze rho gaat niet naar oneindig
    return sigma
