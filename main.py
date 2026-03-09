# -*- coding: utf-8 -*-
"""
Created on Tue Feb 24 13:07:33 2026

@author: group 5: 2-D scattering by dielectric circular cylinder 
"""
import functions as f
import numpy as np
import matplotlib.pyplot as plt

# PARAMETERS
wavelength = 1
m_max = 60  # +1 of what comes out of the tolerance function, otherwise it is not working, m should be m>>ka
phi = np.linspace(-np.pi, np.pi, m_max*2)
significance = 1e-1 # tolerance
omega = 2*np.pi*3e8/wavelength
mu = 4*np.pi*10**-7   #mu0 voor gebruikt
mu1 = 1/3e8  # zie schrift voor omschrijven

E0 = 1

a = wavelength*3/2
b = wavelength
rho = 1e5 # pseudo infinity

n1 = np.sqrt(1)
n2 = np.sqrt(1e8)
k1 = 2*np.pi/wavelength *n1
k2 = k1 * n2

Y1 = k1/(omega*mu1)
Y2 = np.sqrt(1/mu1)  # epsilon 1 = 1

### SUBPROBLEMS
## 1
# set homogeneous PEC interior like NASA paper:
n1 = np.sqrt(1)
n2 = np.sqrt(1e8)
k1 = 2*np.pi/wavelength *n1
k2 = k1 * n2

Y1 = k1/(omega*mu1)
Y2 = np.sqrt(1/mu1)

# compute fields
am_r, am_t = f.compute_complex_amplitudes(k1, k2, m_max, a)
Ez, Escat = f.compute_E_z(rho, phi, k1, k2, am_r, am_t, E0, a)
H_phi = f.compute_H_phi(rho, phi, k1, k2, am_r, am_t, E0, Y1, a)
sigma_scat = f.compute_sigma_scat(rho, phi, Escat, E0)   # functie uit reader

# plots
f.plot_am_log(am_r, m_max)
f.plot_sigma(phi, sigma_scat, wavelength)

## 2
# set PEC coating like NASA paper
n1 = np.sqrt(1)
n2 = np.sqrt(2)
k1 = 2*np.pi/wavelength *n1
k2 = k1 * n2

Y1 = k1/(omega*mu1)
Y2 = np.sqrt(1/mu1)

# compute fields
am_r, am_c = f.compute_complex_amplitudes_coated_PEC(k1, k2, m_max, a,b)
Ez_cP, Escat_cP = f.compute_E_z_coated_PEC(rho, phi, k1, k2, am_r, am_c, E0, a, b)
sigma_scat_cP = f.compute_sigma_scat(rho, phi, Escat_cP, E0)   # functie uit reader
# plots
f.plot_sigma(phi, sigma_scat_cP, wavelength)

# only call show at end (helps showing all plots at once on OS X)
plt.show()
