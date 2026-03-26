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
m_max = 13 # +1 of what comes out of the tolerance function, otherwise it is not working, m should be m>>ka
phi = np.linspace(-np.pi, np.pi, 400)
significance = 1e-1 # tolerance
omega = 2*np.pi*3e8/wavelength
eps0 = 8.854187e-12         # [F/m]
mu0 = 4*np.pi*10**-7        # [N*A^(-2)]
mu1 = 1/3e8  # zie schrift voor omschrijven

E0 = 1

a = wavelength*3/2
b = wavelength
rho = 1e5 # pseudo infinity

n1 = np.sqrt(1)
n2 = np.sqrt(1e9)
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
Z1 = 1/Y1
Y2 = np.sqrt(1/mu1)

# compute fields
am_r, am_t = f.compute_complex_amplitudes(k1, k2, m_max, a)
Ez, Escat = f.compute_E_z(rho, phi, k1, k2, am_r, am_t, E0, a)
H_phi = f.compute_H_phi(rho, phi, k1, k2, am_r, am_t, E0, Y1, a)
sigma_scat = f.compute_sigma_scat(rho, phi, Escat, E0)   # functie uit reader

# plots
f.plot_am_log(am_r, m_max)
f.plot_sigma(phi, sigma_scat, wavelength)
plt.show()


#%%
## 2
# ANALYTICAL
# set PEC coating like NASA paper
n1 = np.sqrt(1)
n2 = np.sqrt(2)
k1 = 2*np.pi/wavelength *n1
k2 = k1 * n2

Y1 = k1/(omega*mu1)
Y2 = np.sqrt(1/mu1)

# compute fields
am_r, am_c = f.compute_complex_amplitudes_coated_PEC(k1, k2, m_max, a,b)
Ez_cP, Escat_cP, Ein_cP = f.compute_E_z_coated_PEC(rho, phi, k1, k2, am_r, am_c, E0, a, b)
sigma_scat_cP = f.compute_sigma_scat(rho, phi, Escat_cP, E0)   # functie uit reader

# NUMERICAL

def n_function(r):   # hier kunnen we later de gradient in stoppen
    return n2
# compute integrate fields
rho = 10    #1.7 is de sweet spot
epsr = np.sqrt(2)

Ez_cP_int, Hphi_cP_int, soly = f.compute_fields_coated_PEC(rho, phi, k1, n1, n_function, a, b, m_max)

# hier wordt de bessel en hankel dus wel voor het buitenveld meegenomen,
#daarom is de plot overlappend bij grote rho omdat dat een hybride oplossing geeft
Ez_cP_int2 = Ez_cP_int+Ein_cP+Escat_cP

# je moet niet de absolute waarde nemen en ook niet normaliseren want daardoor liep ik hier op vast
plt.plot(phi / (2*np.pi) * 360,
         (Ez_cP_int2))
plt.plot(phi / (2*np.pi) * 360,
         (Ez_cP), "--")
plt.title("Analytical and numerical")
plt.legend(["Numerical", "Analytical"])
plt.show()


#logisch dat die sigma plots niet overeen komen omdat je bij de analytical 
#alleen Escat hebt gebruikt en bij die integrated alles.

sigma_num = f.compute_sigma_scat(rho, phi, Ez_cP_int, E0)
# plots
f.plot_sigma(phi, sigma_scat_cP, wavelength)
f.plot_sigma(phi, sigma_num, wavelength)

# only call show at end (helps showing all plots at once on OS X)
plt.show()
