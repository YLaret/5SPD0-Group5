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
m_max = 15 # +1 of what comes out of the tolerance function, otherwise it is not working, m should be m>>ka
phi = np.linspace(-np.pi, np.pi, 400)
significance = 1e-1 # tolerance
omega = 2*np.pi*3e8/wavelength
eps0 = 8.854187e-12         # [F/m]
mu0 = 4*np.pi*10**-7        # [N*A^(-2)]
mu1 = mu0  # zie schrift voor omschrijven

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
Z1 = 1/Y1

### SUBPROBLEMS
## 1
# set homogeneous PEC interior like NASA paper
n1 = np.sqrt(1)
n2 = np.sqrt(1e8)
k1 = 2*np.pi/wavelength *n1
k2 = k1 * n2

# Compute fields
am_r, am_t = f.compute_complex_amplitudes(k1, k2, m_max, a)
Ez, Escat = f.compute_E_z(rho, phi, k1, k2, am_r, am_t, E0, a)
H_phi = f.compute_H_phi(rho, phi, k1, k2, am_r, am_t, E0, Y1, a)
sigma_scat = f.compute_sigma_scat(rho, phi, Escat, E0)   # functie uit reader

# Plots
f.plot_am_log(am_r, m_max)
f.plot_sigma(phi, sigma_scat, wavelength)

## 2
rho = (a + b) / 2
# ANALYTICAL
# Set PEC coating like NASA paper
n1 = np.sqrt(1)
n2 = np.sqrt(2)
k1 = omega * np.sqrt(mu0 * n1**2 * eps0)
k2 = omega * np.sqrt(mu0 * n2**2 * eps0)

Y1 = k1/(omega*mu1)
Y2 = np.sqrt(1/mu1)

# Compute fields
am_r, am_c = f.compute_complex_amplitudes_coated_PEC(k1, k2, m_max, a,b)
Ez_cP, Escat_cP, Ein_cP = f.compute_E_z_coated_PEC(rho, phi, k1, k2, am_r, am_c, E0, a, b)
sigma_scat_cP = f.compute_sigma_scat(rho, phi, Escat_cP, E0)   # functie uit reader

# NUMERICAL
def n_function(r):
    return n2

# Calculate the hybrid field (numerical inside, analytical outside)
Ez_hybrid = f.compute_fields_hybrid(rho, phi, k1, n1, n_function, a, b, m_max, E0, Y1)

# Calculate the analytical field for comparison
Ez_analytical, _, _ = f.compute_E_z_coated_PEC(rho, phi, k1, k2, am_r, am_c, E0, a, b)

# 3. Plot - They should now overlap in absolute value and phase!
# plt.figure(figsize=(10,6))
# plt.plot(phi, np.real(Ez_hybrid), label="Hybrid (Num/Analyt)")
# plt.plot(phi, np.real(Ez_analytical), "--", label="Analytical Reference")
# plt.title(f"Comparison of Total Real Field at rho={rho}")
# plt.xlabel("Phi (rad)")
# plt.ylabel("Re(Ez)")
# plt.legend()
# plt.show()

## 3
rho_3 = 6*wavelength
def n_function(r, n1, n2, rho_3):
    epsr = (n2**2 - n1**2) * (1 - rho_3**2 / a**2) + n1**2
    return epsr

Ez_hybrid_3 = f.compute_fields_graded_index(rho_3, phi, k1, n1, n2, n_function, a, b, m_max, E0, Y1)

# Plot for comparison
plt.figure(figsize=(10,6))
plt.plot(phi, np.real(Ez_hybrid), label="Hybrid (Num/Analyt)")
plt.plot(phi, np.real(Ez_analytical), "--", label="Analytical Reference")
plt.plot(phi, np.real(Ez_hybrid_3), ".", label="Graded index")

plt.title(f"Comparison of Total Real Field at rho={rho_3}")
plt.xlabel("Phi (rad)")
plt.ylabel("Re(Ez)")
plt.legend()
plt.show()

# |E|^2/E0 plot
Ez_plot = abs(Ez_hybrid_3)**2 / E0
plt.figure(figsize=(10,6))
plt.plot(phi, np.real(Ez_plot))
plt.title(f"Ez for graded-index cylinder at rho={rho_3}")
plt.xlabel("Phi (rad)")
plt.ylabel("Re(Ez)")
plt.grid()
plt.show()

# # compute integrate fields
# epsr = np.sqrt(2)

# def n_func(rho):
#     if rho < b:
#         return 1e8 # PEC
#     elif rho < a:
#         return np.sqrt(2) # coating
#     else:
#         return 1

# Ez_cP_int, Hphi_cP_int = f.compute_fields_coated_PEC(rho, phi, omega, k1, n1, n_func, Y1, a, b, m_max, eps0, mu0)

# # je moet niet de absolute waarde nemen en ook niet normaliseren want daardoor liep ik hier op vast
# plt.figure()
# plt.plot(phi / (2*np.pi) * 360, np.abs(Ez_cP)/np.max(np.abs(Ez_cP)), "--")
# plt.plot(phi / (2*np.pi) * 360, np.abs(Ez_cP_int)/np.max(np.abs(Ez_cP_int)))
# plt.xlabel(r'$\phi$ [deg]')
# plt.ylabel(r'$|E_z|/\text{max}(E_z)$ [-]')

# plt.title(r'Analytical and numerical @ $\rho=5$')
# plt.legend(["Analytical", "Numerical"])
# #logisch dat die sigma plots niet overeen komen omdat je bij de analytical
# #alleen Escat hebt gebruikt en bij die integrated alles.

# sigma_num = f.compute_sigma_scat(rho, phi, Ez_cP_int, E0)
# # plots
# f.plot_sigma(phi, sigma_scat_cP, wavelength)

# # only call show at end (helps showing all plots at once on OS X)
# plt.show()
