# -*- coding: utf-8 -*-
"""
Created on Tue Feb 24 13:07:33 2026

@author: group 5: 2-D scattering by dielectric circular cylinder 
"""
import functions as f
import numpy as np
import matplotlib.pyplot as plt
import time

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
n2 = np.sqrt(2)
k1 = omega * np.sqrt(mu0 * n1**2 * eps0)
k2 = omega * np.sqrt(mu0 * n2**2 * eps0)

Y1 = k1/(omega*mu1)
Y2 = np.sqrt(1/mu1)
Z1 = 1/Y1

### SUBPROBLEMS
## 1

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

# Plot for comparison
plt.figure(figsize=(10,6))
plt.plot(phi, np.real(Ez_hybrid), label="Hybrid (Num/Analyt)")
plt.plot(phi, np.real(Ez_analytical), "--", label="Analytical Reference")
plt.title(f"Comparison of Total Real Field at rho={rho}")
plt.xlabel("Phi (rad)")
plt.ylabel("Re(Ez)")
plt.legend()
plt.show()

## 3
rho_3 = 6 * wavelength
def n_function(r, n1, n2, rho_3):
    epsr = (n2**2 - n1**2) * (1 - rho_3**2 / a**2) + n1**2
    return epsr

Ez_hybrid_3 = f.compute_fields_graded_index(rho_3, phi, k1, n1, n2, n_function, a, b, m_max, E0, Y1)

# # Plot for comparison
# plt.figure(figsize=(10,6))
# plt.plot(phi, np.real(Ez_hybrid), label="Hybrid (Num/Analyt)")
# plt.plot(phi, np.real(Ez_analytical), "--", label="Analytical Reference")
# plt.plot(phi, np.real(Ez_hybrid_3), ".", label="Graded index")
# plt.title(f"Comparison of Total Real Field at rho={rho_3}")
# plt.xlabel("Phi (rad)")
# plt.ylabel("Re(Ez)")
# plt.legend()
# plt.show()

# Plot
plt.figure(figsize=(10,6))
plt.plot(phi, np.real(abs(Ez_hybrid_3)**2 / E0))
plt.title(f"Ez for graded-index cylinder at rho={rho_3}")
plt.xlabel("Phi (rad)")
plt.ylabel("Re(Ez)")
plt.grid()
plt.show()

## 4
rho_4 = 24 * wavelength
Ez_hybrid_4 = f.compute_fields_graded_index(rho_4, phi, k1, n1, n2, n_function, a, b, m_max, E0, Y1)

# Plot
plt.figure(figsize=(10,6))
plt.plot(phi, np.real(abs(Ez_hybrid_4)**2 / E0))
plt.title(f"Ez for graded-index cylinder at rho={rho_4}")
plt.xlabel("Phi (rad)")
plt.ylabel("Re(Ez)")
plt.grid()
plt.show()

# Scaling factors to test
scalings = [6, 24]
results_time = {}

for s in scalings:
    a_scaled = s * wavelength
    # IMPORTANT: m_max must scale with 'a' to ensure convergence
    # Rule of thumb: m_max ~ k1 * a + safety_margin
    m_max_scaled = int(np.ceil(k1 * a_scaled) + 15) 
    
    start_time = time.time()
    
    # Run the graded-index calculation
    # Ensure your n_function uses a_scaled correctly inside!
    Ez_result = f.compute_fields_graded_index(
        a_scaled, phi, k1, n1, n2, n_function, a_scaled, b, m_max_scaled, E0, Y1
    )
    
    end_time = time.time()
    duration = end_time - start_time
    results_time[s] = duration
    
    print(f"Radius {s}lambda: m_max={m_max_scaled}, Time={duration:.4f} seconds")

# Calculate scaling ratio
ratio = results_time[24] / results_time[6]
print(f"The computational effort increased by a factor of: {ratio:.2f}")