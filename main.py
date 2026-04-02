# -*- coding: utf-8 -*-
"""
Created on Tue Feb 24 13:07:33 2026

@author: group 5: 2-D scattering by dielectric circular cylinder 
"""
import functions as f
import numpy as np
import matplotlib.pyplot as plt
import time

# CONSTANTS
c0 = 299792458                          # [m/s]
mu0 = 4*np.pi*1e-7                      # [N*A^(-2)] (pre-SI revision of 2019)
eps0 = 1/(mu0*c0**2)                    # [F/m]

# GLOBAL PARAMETERS
# precision
m_max = 60
phi = np.linspace(-np.pi, np.pi, 400)
significance = 1e-1

### SUBPROBLEMS
## 1
# (a)
# LOCAL PARAMETERS
# wave characteristics
wavelength = 1.6                        # [m]
omega = 2*np.pi*c0/wavelength           # [rad/s]
E0 = 1                                  # [V/m]

# cylinder definition
a = wavelength                          # [m]

n1 = np.sqrt(1)                         # [-] (open space)
n2 = np.sqrt(1e21)                      # [-] (inside cylinder, pseudo PEC)
k1 = omega*np.sqrt(mu0*n1**2*eps0)      # [1/m]
k2 = omega*np.sqrt(mu0*n2**2*eps0)      # [1/m]

Y1 = k1/(omega*mu0)                     # [rad*N/(m*A^2)]

# evaluation point
rho = 1e6                               # [m] (pseudo-infinity)

# SOLVE
am_r, am_t = f.compute_complex_amplitudes(k1, k2, m_max, a)
Ez, Escat = f.compute_E_z(rho, phi, k1, k2, am_r, am_t, E0, a)
H_phi = f.compute_H_phi(rho, phi, k1, k2, am_r, am_t, E0, Y1, a)
sigma_scat = f.compute_sigma_scat(rho, phi, Escat, E0)

# PLOT
f.plot_am_log(am_r, m_max, '1_am_log')
f.plot_sigma(phi, sigma_scat, wavelength, ylim=[-15,25], filename='1_sigma_PEC')

# (b)
# LOCAL PARAMETERS
# wave characteristics
wavelength = 1.6                        # [m]
omega = 2*np.pi*c0/wavelength           # [rad/s]
E0 = 1                                  # [V/m]

# cylinder definition
a = wavelength                          # [m]

n1 = np.sqrt(1)                         # [-] (open space)
n2 = np.sqrt(2)                         # [-] (inside cylinder)
k1 = omega*np.sqrt(mu0*n1**2*eps0)      # [1/m]
k2 = omega*np.sqrt(mu0*n2**2*eps0)      # [1/m]

Y1 = k1/(omega*mu0)                     # [rad*N/(m*A^2)]

# evaluation point
rho = 1e6                               # [m] (pseudo-infinity)

# SOLVE
am_r, am_t = f.compute_complex_amplitudes(k1, k2, m_max, a)
Ez, Escat = f.compute_E_z(rho, phi, k1, k2, am_r, am_t, E0, a)
H_phi = f.compute_H_phi(rho, phi, k1, k2, am_r, am_t, E0, Y1, a)
sigma_scat = f.compute_sigma_scat(rho, phi, Escat, E0)

# PLOT
f.plot_am_log(am_r, m_max, '1_am_log')
f.plot_sigma(phi, sigma_scat, wavelength, filename='1_sigma_coating')

## 2
# (a)
# LOCAL PARAMETERS
# wave characteristics
wavelength = 1.6e-6                     # [m]
omega = 2*np.pi*c0/wavelength           # [rad/s]
E0 = 1                                  # [V/m]

# cylinder definition
a = wavelength * 3/2                    # [m]
b = wavelength

n1 = np.sqrt(1)                         # [-] (open space)
n2 = np.sqrt(2)                         # [-] (coating)
k1 = omega*np.sqrt(mu0*n1**2*eps0)      # [1/m]
k2 = omega*np.sqrt(mu0*n2**2*eps0)      # [1/m]

Y1 = k1/(omega*mu0)                     # [rad*N/(m*A^2)]

# evaluation point
rho = 1e6                               # [m] (pseudo-infinity)

# SOLVE
# analytical
am_r, am_c = f.compute_complex_amplitudes_coated_PEC(k1, k2, m_max, a,b)
Ez_cP_ana, Escat_cP, Ein_cP = f.compute_E_z_coated_PEC(rho, phi, k1, k2, am_r, am_c, E0, a, b)
sigma_scat_cP = f.compute_sigma_scat(rho, phi, Escat_cP, E0)

#PLOT
f.plot_sigma(phi, sigma_scat_cP, wavelength, ylim=[-15, 20], filename='2_sigma_coatedPEC')

# (b)
# LOCAL PARAMETERS
# wave characteristics
wavelength = 1.6e-6                     # [m]
omega = 2*np.pi*c0/wavelength           # [rad/s]
E0 = 1                                  # [V/m]

# cylinder definition
a = wavelength * 3/2                    # [m]
b = wavelength

n1 = np.sqrt(1)                         # [-] (open space)
n2 = np.sqrt(2)                         # [-] (coating)
k1 = omega*np.sqrt(mu0*n1**2*eps0)      # [1/m]
k2 = omega*np.sqrt(mu0*n2**2*eps0)      # [1/m]

Y1 = k1/(omega*mu0)                     # [rad*N/(m*A^2)]

# evaluation point
rho = (a + b)/2                         # [m]

# SOLVE
# analytical
am_r, am_c = f.compute_complex_amplitudes_coated_PEC(k1, k2, m_max, a,b)
Ez_cP_ana, Escat_cP, Ein_cP = f.compute_E_z_coated_PEC(rho, phi, k1, k2, am_r, am_c, E0, a, b)
sigma_scat_cP = f.compute_sigma_scat(rho, phi, Escat_cP, E0)

# numerical
def n_function(n1, n2, rho):
    return n2

Ez_cP_hybrid = f.compute_fields_numerical(rho, phi, k1, n1, n2, n_function, a, b, m_max, E0, Y1)

#PLOT
f.plot_comparison_Emethod(phi, rho, Ez_cP_ana, Ez_cP_hybrid, E0, '2_methodComparison')


## 3
# LOCAL PARAMETERS
# wave characteristics
wavelength = 1.6e-6                     # [m]
omega = 2*np.pi*c0/wavelength           # [rad/s]
E0 = 1                                  # [V/m]

# cylinder definition
a = wavelength * 6                      # [m]
b = wavelength * 1e-2                   # [m] (pseudo-zero)

n1 = np.sqrt(1)                         # [-] (open space)
n2 = np.sqrt(2)                         # [-] (inside cylinder)
k1 = omega*np.sqrt(mu0*n1**2*eps0)      # [1/m]
k2 = omega*np.sqrt(mu0*n2**2*eps0)      # [1/m]

Y1 = k1/(omega*mu0)                     # [rad*N/(m*A^2)]

# evaluation point
rho = a                                 # [m]

# Lüneberg lens
def n_function(n1, n2, rho):
    epsr = (n2**2 - n1**2) * (1 - rho**2 / a**2) + n1**2
    return epsr

#SOLVE
Ez_gI_6 = f.compute_fields_numerical(rho, phi, k1, n1, n2, n_function, a, b, m_max, E0, Y1)

# PLOT
f.plot_E_boundary(phi, rho, Ez_gI_6, E0, wavelength, '3_boundary_a6')

## 4
# LOCAL PARAMETERS
# wave characteristics
wavelength = 1.6e-6                     # [m]
omega = 2*np.pi*c0/wavelength           # [rad/s]
E0 = 1                                  # [V/m]

# cylinder definition
a = wavelength * 24                      # [m]
b = wavelength * 1e-2                   # [m] (pseudo-zero)

n1 = np.sqrt(1)                         # [-] (open space)
n2 = np.sqrt(2)                         # [-] (inside cylinder)
k1 = omega*np.sqrt(mu0*n1**2*eps0)      # [1/m]
k2 = omega*np.sqrt(mu0*n2**2*eps0)      # [1/m]

Y1 = k1/(omega*mu0)                     # [rad*N/(m*A^2)]

# evaluation point
rho = a                                 # [m]

# Lüneberg lens
def n_function(n1, n2, rho):
    epsr = (n2**2 - n1**2) * (1 - rho**2 / a**2) + n1**2
    return epsr

# SOLVE
Ez_gI_24 = f.compute_fields_numerical(rho, phi, k1, n1, n2, n_function, a, b, m_max, E0, Y1)

# PLOT
f.plot_E_boundary(phi, rho, Ez_gI_6, E0, wavelength, '4_boundary_a24')

# TIMETRIAL
# scaling factors to test
scalings = [6, 24]
results_time = {}

for scale in scalings:
    a = scale * wavelength
    
    start_time = time.time()
    
    Ez_gI = f.compute_fields_numerical(rho, phi, k1, n1, n2, n_function, a, b, m_max, E0, Y1)
    
    end_time = time.time()
    duration = end_time - start_time
    results_time[scale] = duration
    
    print(f'Radius = {scale}lambda,\tTime = {duration:.4f} seconds')

# calculate scaling ratio
ratio = results_time[24] / results_time[6]
print(f'The computational effort increased by a factor of: {ratio:.2f}')

## 5
# wave characteristics
wavelength = 1.55e-6                    # [m]
omega_p = 2*np.pi*c0/wavelength         # [rad/s]
E0 = 1                                  # [V/m]

# bandwidth
Tmax = 200e-15                          # [s]
domega = 2*np.pi/Tmax                   # [rad/s]
N_omega = 256                           # [-]
dt = 2*np.pi / (N_omega * domega)
t = np.arange(N_omega)*dt

omega_list = omega_p + (np.arange(N_omega) - N_omega//2) * domega

# cylinder definition
a = 6*wavelength                        # [m]

n1 = np.sqrt(1)                         # [-] (open space)
n2 = np.sqrt(2)                         # [-] (inside cylinder)

# evaluation point
rho = a                                 # [m]

# SOLVE
Ez_omega = []

for i, omega in enumerate(omega_list):
    k1 = omega*np.sqrt(mu0*n1**2*eps0)
    k2 = omega*np.sqrt(mu0*n2**2*eps0)
    
    am_r, am_t = f.compute_complex_amplitudes(k1, k2, m_max, a)
    Ez, Escat = f.compute_E_z(rho, phi, k1, k2, am_r, am_t, E0, a)
    
    ricker_spectrum = 4*np.sqrt(np.pi)*omega**2/(omega_p**3) * np.exp(-(omega/omega_p)**2)
    Ez_omega.append(Ez * ricker_spectrum)

    phase = np.exp(-1j*omega*(a*np.cos(phi)+2*a)/c0)
    Ez_omega[i] *= phase

Ez_omega = np.array(Ez_omega)           # (N_omega, N_phi)

Ez_omega = np.fft.ifftshift(Ez_omega, axes=0)
Ez_time = np.fft.ifft(Ez_omega, axis=0)

# PLOT
f.plot_E_animation(phi, rho, Ez_time, E0, wavelength, t, N_omega,'5_whispering_gallery')

###
plt.show()
