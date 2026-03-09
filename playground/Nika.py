# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 13:47:43 2026

@author: 20213134
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy.special import jv, jvp, hankel2, h2vp
# jv = Bessel function of the first kind
# hankel2 = Hankel function of the second kind
# jvp = derivative of Bessel function of the first kind
# h2p = derivative of Hankel function of the second kind
from scipy.integrate import solve_ivp


# Initial parameters
wavelength = 1

n1 = np.sqrt(1)  # free space
n2 = np.sqrt(1e9)  # coating

k1 = 2*np.pi/wavelength *n1 #w/c = 2pi/labda
#k1 = 2*np.pi / wavelength
k2 = k1 * n2    # because epsilon_r = 2


a = 1.5 * wavelength
b = wavelength
rho = 1e5 # should near infinite for sigma to be far field approx
# m should be m>>ka

m_max = 40  # +1 of what comes out of the tolerance function, otherwise it is not working
significance = 1e-1 # tolerance 

phi = np.linspace(-np.pi, np.pi, m_max*2)
omega = 2*np.pi*3e8/wavelength

mu = 4*np.pi*10**-7   #mu0 voor gebruikt
mu1 = 1/3e8  # zie schrift voor omschrijven

Y1 = k1/(omega*mu1)
Y2 = np.sqrt(1/mu1)  # epsilon 1 = 1

E0 = 1


def compute_complex_amplitudes(k1, k2, a, m_max):
    # Bessel/Hankel function arguments
    x1 = k1 * a
    x2 = k2 * a

    am_r = []
    am_t = []

    for m in range(-m_max, m_max):
        # Definitions of Bessel and Hankel functions
        Jm_x1 = jv(m, x1)   # m = order, x = argument
        Jm_x2 = jv(m, x2)
        Hm_x1 = hankel2(m, x1)

        # Their derivatives
        Jm_x1_p = jvp(m, x1)
        Jm_x2_p = jvp(m, x2)
        Hm_x1_p = h2vp(m, x1)

        delta = 1j * Jm_x2 * Hm_x1_p - 1j * (k2/k1) * Jm_x2_p * Hm_x1

        am_r.append(1 / (1j*delta) * (Jm_x2 * Jm_x1_p - k2/k1 * Jm_x2_p * Jm_x1))
        am_t.append(2 / (np.pi * k1 * a * delta))
        
    am_r = np.array(am_r, dtype=complex)
    am_t = np.array(am_t, dtype=complex)

    return am_r, am_t


def compute_log_scale_plot(am_r, m_max):
    am_r = np.abs(am_r[m_max::])
    m_list = np.arange(0, m_max )

    plt.plot(m_list, am_r)
    plt.yscale('log')
    plt.xlabel('Order m')
    plt.ylabel('Complex amplitude a_m^r (log)')
    plt.title('Log-scale plot of a_m^r as a function of m')
    plt.grid()
    plt.show()
    
def min_m_order(am_r, m_max, significance):
    
    order = np.where(np.abs(am_r[m_max::]) <= significance)[0][0]
    if order == 0:
        return m_max
    else:
        return order

def compute_E_z(rho, phi, k1, k2, am_r, am_t, m_max, E0, a):
    x1 = k1 * rho
    x2 = k2 * rho
    Escat = 0j
    Ez = 0j
    for m in range(-m_max, m_max):
        # Definitions of Bessel and Hankel functions
        Jm_x1 = jv(m, x1)   # m = order, x = argument
        Jm_x2 = jv(m, x2)
        Hm_x1 = hankel2(m, x1)

        # Angular harmonic coefficients
        em_i = (-1j)**m*Jm_x1
        em_r = am_r[m+m_max]*(-1j)**m*Hm_x1
        em_t = am_t[m_max+m]*(-1j)**m*Jm_x2
        
        if rho >= a:
            Escat += np.exp(1j*m*phi)*E0*em_r
            Ez += np.exp(1j*m*phi)*E0*(em_i+em_r)     # only the scattered field (outside the cylinder). rho<a does not contribute to far-field scattering width sigma
        elif b < rho <= a:
            Ez += np.exp(1j*m*phi)*E0*em_t
        elif rho < b:
            Ez = 0      
        
    return Ez, Escat

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

def plot_sigma(a, b, phi, sigma, wavelength):
    sig_lab = 10*np.log10(sigma/wavelength)
    plt.figure()
    plt.plot(phi, sig_lab)
    plt.xlabel(r'$\phi$ [rad]')
    plt.ylabel(r'$\sigma/\lambda$ [dB]')
    plt.title(r'$\sigma/\lambda$ as a function of $\phi$')
    plt.grid()

def ode_system(rho, y, m_max, omega, mu):
    Ez, Hphi = y
    eps_r = 2.0 
    eps0 = 8.854187e-12
    epsilon = eps_r * eps0

    # Define system of ODE's
    dEz_drho = -1j * omega * mu * Hphi
    dHphi_drho = -1/rho * Hphi - 1j * omega * epsilon * Ez + 1j * m_max**2 / omega * mu * rho**2 * Ez

    return [dEz_drho, dHphi_drho]

def integrate_coated_cylinder(m_max, a, b, omega, mu):
    # Initial value for ODE
    y0 = [0, 1 + 0j]

    # Integrate from b to a
    sol = solve_ivp(ode_system, [b, a], y0, method='RK45', args=(m_max, omega, mu))

    # Return the values of Ez and Hphi at the outer boundary rho = a
    return sol.y[0, -1], sol.y[1, -1]

def compute_step2_sigma(rho, E0, Ez):
    sigma = 2 * np.pi * rho * (np.abs(Ez))**2 / E0**2

    return sigma

def plot_step2_sigma(phi, sigma, wavelength):
    sig_lab = 10*np.log10(sigma/wavelength)
    print('Value for sig/lab:', sig_lab)
    plt.figure()
    plt.plot(phi, sig_lab)
    plt.xlabel(r'$\phi$ [rad]')
    plt.ylabel(r'$\sigma/\lambda$ [dB]')
    plt.title(r'$\sigma/\lambda$ as a function of $\phi$')
    plt.grid()
    plt.show()

Ez_a, Hphi_a = integrate_coated_cylinder(m_max, a, b, omega, mu)
print(f"Ez at boundary a: {Ez_a}")

sigma = compute_step2_sigma(rho, E0, Ez_a)
print('Value for sigma:', sigma)
plot_step2_sigma(phi, sigma, wavelength)