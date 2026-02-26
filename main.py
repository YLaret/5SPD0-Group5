# -*- coding: utf-8 -*-
"""
Created on Tue Feb 24 13:07:33 2026

@author: group 5: 2-D scattering by dielectric circular cylinder 
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import jv, jvp, hankel2, h2vp
# jv = Bessel function of the first kind
# hankel2 = Hankel function of the second kind
# jvp = derivative of Bessel function of the first kind
# h2p = derivative of Hankel function of the second kind

# PARAMETERS
wavelength = 1
n1 = np.sqrt(1)
n2 = np.sqrt(2)
k1 = 2*np.pi/wavelength *n1 #w/c = 2pi/lambda
#k1 = 2*np.pi / wavelength
k2 = k1 * n2    # because epsilon_r = 2
a = wavelength
rho = 1 # should near infinite for sigma to be far field approx
# m should be m>>ka
m_max = 20  # +1 of what comes out of the tolerance function, otherwise it is not working
phi = np.linspace(-np.pi, np.pi, m_max*2)
significance = 1e-1 # tolerance 
omega = 2*np.pi*3e8/wavelength
mu = 4*np.pi*10**-7   #mu0 voor gebruikt
mu1 = 1/3e8  # zie schrift voor omschrijven
Y1 = k1/(omega*mu1)
Y2 = np.sqrt(1/mu1)  # epsilon 1 = 1
E0 = 1

# EM-FUNCTIONS
def compute_E_z(rho, phi, k1, k2, am_r, am_t, m_max, E0, a):
    x1 = k1 * rho
    x2 = k2 * rho

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
        
        if rho>=a:
            Ez += np.exp(1j*m*phi)*E0*(em_i+em_r)     # only the scattered field (outside the cylinder). rho<a does not contribute to far-field scattering width sigma
        elif rho<=a:
            Ez += np.exp(1j*m*phi)*E0*em_t
        
        
    return Ez

def compute_H_phi(rho, phi, k1, k2, am_r, am_t, m_max, E0, a, Y1):
    x1 = k1 * rho
    x2 = k2 * rho

    H_phi = 0j
    for m in range(-m_max, m_max):
        # Their derivatives
        Jm_x1_p = jvp(m, x1)
        Jm_x2_p = jvp(m, x2)
        Hm_x1_p = h2vp(m, x1)

        # Angular harmonic coefficients
        hm_i = (-1j)**m*(k1/(1j*k1))*Jm_x1_p                    #omega*mu*Y1=k1
        hm_r = am_r[m_max+m]*(-1j)**m*(k1/(1j*k1))*Hm_x1_p
        hm_t = am_t[m_max+m]*(-1j)**m*(k2/(1j*k1))*Jm_x2_p
        
        if rho>=a:
            H_phi += np.exp(1j*m*phi)*E0*Y1*(hm_i+hm_r)     # only the scattered field (outside the cylinder). rho<a does not contribute to far-field scattering width sigma
        elif rho<=a:
            H_phi += np.exp(1j*m*phi)*E0*Y1*hm_t
        
        
    return H_phi

def compute_complex_amplitudes(k1, k2, rho, m_max):
    # Bessel/Hankel function arguments
    x1 = k1 * rho
    x2 = k2 * rho

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

def compute_sigma(Ez, E0, phi, rho):
    sigma = 2 * np.pi * np.abs(Ez)**2/E0**2 * rho # ik denk dat rho er nog wel inmoet want onze rho gaat niet naar oneindig
    return sigma


### PLOTTING
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




def compute_sigma_plot(phi, sigma, wavelength):
    sig_lab = sigma/wavelength
    plt.plot(phi, sig_lab)
    plt.xlabel('phi in radians')
    plt.ylabel('sigma/lambda')
    plt.title('sigma/labda plot as a function of phi')
    plt.grid()
    plt.show()
    

## 1
am_r, am_t = compute_complex_amplitudes(k1, k2, rho, m_max)
compute_log_scale_plot(am_r, m_max)
#morder = min_m_order(am_r, m_max, significance)
#print("The complex amplitude reached significance at:", morder)
Ez = compute_E_z(rho, phi, k1, k2, am_r, am_t, m_max, E0, a)
H_phi = compute_H_phi(rho, phi, k1, k2, am_r, am_t, m_max, E0, a, Y1)
sigma = compute_sigma(Ez, 1, phi, rho)
compute_sigma_plot(phi, sigma, wavelength)


plt.plot(Ez.real, Ez.imag, label = 'Ez')
plt.plot(H_phi.real, H_phi.imag, label='H_phi')
plt.xlabel('phi in radians')
plt.ylabel('The electromagnetic field')  # check in welke eenheid dit is
plt.title('The electromagnetic field')
plt.legend()
plt.show()

## 2

# LLM-CREATED

def visualize_field(k1, k2, am_r, am_t, m_max, E0, a, Y1):
    # prompted chat gpt code
    # Create 2D grid
    N = 50
    x = np.linspace(-2*a, 2*a, N)
    y = np.linspace(-2*a, 2*a, N)
    X, Y = np.meshgrid(x, y)

    # Convert to polar coordinates
    R = np.sqrt(X**2 + Y**2)
    Phi = np.arctan2(Y, X)

    Ez_field = np.zeros_like(R, dtype=complex)
    H_phi_field = np.zeros_like(R, dtype=complex)

    # Compute field point-by-point
    for i in range(N):
        for j in range(N):
            Ez_field[i, j] = compute_E_z(R[i, j],Phi[i, j],k1,k2,am_r,am_t,m_max,E0, a)
            H_phi_field[i,j] = compute_H_phi(R[i,j], Phi[i,j], k1, k2, am_r, am_t, m_max, E0, a, Y1)
            print(i, j)

    # Plot real part
    plt.figure(figsize=(7,6))
    plt.pcolormesh(X, Y, Ez_field.real, shading='auto')
    plt.colorbar(label='Re(Ez)')
    plt.gca().set_aspect('equal')

    # Draw cylinder boundary
    circle = plt.Circle((0,0), a, color='black', fill=False, linewidth=2)
    plt.gca().add_patch(circle)

    plt.title('Real part of $E_z$ (incident + scattered)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()


    # Plot real part
    plt.figure(figsize=(7,6))
    plt.pcolormesh(X, Y, H_phi_field.real, shading='auto')
    plt.colorbar(label='Re($H_{phi}$)')
    plt.gca().set_aspect('equal')

    # Draw cylinder boundary
    circle = plt.Circle((0,0), a, color='black', fill=False, linewidth=2)
    plt.gca().add_patch(circle)

    plt.title('Real part of $H_{phi}$ (incident + scattered)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

visualize_field(k1, k2, am_r, am_t, m_max, E0, a, Y1)
