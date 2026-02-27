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
n2 = np.sqrt(1e8)
k1 = 2*np.pi/wavelength *n1 #w/c = 2pi/lambda
#k1 = 2*np.pi / wavelength
k2 = k1 * n2    # because epsilon_r = 2
a = wavelength
rho = 1 # should near infinite for sigma to be far field approx
# m should be m>>ka
m_max = 40  # +1 of what comes out of the tolerance function, otherwise it is not working
phi = np.linspace(-np.pi, np.pi, m_max*2)
significance = 1e-1 # tolerance 
omega = 2*np.pi*3e8/wavelength
mu = 4*np.pi*10**-7   #mu0 voor gebruikt
mu1 = 1/3e8  # zie schrift voor omschrijven
Y1 = k1/(omega*mu1)
Y2 = np.sqrt(1/mu1)  # epsilon 1 = 1
E0 = 1

# EM-FUNCTIONS
def compute_E_z(rho, phi, k1, k2, m_max, E0, a):
    Ez = 0j
    
    for m in range(-m_max, m_max):
        # Delta and amplitudes
        Jm_k2a = jv(m,k2*a)
        H2m_k1a_p = h2vp(m,k1*a)
        
        Jm_k2a_p = jvp(m,k2*a)
        H2m_k1a = hankel2(m,k1*a)
        
        Jm_k1a_p = jvp(m,k1*a)
        Jm_k1a = jv(m,k1*a)
        
        Delta = 1j*Jm_k2a*H2m_k1a_p-1j*k2/k1*Jm_k2a_p*H2m_k1a
        
        am_r = 1/(1j*Delta)*(Jm_k2a*Jm_k1a_p-k2/k1*Jm_k2a_p*Jm_k1a)
        am_t = 2/(np.pi*k1*a*Delta)
        
        # Angular harmonic coefficients
        Jm_k1rho = jv(m, k1*rho)
        Jm_k2rho = jv(m, k2*rho)
        H2m_k1rho = hankel2(m, k1*rho)

        em_i = (-1j)**m*Jm_k1rho
        em_r = am_r*(-1j)**m*H2m_k1rho
        em_t = am_t*(-1j)**m*Jm_k2rho
        
        if rho>=a:
            Ez += np.exp(1j*m*phi)*E0*(em_i+em_r)
        elif rho<=a:
            Ez += np.exp(1j*m*phi)*E0*em_t
            
    return Ez

def compute_H_phi(rho, phi, k1, k2, m_max, E0, a, Y1):
    H_phi = 0j
    for m in range(-m_max, m_max):
        # Delta and amplitudes
        Jm_k2a = jv(m,k2*a)
        H2m_k1a_p = h2vp(m,k1*a)
        
        Jm_k2a_p = jvp(m,k2*a)
        H2m_k1a = hankel2(m,k1*a)
        
        Jm_k1a_p = jvp(m,k1*a)
        Jm_k1a = jv(m,k1*a)
        
        Delta = 1j*Jm_k2a*H2m_k1a_p-1j*k2/k1*Jm_k2a_p*H2m_k1a
        
        am_r = 1/(1j*Delta)*(Jm_k2a*Jm_k1a_p-k2/k1*Jm_k2a_p*Jm_k1a)
        am_t = 2/(np.pi*k1*a*Delta)
        
        # Angular harmonic coefficients
        Jm_k1rho_p = jvp(m, k1*rho)
        Jm_k2rho_p = jvp(m, k2*rho)
        H2m_k1rho_p = h2vp(m, k1*rho)
        
        hm_i = (-1j)**m*(k1/(1j*k1))*Jm_k1rho_p                    #omega*mu*Y1=k1
        hm_r = am_r*(-1j)**m*(k1/(1j*k1))*H2m_k1rho_p
        hm_t = am_t*(-1j)**m*(k2/(1j*k1))*Jm_k2rho_p
        
        if rho>=a:
            H_phi += np.exp(1j*m*phi)*E0*Y1*(hm_i+hm_r)
        elif rho<=a:
            H_phi += np.exp(1j*m*phi)*E0*Y1*hm_t
        
        
    return H_phi

def compute_complex_amplitudes(k1, k2, a, m_max):
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
"""
def compute_sigma(phi, k1, k2, m_max, E0, a):
    # infty approx
    rho = 1e0
    Ez = compute_E_z(rho, phi, k1, k2, m_max, E0, a)
    sigma = 2*np.pi*rho*np.abs(Ez)**2/(E0**2)
    #sigma = 2 * np.pi * np.abs(Ez)**2/E0**2 * rho # ik denk dat rho er nog wel inmoet want onze rho gaat niet naar oneindig
    return sigma
""";
def compute_sigma(phi, k1, k2, a, m_max, E0):

    sigma = np.zeros(len(phi),dtype=np.complex128)
    
    for p in range(len(phi)):
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

            sigma[p] = sigma[p] + am_r_tmp*np.exp(1j*m*phi[p])
        sigma[p] = 4/(k1*E0**2)*abs(sigma[p])**2

    return sigma

def min_m_order(am_r, m_max, significance):
    order = np.where(np.abs(am_r[m_max::]) <= significance)[0][0]
    if order == 0:
        return m_max
    else:
        return order

### PLOTTING
def plot_am_log(am_r, m_max):
    am_r = np.abs(am_r[m_max::])
    m = np.arange(0, m_max )

    plt.plot(m, am_r)
    plt.yscale('log')
    plt.xlabel('$m$')
    plt.ylabel('$|a_m^r|$')
    plt.title('$|a_m^r|$ as a function of $m$ (log-scale)')
    plt.grid()
    plt.show()


def plot_sigma(phi, sigma, wavelength):
    sig_lab = 20*np.log10(sigma/wavelength)
    plt.plot(phi, sig_lab)
    plt.xlabel(r'$\phi$ [rad]')
    plt.ylabel(r'$\sigma/\lambda$ [dB]')
    plt.title(r'$\sigma/\lambda$ as a function of $\phi$')
    plt.grid()
    plt.show()
    

## 1
am_r, am_t = compute_complex_amplitudes(k1, k2, a, m_max)
Ez = compute_E_z(rho, phi, k1, k2, m_max, E0, a)
H_phi = compute_H_phi(rho, phi, k1, k2, m_max, E0, a, Y1)
sigma = compute_sigma(phi, k1, k2, a, m_max, E0)

plot_am_log(am_r, m_max)
plot_sigma(phi, sigma, wavelength)

## 2



"""
plt.plot(Ez.real, Ez.imag, label = 'Ez')
plt.plot(H_phi.real, H_phi.imag, label='H_phi')
plt.xlabel('phi in radians')
plt.ylabel('The electromagnetic field')  # check in welke eenheid dit is
plt.title('The electromagnetic field')
plt.legend()
plt.show()

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
""";
