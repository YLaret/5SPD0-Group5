# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 15:27:46 2026

@author: 20213134
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 17:59:01 2026

@author: 20213134
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import jv, jvp, hankel2, h2vp
# jv = Bessel function of the first kind
# hankel2 = Hankel function of the second kind
# jvp = derivative of Bessel function of the first kind
# h2p = derivative of Hankel function of the second kind

# Initial parameters
wavelength = 1

n1 = np.sqrt(1)  # free space
n2 = np.sqrt(2)  # coating
n3 = np.sqrt(1e8) # PEC

k1 = 2*np.pi/wavelength *n1 #w/c = 2pi/labda
#k1 = 2*np.pi / wavelength
k2 = k1 * n2    # because epsilon_r = 2
k3 = k1 * n3

#a = wavelength
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

a1 = wavelength
a2 = 3*wavelength/2


k = [k1, k2, k3]  # freespace, coating, PEC
a = [a2, a1]      # coating, PEC

def complex_amplitudes(k, a, m_max):
    am_r_tot = []
    am_t_tot = []
    am_t_final = []
    am_r_final = []
    for ai in range(len(a)):
        for ki in range(len(k)-1):
            x1 = k[ki]*a[ai]
            x2 = k[ki+1]*a[ai]
            
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

                delta = 1j * Jm_x2 * Hm_x1_p - 1j * (k[ki+1]/k[ki]) * Jm_x2_p * Hm_x1

                am_r.append(1 / (1j*delta) * (Jm_x2 * Jm_x1_p - k[ki+1]/k[ki] * Jm_x2_p * Jm_x1))
                am_t.append(2 / (np.pi * k[ki] * a[ai] * delta))
        
            am_r = np.array(am_r, dtype=complex)
            am_t = np.array(am_t, dtype=complex)
            
        am_r_tot.append(am_r)
        am_t_tot.append(am_t)
        
    am_r_tot = np.array(am_r_tot, dtype=complex)
    am_t_tot = np.array(am_t_tot, dtype=complex)
        
    for lijst in am_r_tot.T:
        am_r_final.append(sum(lijst))
        
    for lijst2 in am_t_tot.T:    
        am_t_final.append(sum(lijst2))
    
    am_r_final = np.array(am_r_final, dtype=complex)
    am_t_final = np.array(am_t_final, dtype=complex)
    
    
    return am_r_final, am_t_final



def compute_log_scale_plot(am_r, m_max):
    m_list = np.arange(0, m_max )
    #for i in range(len(am_r.T)):
        #am_ri = np.abs(am_r[i][m_max::])
        #plt.plot(m_list, am_ri)
    plt.plot(m_list, np.abs(am_r[m_max::]))
    plt.yscale('log')
    plt.xlabel('Order m')
    plt.ylabel('Complex amplitude a_m^r (log)')
    plt.title('Log-scale plot of a_m^r as a function of m')
    plt.grid()
    plt.show()
    
def compute_E_z(rho, phi, k, am_r, am_t, m_max, E0, a):
    Escat_tot = []
    Ez_tot = []
    
    for ai in range(len(a)):
        for ki in range(len(k)-1):
            x1 = k[ki]*a[ai]
            x2 = k[ki+1]*a[ai]
            Escat = 0j
            Ez = 0j
            em_i = 0j
            em_r = 0j
            em_t = 0j
            for m in range(-m_max, m_max):
                # Definitions of Bessel and Hankel functions
                Jm_x1 = jv(m, x1)   # m = order, x = argument
                Jm_x2 = jv(m, x2)
                Hm_x1 = hankel2(m, x1)

                # Their derivatives
                Jm_x1_p = jvp(m, x1)
                Jm_x2_p = jvp(m, x2)
                Hm_x1_p = h2vp(m, x1)
                
                # Angular harmonic coefficients
                em_i += (-1j)**m*Jm_x1
                em_r += am_r[m+m_max]*(-1j)**m*Hm_x1
                em_t += am_t[m_max+m]*(-1j)**m*Jm_x2
        
    if rho>=a[0]:
        Escat += np.exp(1j*m*phi)*E0*em_r
        Ez += np.exp(1j*m*phi)*E0*(em_i+em_r)     # only the scattered field (outside the cylinder). rho<a does not contribute to far-field scattering width sigma
    elif rho<=a[0]:
        Ez += np.exp(1j*m*phi)*E0*em_t
    else:
        Ez = 0
    
        #Escat_tot.append(Escat)
        #Ez_tot.append(Ez)
        
    #Escat_tot = np.array(Escat_tot, dtype=complex)
    #Ez_tot = np.array(Ez_tot, dtype=complex)
    
    return Ez, Escat



#%% calls
am_r, am_t = complex_amplitudes(k, a, m_max)
#%%    
compute_log_scale_plot(am_r, m_max)
#%%
Ez, Escat = compute_E_z(rho, phi, k, am_r, am_t, m_max, E0, a)
#%%

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

def plot_sigma(phi, sigma, wavelength):
    sig_lab = 10*np.log10(sigma/wavelength)
    plt.figure()
    plt.plot(phi, sig_lab)
    plt.xlabel(r'$\phi$ [rad]')
    plt.ylabel(r'$\sigma/\lambda$ [dB]')
    plt.title(r'$\sigma/\lambda$ as a function of $\phi$')
    plt.grid()


sigma = compute_sigma_scat(rho, phi, Escat, E0)
plot_sigma(phi, sigma, wavelength)

#%%

def visualize_field(k, am_r, am_t, m_max, E0, a, Y1):
    # prompted chat gpt code
    # Create 2D grid
    N = 50
    x = np.linspace(-2*a[-1], 2*a[-1], N)
    y = np.linspace(-2*a[-1], 2*a[-1], N)
    X, Y = np.meshgrid(x, y)

    # Convert to polar coordinates
    R = np.sqrt(X**2 + Y**2)
    Phi = np.arctan2(Y, X)

    Ez_field = np.zeros_like(R, dtype=complex)
    #H_phi_field = np.zeros_like(R, dtype=complex)

    # Compute field point-by-point
    for i in range(N):
        for j in range(N):
            Ez_field[i, j] = compute_E_z(R[i, j],Phi[i, j],k,am_r,am_t,m_max,E0, a)[0]
            #H_phi_field[i,j] = compute_H_phi(R[i,j], Phi[i,j], k1, k2, am_r, am_t, m_max, E0, a, Y1)
            print(i, j)

    # Plot real part
    plt.figure(figsize=(7,6))
    plt.pcolormesh(X, Y, Ez_field[0].real, shading='auto')
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
    #plt.figure(figsize=(7,6))
    #plt.pcolormesh(X, Y, H_phi_field.real, shading='auto')
    #plt.colorbar(label='Re($H_{phi}$)')
    #plt.gca().set_aspect('equal')

    # Draw cylinder boundary
    #circle = plt.Circle((0,0), a, color='black', fill=False, linewidth=2)
    #plt.gca().add_patch(circle)

    #plt.title('Real part of $H_{phi}$ (incident + scattered)')
    #plt.xlabel('x')
    #plt.ylabel('y')
    #plt.show()

visualize_field(k, am_r, am_t, m_max, E0, a, Y1)

#%%

def visualize_field(k, am_r, am_t, m_max, E0, a, Y1):
    # Create 2D grid
    N = 25  # finer grid for better visualization
    x = np.linspace(-2*a[0], 2*a[0], N)
    y = np.linspace(-2*a[0], 2*a[0], N)
    X, Y = np.meshgrid(x, y)

    # Convert to polar coordinates
    R = np.sqrt(X**2 + Y**2)
    Phi = np.arctan2(Y, X)

    Ez_field = np.zeros_like(R, dtype=complex)

    # Compute field point-by-point
    for i in range(N):
        for j in range(N):
            r_ij = R[i, j]
            phi_ij = Phi[i, j]

            # Initialize field at this point
            Ez_point = 0j

            # Loop over cylinders
            for ci, radius in enumerate(a):
                # compute_E_z returns arrays for each cylinder
                Ez_cyl, Escat_cyl = compute_E_z(r_ij, phi_ij, k, am_r, am_t, m_max, E0, a)
                
                if r_ij <= radius:
                    # Inside this cylinder -> take transmitted field
                    Ez_point = Ez_cyl[ci]
                    break  # inner-most cylinder dominates inside
                else:
                    # Outside cylinder -> accumulate scattered fields
                    Ez_point += Escat_cyl[ci]

            # Add incident field outside all cylinders (assume plane wave along x)
            if r_ij > a[-1]:
                Ez_point += E0 * np.exp(1j * k[0] * r_ij * np.cos(phi_ij))

            Ez_field[i, j] = Ez_point
            
            print(i,j)

    # Plot real part
    plt.figure(figsize=(8,7))
    plt.pcolormesh(X, Y, Ez_field.real, shading='auto', cmap='RdBu')
    plt.colorbar(label='Re(Ez)')
    plt.gca().set_aspect('equal')

    # Draw cylinder boundaries
    for radius in a:
        circle = plt.Circle((0,0), radius, color='black', fill=False, linewidth=2)
        plt.gca().add_patch(circle)

    plt.title('Real part of $E_z$ (incident + scattered + transmitted)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()


visualize_field(k, am_r, am_t, m_max, E0, a, Y1)