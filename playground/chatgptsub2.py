# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 11:50:31 2026

@author: 20213134
"""

# -*- coding: utf-8 -*-
"""
Coated PEC cylinder scattering simulation
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import jv, jvp, hankel2, h2vp, yv, yvp

# --------------------------------------------------
# PARAMETERS
# --------------------------------------------------

wavelength = 1

n1 = 1                 # air
n2 = np.sqrt(2)        # coating

k1 = 2*np.pi/wavelength * n1
k2 = 2*np.pi/wavelength * n2

a = wavelength            # PEC radius
b = 3/2 * wavelength      # coating outer radius

m_max = 40
E0 = 1

phi = np.linspace(-np.pi, np.pi, 400)

# --------------------------------------------------
# COMPUTE MODE AMPLITUDES
# --------------------------------------------------

def compute_complex_amplitudes(k1,k2,a,b,m_max):

    am_r = []
    Bm = []
    Cm = []

    for m in range(-m_max,m_max):

        k1b = k1*b
        k2b = k2*b
        k2a = k2*a

        # air
        J1 = jv(m,k1b)
        H1 = hankel2(m,k1b)
        J1p = jvp(m,k1b)
        H1p = h2vp(m,k1b)

        # coating
        J2b = jv(m,k2b)
        Y2b = yv(m,k2b)
        J2bp = jvp(m,k2b)
        Y2bp = yvp(m,k2b)

        J2a = jv(m,k2a)
        Y2a = yv(m,k2a)

        A = np.zeros((3,3),dtype=complex)
        rhs = np.zeros(3,dtype=complex)

        # Ez continuity at r=b
        A[0] = [H1, -J2b, -Y2b]
        rhs[0] = -J1

        # derivative continuity
        A[1] = [k1*H1p, -k2*J2bp, -k2*Y2bp]
        rhs[1] = -k1*J1p

        # PEC boundary Ez(a)=0
        A[2] = [0, J2a, Y2a]
        rhs[2] = 0

        sol = np.linalg.solve(A,rhs)

        am_r.append(sol[0])
        Bm.append(sol[1])
        Cm.append(sol[2])

    return np.array(am_r), np.array(Bm), np.array(Cm)

# --------------------------------------------------
# FIELD COMPUTATION
# --------------------------------------------------

def compute_Ez(rho,phi,k1,k2,am_r,Bm,Cm,m_max,E0,a,b):

    Ez = 0j

    for m in range(-m_max,m_max):

        if rho > b:

            Jm = jv(m,k1*rho)
            Hm = hankel2(m,k1*rho)

            term = (-1j)**m*(Jm + am_r[m+m_max]*Hm)

        elif a < rho <= b:

            Jm = jv(m,k2*rho)
            Ym = yv(m,k2*rho)

            term = (-1j)**m*(Bm[m+m_max]*Jm + Cm[m+m_max]*Ym)

        else:

            term = 0

        Ez += E0 * term * np.exp(1j*m*phi)

    return Ez


# --------------------------------------------------
# SCATTERING WIDTH
# --------------------------------------------------

def compute_sigma(phi,k1,am_r,m_max,E0):

    sigma = np.zeros(len(phi),dtype=float)

    for p in range(len(phi)):

        S = 0

        for m in range(-m_max,m_max):

            S += am_r[m+m_max]*np.exp(1j*m*phi[p])

        sigma[p] = 4/k1 * np.abs(S)**2

    return sigma


# --------------------------------------------------
# FIELD VISUALIZATION
# --------------------------------------------------

def visualize_field(k1,k2,am_r,Bm,Cm,m_max,E0,a,b):

    N = 120
    x = np.linspace(-2*b,2*b,N)
    y = np.linspace(-2*b,2*b,N)

    X,Y = np.meshgrid(x,y)

    R = np.sqrt(X**2 + Y**2)
    Phi = np.arctan2(Y,X)

    Ez_field = np.zeros_like(R,dtype=complex)

    for i in range(N):
        for j in range(N):

            Ez_field[i,j] = compute_Ez(
                R[i,j],Phi[i,j],k1,k2,am_r,Bm,Cm,m_max,E0,a,b
            )

    plt.figure(figsize=(7,6))

    plt.pcolormesh(X,Y,Ez_field.real,shading='auto')
    plt.colorbar(label='Re(Ez)')

    plt.gca().set_aspect('equal')

    circle1 = plt.Circle((0,0),a,color='black',fill=False,linewidth=2)
    circle2 = plt.Circle((0,0),b,color='white',fill=False,linewidth=2)

    plt.gca().add_patch(circle1)
    plt.gca().add_patch(circle2)

    plt.title("Real(Ez) for coated PEC cylinder")

    plt.xlabel("x")
    plt.ylabel("y")

    plt.show()

# --------------------------------------------------
# RUN SIMULATION
# --------------------------------------------------

am_r, Bm, Cm = compute_complex_amplitudes(k1,k2,a,b,m_max)

sigma = compute_sigma(phi,k1,am_r,m_max,E0)

plt.figure()
plt.plot(phi,20*np.log10(sigma/wavelength))
plt.xlabel("phi [rad]")
plt.ylabel("sigma/lambda [dB]")
plt.title("Scattering width")
plt.grid()
plt.show()

visualize_field(k1,k2,am_r,Bm,Cm,m_max,E0,a,b)