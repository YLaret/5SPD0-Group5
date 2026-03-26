# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 15:07:38 2026

@author: 20213134
"""

import numpy as np
from scipy.special import jv, h2vp, hankel2
from scipy.integrate import solve_ivp

def compute_fields_rk45(rho, phi, eps0, mu0, epsr, E0, omega, a, b, m_max):
    k0 = omega * np.sqrt(eps0 * mu0)
    eta0 = np.sqrt(mu0 / eps0)
    
    Ez_total = 0j
    Hphi_total = 0j

    def system(r, y, m):
        Ez_m, Hphi_m = y
        # Maxwell in cilindrische coördinaten (TM mode)
        dEz_m = -1j * omega * mu0 * Hphi_m
        # Let op de - 1/r term voor de cilindrische geometrie
        dHphi_m = (1j * omega * eps0 * epsr - 1j * m**2 / (omega * mu0 * r**2)) * Ez_m - (1/r) * Hphi_m
        return [dEz_m, dHphi_m]

    for m in range(-m_max, m_max + 1):
        # --- STAP 1: Integreer de coating 'basis' oplossing ---
        # We starten bij de PEC (r=b) met Ez=0 en een willekeurige Hphi=1
        y_start = [0.0, 1.0 + 0j]
        
        # Integreer van b naar a
        sol_coating = solve_ivp(system, [b, a], y_start, args=(m,), method='RK45', rtol=1e-8)
        
        # Waarden aan de buitenrand van de coating (r=a)
        Ez_a_num = sol_coating.y[0, -1]
        Hphi_a_num = sol_coating.y[1, -1]
        
        # --- STAP 2: Boundary Matching bij r=a ---
        # De invallende golf (plane wave expansion)
        # E_inc = E0 * j**(-m) * J_m(k0*a)
        # H_inc = -j * E0 * j**(-m) * J_m'(k0*a) / eta0
        jm_ka = jv(m, k0 * a)
        jpm_ka = 0.5 * (jv(m-1, k0 * a) - jv(m+1, k0 * a)) # Afgeleide Jm
        
        E_inc_a = E0 * (1j**-m) * jm_ka
        H_inc_a = 1j * E0 * (1j**-m) * jpm_ka / eta0
        
        # De verstrooide golf (Hankel functie voor uitgaande golven)
        Hm_ka = hankel2(m, k0 * a)
        Hpm_ka = 0.5 * (hankel2(m-1, k0 * a) - hankel2(m+1, k0 * a))
        
        # We lossen op: C * Ez_a_num = E_inc_a + B * Hm_ka
        # En:          C * Hphi_a_num = H_inc_a + B * (-j/eta0 * Hpm_ka)
        # Dit geeft de schaalfactor C voor onze numerieke integratie:
        Z_scat = 1j * eta0 * Hm_ka / Hpm_ka # 'Impedantie' van de verstrooide golf
        
        # Schaling coëfficiënt C
        numerator = H_inc_a * Z_scat - E_inc_a
        denominator = Hphi_a_num * Z_scat - Ez_a_num
        C = numerator / denominator
        
        # --- STAP 3: Bereken veld op locatie rho ---
        if rho < b:
            Ez_m_rho, Hphi_m_rho = 0j, 0j
        elif rho <= a:
            # Integreer tot exact rho en schaal met C
            sol_rho = solve_ivp(system, [b, rho], y_start, t_eval=[rho], args=(m,), method='RK45')
            Ez_m_rho = C * sol_rho.y[0, 0]
            Hphi_m_rho = C * sol_rho.y[1, 0]
        else:
            # Buiten de cilinder: E_inc + E_scat
            # B is de scattering amplitude
            B = (C * Ez_a_num - E_inc_a) / Hm_ka
            
            jm_krho = jv(m, k0 * rho)
            jpm_krho = 0.5 * (jv(m-1, k0 * rho) - jv(m+1, k0 * rho))
            Hm_krho = hankel2(m, k0 * rho)
            Hpm_krho = 0.5 * (hankel2(m-1, k0 * rho) - hankel2(m+1, k0 * rho))
            
            Ez_m_rho = E0 * (1j**-m) * jm_krho + B * Hm_krho
            Hphi_m_rho = (1j/eta0) * (E0 * (1j**-m) * jpm_krho + B * Hpm_krho)

        Ez_total += Ez_m_rho * np.exp(1j * m * phi)
        Hphi_total += Hphi_m_rho * np.exp(1j * m * phi)
            
    return Ez_total, Hphi_total