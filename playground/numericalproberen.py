import numpy as np
import matplotlib.pyplot as plt

# PARAMETERS
wavelength = 1
phi = np.linspace(-np.pi, np.pi, 400)
omega = 2*np.pi*3e8/wavelength
eps0 = 8.854187e-12         
mu0 = 4*np.pi*1e-7        
mu1 = 1/3e8  

# initial conditions
Ez_ini = 0
Hphi_ini = 1  

epsilon = eps0*np.sqrt(2)
mu = mu0*mu1

# geometry
a = 1.5
b = 1.0
N = 1000
h = (a - b)/N

Ez_phi = []  # Ez for each phi

for ph in phi:
    fun = [Hphi_ini]
    fun1 = [Ez_ini]
    rho = b
    
    for i in range(N):
        # RK4
        k1 = [ 1/rho*fun[i] + 1j*omega*epsilon*fun1[i], -1j*omega*mu*fun[i] ]
        rho2 = rho + h/2
        fun2 = [fun[i] + h/2*k1[0], fun1[i] + h/2*k1[1]]
        k2 = [1/rho2*fun2[0] + 1j*omega*epsilon*fun2[1], -1j*omega*mu*fun2[0]]
        fun3 = [fun[i] + h/2*k2[0], fun1[i] + h/2*k2[1]]
        k3 = [1/rho2*fun3[0] + 1j*omega*epsilon*fun3[1], -1j*omega*mu*fun3[0]]
        rho4 = rho + h
        fun4 = [fun[i] + h*k3[0], fun1[i] + h*k3[1]]
        k4 = [1/rho4*fun4[0] + 1j*omega*epsilon*fun4[1], -1j*omega*mu*fun4[0]]
        
        # update
        fun.append(fun[i] + h/6*(k1[0] + 2*k2[0] + 2*k3[0] + k4[0]))
        fun1.append(fun1[i] + h/6*(k1[1] + 2*k2[1] + 2*k3[1] + k4[1]))
        rho += h
    
    # take Ez at the outer radius (or any rho of interest)
    Ez_phi.append(fun1[-1]*np.exp(1j * i * phi))

# convert to numpy array
Ez_phi = np.array(Ez_phi)

# plot
plt.figure(figsize=(8,5))
plt.plot(phi, Ez_phi.real, label='Re(Ez)')
plt.plot(phi, Ez_phi.imag, label='Im(Ez)')
plt.xlabel('phi [rad]')
plt.ylabel('Ez')
plt.title('Ez vs phi')
plt.legend()
plt.grid(True)
plt.show()


#