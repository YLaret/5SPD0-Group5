import matplotlib.pyplot as plt
import numpy as np

def plot_sigma(phi, sigma, wavelength):
    sig_lab = 10*np.log10(sigma/wavelength)
    plt.figure()
    plt.plot(phi, sig_lab)
    plt.xlabel(r'$\phi$ [rad]')
    plt.ylabel(r'$\sigma/\lambda$ [dB]')
    plt.title(r'$\sigma/\lambda$ as a function of $\phi$')
    plt.grid()
