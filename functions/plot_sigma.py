import matplotlib.pyplot as plt
import numpy as np

def plot_sigma(phi, sigma, wavelength):
    sig_lab = 10*np.log10(sigma/wavelength)
    plt.figure()
    plt.plot(phi/2/np.pi*360, sig_lab)
    plt.xlim(0,180)
    plt.xlabel(r'$\phi$ [deg]')
    plt.ylabel(r'$\sigma/\lambda$ [dB]')
    plt.title(r'$\sigma/\lambda$ as a function of $\phi$')
    plt.grid()
