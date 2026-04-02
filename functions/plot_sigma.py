import matplotlib.pyplot as plt
import numpy as np

def plot_sigma(phi, sigma, wavelength, ylim=None, filename=None):
    sig_lab = 10*np.log10(sigma/wavelength)
    fig, ax = plt.subplots(1)
    plt.plot(phi/2/np.pi*360, sig_lab, color='0')
    plt.xlim(0,180)
    if ylim:
        plt.ylim(ylim)
    plt.xlabel(r'$\phi$ [deg]')
    plt.ylabel(r'$\sigma/\lambda$ [dB]')
    plt.title(r'$\sigma/\lambda$ as a function of $\phi$')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)

    if filename:
        plt.savefig(filename + '.eps', format='eps')
