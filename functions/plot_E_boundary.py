import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import numpy as np

def plot_E_boundary(phi, rho, Ez, E0, wavelength, filename=None):
    fig, ax = plt.subplots()
    plt.plot(phi, np.abs(Ez)**2/E0**2, color='0')
    plt.xlim(-np.pi,np.pi)
    plt.xlabel(r'$\phi$ [rad]')
    plt.ylabel(r'$|E|^2/E_0^2$ [-]')
    ax.xaxis.set_major_formatter(plticker.FormatStrFormatter('%g $\\pi$'))
    ax.xaxis.set_major_locator(plticker.MultipleLocator(base=1.0))
    plt.title(f'Graded index at boundary ($a= {rho/wavelength} \\lambda$)')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)

    if filename:
        plt.savefig(filename + '.eps', format='eps')

