import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import numpy as np

def plot_comparison_Emethod(phi, rho, Ez_ana, Ez_hybrid, E0, filename=None):
    fig, ax = plt.subplots()
    plt.plot(phi, np.abs(Ez_ana)**2/E0**2, color='r', linestyle='dotted',label='Analytical')
    plt.plot(phi, np.abs(Ez_hybrid)**2/E0**2, color='b', linestyle='dashdot', label='Hybrid (Num/Ana)')
    plt.xlim(-np.pi,np.pi)
    plt.xlabel(r'$\phi$ [rad]')
    plt.ylabel(r'$|E|^2/E_0^2$ [-]')
    ax.xaxis.set_major_formatter(plticker.FormatStrFormatter('%g $\\pi$'))
    ax.xaxis.set_major_locator(plticker.MultipleLocator(base=1.0))
    plt.title(f'Comparison of solving methods at $\\rho= $ {rho}')
    plt.legend()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)

    if filename:
        plt.savefig(filename + '.eps', format='eps')

