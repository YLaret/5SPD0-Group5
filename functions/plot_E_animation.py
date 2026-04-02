import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import matplotlib.animation as animation
import numpy as np

def plot_E_animation(phi, rho, Ez_time, E0, wavelength, t, N_omega,filename=None):
    fig, ax = plt.subplots()
    plt.xlim(-np.pi,np.pi)
    plt.xlabel(r'$\phi$ [rad]')
    plt.ylabel(r'$|E|^2/E_0^2$ [-]')
    ax.xaxis.set_major_formatter(plticker.FormatStrFormatter('%g $\\pi$'))
    ax.xaxis.set_major_locator(plticker.MultipleLocator(base=1.0))
    plt.title(f'Homogeneous cylinder at boundary ($a= {rho/wavelength} \\lambda$)')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    
    I = np.abs(Ez_time)**2 / E0**2

    line, = ax.plot(phi, I[0], color='0')

    ax.set_xlim(-np.pi, np.pi)
    ax.set_ylim(0, np.max(I))
    ax.set_xlabel("$\\phi$ [rad]")
    ax.set_ylabel("$|E|^2 / E_0^2$")

    def update(frame):
        line.set_ydata(I[frame])
        ax.set_title(f"t = {t[frame]*1e15:.1f} fs")
        return line,

    ani = animation.FuncAnimation(fig, update, frames=N_omega, interval=0.1)

    if filename:
        writer = animation.PillowWriter(fps=15,
                                        metadata=dict(artist='Me'),
                                        bitrate=1800)
        ani.save(filename+'.gif', writer=writer)

