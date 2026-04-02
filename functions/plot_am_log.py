import matplotlib.pyplot as plt
import numpy as np

def plot_am_log(am_r, m_max, filename=None):
    am_r = np.abs(am_r[m_max::])
    m = np.arange(0, m_max )
    
    fig, ax = plt.subplots(1)
    plt.plot(m, am_r,color='0')
    plt.yscale('log')
    plt.xlim(0, m_max-1)
    plt.xlabel('$m$')
    plt.ylabel('$|a_m^r|$')
    plt.title('$|a_m^r|$ as a function of $m$ (log-scale)')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)

    if filename:
        plt.savefig(filename + '.eps', format='eps')
