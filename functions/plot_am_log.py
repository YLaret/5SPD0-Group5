import matplotlib.pyplot as plt
import numpy as np

def plot_am_log(am_r, m_max):
    am_r = np.abs(am_r[m_max::])
    m = np.arange(0, m_max )
    
    plt.figure()
    plt.plot(m, am_r)
    plt.yscale('log')
    plt.xlabel('$m$')
    plt.ylabel('$|a_m^r|$')
    plt.title('$|a_m^r|$ as a function of $m$ (log-scale)')
    plt.grid()
