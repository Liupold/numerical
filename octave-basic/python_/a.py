"""
solve harmonic oscillator (diff eqn).
"""
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def harmonicFunc(g, t, gma, omega):
    """
    diffential eqn of the harmonic motion.
    """
    x = g[0]
    v = g[1]  # y = dx/dt
    dxdt = v
    dvdt = - 2 * gma * v - (omega ** 2) * x  # dv/dt
    return [dxdt, dvdt]


t = np.linspace(0, 10, 100)
OMEGA = 2
FILENAME = 'harmonic-oscillator-python.pdf'

with PdfPages(FILENAME) as pdf:

    figure = plt.gcf()
    figure.set_size_inches(8.3, 15)

    for gma in {0.1, 2, 4}:
        sol = odeint(harmonicFunc, (2, 1), t, args=(gma, OMEGA))
        x = sol[:, 0]
        v = sol[:, 1]

        plt.subplot(3, 1, 1,
                    title=r'time(t) vs Position(x) [$\gamma={}, \omega={}$]'
                    .format(gma, OMEGA))
        plt.plot(t, x)
        plt.xlabel(r"Time (t) $\rightarrow$ ")
        plt.ylabel(r"Position (x) $\rightarrow$ ")
        plt.grid(linestyle='-', linewidth=2)

        plt.subplot(3, 1, 2,
                    title=r'time(t) vs Velocity(v) [$\gamma={}, \omega={}$]'
                    .format(gma, OMEGA))
        plt.plot(t, v)
        plt.xlabel(r"Time (t) $\rightarrow$ ")
        plt.ylabel(r"Velocity (v) $\rightarrow$ ")
        plt.grid(linestyle='-', linewidth=2)

        plt.subplot(3, 1, 3,
                    title=r'Position(x) vs Velocity(v) [$\gamma={}, \omega={}$]'
                    .format(gma, OMEGA))
        plt.plot(x, v)
        plt.xlabel(r"Position (x) $\rightarrow$ ")
        plt.ylabel(r"Velocity (v) $\rightarrow$ ")
        plt.grid(linestyle='-', linewidth=2)
        pdf.savefig()
        print("Ploted For: gammma={}, omega={}".format(gma, OMEGA))
        plt.clf()
