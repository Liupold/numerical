# Name of the student: Rohn chatterjee
# SHM simulation and phase space

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def SolveByRK4_O2(f1_dash, f2_dash, t0: float,
                  x0: float, x0_dash: float, tn: float, h=0.1):
    """
    This function solves a differential eqn using
    Runge Kutta 4 method (for order 2).
    """
    N = round(abs((tn - t0) / h))
    t_array = np.zeros(N + 1)
    x_array = np.zeros(N + 1)
    x_dash_array = np.zeros(N + 1)

    t_array[0], x_array[0], x_dash_array[0] = t, x, v = t0, x0, x0_dash

    for i in range(1, N + 1):
        k_1x = f1_dash(t, x, v)
        k_1v = f2_dash(t, x, v)

        k_2x = f1_dash(t + (h / 2), x + h * (k_1x / 2), v + h * (k_1v / 2))
        k_2v = f2_dash(t + (h / 2), x + h * (k_1x / 2), v + h * (k_1v / 2))

        k_3x = f1_dash(t + (h / 2), x + h * (k_2x / 2), v + h * (k_2v / 2))
        k_3v = f2_dash(t + (h / 2), x + h * (k_2x / 2), v + h * (k_2v / 2))

        k_4x = f1_dash(t + h, x + h * k_3x, v + h * k_3v)
        k_4v = f2_dash(t + h, x + h * k_3x, v + h * k_3v)

        x = x + (h / 6) * (k_1x + 2 * k_2x + 2 * k_3x + k_4x)
        v = v + (h / 6) * (k_1v + 2 * k_2v + 2 * k_3v + k_4v)
        t += h
        t_array[i], x_array[i], x_dash_array[i] = t, x, v

    return t_array, x_array, x_dash_array


def SolveByEulers_O2(f1_dash, f2_dash, t0: float, x0: float,
                     x0_dash: float, tn: float, h=0.1):
    """
    This function solves a differential eqn using
    Euler's Method. (for order 2)
    """
    N = round(abs((tn - t0) / h))
    t_array = np.zeros(N + 1)
    x_array = np.zeros(N + 1)
    x_dash_array = np.zeros(N + 1)

    t_array[0], x_array[0], x_dash_array[0] = t, x, v = t0, x0, x0_dash

    for i in range(1, N + 1):
        f1, f2 = f1_dash(t, x, v), f2_dash(t, x, v)
        x = x + h * f1
        v = v + h * f2
        t += h
        t_array[i], x_array[i], x_dash_array[i] = t, x, v
    return t_array, x_array, x_dash_array


# system specification
m = 1  # unit mass
k = np.pi ** 2

# differential eqns


def x_dot(t, x, v): return v
def v_dot(t, x, v): return -(k / m) * x


# initial condition
t0 = 0
x0 = 1
v0 = 0

# exact analytical solution


def fn_x(t): return x0 * np.cos(np.sqrt(k / m) * t)
def fn_v(t): return -np.sqrt(k / m) * x0 * np.sin(np.sqrt(k / m) * t)


# calculation limit
tn = 10
dt = 0.02


# exact calculation
t_array_ex = np.append(np.arange(t0, tn, dt), tn)
x_array_ex = fn_x(t_array_ex)
v_array_ex = fn_v(t_array_ex)

# eulers calculation
t_array_eu, x_array_eu, v_array_eu = SolveByEulers_O2(x_dot,
                                                      v_dot, t0, x0, v0, tn, h=dt)
# rk4 calculation
t_array_rk, x_array_rk, v_array_rk = SolveByRK4_O2(x_dot,
                                                   v_dot, t0, x0, v0, tn, h=dt)

with PdfPages('plot_c.pdf') as pdf:

    # analytical plot
    figure = plt.gcf()
    figure.set_size_inches(8.3, 11)

    plt.subplot(2, 1, 1, title='time(t) vs Position(x) (analytical)')
    plt.plot(t_array_ex, x_array_ex)
    plt.xlabel(r"Time (t) $\rightarrow$ ")
    plt.ylabel(r"Position (x) $\rightarrow$ ")
    plt.grid()

    plt.subplot(2, 1, 2, title='Position(x) vs Momentum(mv) (analytical)')
    plt.plot(x_array_ex, m * v_array_ex)
    plt.xlabel(r"Position (x) $\rightarrow$ ")
    plt.ylabel(r"Momentum (mv) $\rightarrow$ ")
    plt.grid()
    pdf.savefig()

    # eulers plot
    plt.subplot(2, 1, 1, title='time(t) vs Position(x) (eulers)')
    plt.plot(t_array_ex, x_array_ex, label="analytical")
    plt.plot(t_array_eu, x_array_eu, label="eulers")
    plt.xlabel(r"Time (t) $\rightarrow$ ")
    plt.ylabel(r"Position (x) $\rightarrow$ ")
    plt.legend()
    plt.grid()

    plt.subplot(2, 1, 2, title='Position(x) vs Momentum(mv) (eulers)')
    plt.plot(x_array_ex, m * v_array_ex, label="analytical")
    plt.plot(x_array_eu, m * v_array_eu, label="eulers")
    plt.xlabel(r"Position (x) $\rightarrow$ ")
    plt.ylabel(r"Momentum (mv) $\rightarrow$ ")
    plt.grid()
    plt.legend()
    pdf.savefig()

    # rk4 plot
    plt.subplot(2, 1, 1, title='time(t) vs Position(x) (rk4)')
    plt.plot(t_array_ex, x_array_ex, label="analytical")
    plt.plot(t_array_rk, x_array_rk, label="Rk4")
    plt.xlabel(r"Time (t) $\rightarrow$ ")
    plt.ylabel(r"Position (x) $\rightarrow$ ")
    plt.legend()
    plt.grid()

    plt.subplot(2, 1, 2, title='Position(x) vs Momentum(mv) (rk4)')
    plt.plot(x_array_ex, m * v_array_ex, label="analytical")
    plt.plot(x_array_rk, m * v_array_rk, label="Rk4")
    plt.xlabel(r"Position (x) $\rightarrow$ ")
    plt.ylabel(r"Momentum (mv) $\rightarrow$ ")
    plt.legend()
    plt.grid()
    pdf.savefig()

print('saved to plot_c.pdf')
