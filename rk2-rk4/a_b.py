#Name of the student: Rohn Chatterjee
# runge-kutta

from math import exp
import numpy as np
from matplotlib import pyplot as plt

def SolveByRK4(f_dash, x0:float, y0:float, xn: float, h=0.1):
    """
    This function solves a differential eqn using
    Runge Kutta 4 method.
    """
    N = round((xn - x0) / h)
    x_array = np.zeros(N + 1)
    y_array = np.zeros(N + 1)

    x, y = (x0, y0)
    x_array[0], y_array[0] = x, y
    for i in range(1, N + 1):
        k_1 = h * f_dash(x, y)
        k_2 = h * f_dash(x + (h / 2), y + (k_1 / 2))
        k_3 = h * f_dash(x + (h / 2), y + (k_2 / 2))
        k_4 = h * f_dash(x + h, y + k_3)
        y = y + (1/6) * (k_1 + 2 * k_2 + 2 * k_3 + k_4)
        x += h
        x_array[i], y_array[i] = x, y
    return x_array, y_array


def SolveByRK2(f_dash, x0:float, y0:float, xn: float, h=0.1):
    """
    This function solves a differential eqn using
    Runge Kutta 2 method.
    """
    N = round((xn - x0) / h)
    x_array = np.zeros(N + 1)
    y_array = np.zeros(N + 1)

    x, y = (x0, y0)
    x_array[0], y_array[0] = x, y
    for i in range(1, N + 1):
        k_1 = h * f_dash(x, y)
        k_2 = h * f_dash(x + h, y + k_1)
        y = y + (1/2) * (k_1 + k_2)
        x += h
        x_array[i], y_array[i] = x, y
    return x_array, y_array


def SolveByEulers(f_dash, x0:float, y0:float, xn: float, h=0.1):
    """
    This function solves a differential eqn using
    Euler's Method.
    """
    N = round((xn - x0) / h)
    x_array = np.zeros(N + 1)
    y_array = np.zeros(N + 1)

    x, y = (x0, y0)
    x_array[0], y_array[0] = x, y
    for i in range(1, N + 1):
        y = y + (f_dash(x, y) * h)
        x += h
        x_array[i], y_array[i] = x, y
    return x_array, y_array

y_dash = lambda x, y: np.cos(np.pi * x)
x0, y0 = 0, 0 ; x = 2

h = 0.1
x_eu , y_eu = SolveByEulers(y_dash, x0=x0, y0=y0, xn=x, h=h)
x_rk4 , y_rk4 = SolveByRK2(y_dash, x0=x0, y0=y0, xn=x, h=h)
x_rk2 , y_rk2 = SolveByRK2(y_dash, x0=x0, y0=y0, xn=x, h=h)


plt.subplot(2, 1, 1)
plt.plot(x_rk4, y_rk4, '--', label='RK4 Method')
plt.plot(x_rk2, y_rk2, 'o', label='RK2 Method')
plt.plot(x_eu, y_eu, '-+', label='Euler\'s Method')
plt.title(r'solution of $\frac{d(f)}{dt} = cos(\pi t) $ with h=0.1')
plt.ylabel(r"f(t) $\rightarrow$")
plt.xlabel(r"t $\rightarrow$")
plt.legend()
plt.grid()

h = h / 5
x_eu , y_eu = SolveByEulers(y_dash, x0=x0, y0=y0, xn=x, h=h)
x_rk4 , y_rk4 = SolveByRK2(y_dash, x0=x0, y0=y0, xn=x, h=h)
x_rk2 , y_rk2 = SolveByRK2(y_dash, x0=x0, y0=y0, xn=x, h=h)

plt.subplot(2, 1, 2, title='With h={}'.format(h))
plt.plot(x_rk4, y_rk4, '--', label='RK4 Method')
plt.plot(x_rk2, y_rk2, 'o', label='RK2 Method')
plt.plot(x_eu, y_eu, '-+', label='Euler\'s Method')
plt.title(r'solution of $\frac{d(f)}{dt} = cos(\pi t) $ with h=0.02')
plt.ylabel(r"f(t) $\rightarrow$")
plt.xlabel(r"t $\rightarrow$")
plt.legend()
plt.grid()

figure = plt.gcf()
figure.set_size_inches(8.3, 11)

plt.savefig('plot_a_and_b.pdf', dpi=100)
print('saved to plot_a_and_b.pdf!')