#!/usr/bin/python3
import scipy
import numpy as np
# Name of the student: Rohn Chatterjee
# integral (a)

from matplotlib import pyplot as plt
from scipy.integrate import quad

# function parameter and function definition
alpha = 0.5
def f1(x): return np.exp(-alpha * (x ** 2))
def f2(x): return x * np.exp(-alpha * (x ** 2))

# Fourier Transform


def cquad(f, a, b, **kargs):
    return quad(lambda x: np.real(f(x)), a, b)[0] + \
        1j * quad(lambda x: np.imag(f(x)), a, b)[0]


FT_factor = 1 / np.sqrt(2 * np.pi)


def FT(f): return lambda s: FT_factor * \
    cquad(lambda x: f(x) * np.exp(1j * s * x), -np.inf, np.inf)


# Plotting
s_range = np.arange(-10, 10, step=0.05)

F1_s = [FT(f1)(s) for s in s_range]
F2_s = [FT(f2)(s) for s in s_range]

figure = plt.gcf()
figure.set_size_inches(8.3, 11)

plt.subplot(2, 1, 1)
plt.title(r"Fourier Transform of $e^{-\alpha x^2}$")
plt.plot(s_range, np.real(F1_s), label='real')
plt.plot(s_range, np.imag(F1_s), label='imaginary')
plt.grid()
plt.xlabel(r's $\rightarrow$')
plt.ylabel(r'F(s) $\rightarrow$')
plt.legend()

plt.subplot(2, 1, 2)
plt.title(r"Fourier Transform of $x e^{-\alpha x^2}$")
plt.plot(s_range, np.real(F2_s), label='real')
plt.plot(s_range, np.imag(F2_s), label='imaginary')
plt.grid()
plt.xlabel(r's $\rightarrow$')
plt.ylabel(r'F(s) $\rightarrow$')
plt.legend()

filename = 'Plt_a.pdf'
plt.savefig(filename)
print("Plot saved to {} !".format(filename))
