# Name of the student: Rohn Chatterjee
# integral (b)

import numpy as np
from scipy.integrate import quad

# Given


def g(x, mu, sigma): return (1 / (sigma * np.sqrt(2 * np.pi))) * \
    np.exp(-0.5 * ((x - mu) / sigma)**2)


mu = 2
sigma_set = [10, 1, 0.1, 0.01, 0.001]


def f(x): return x**2


a = -3


def I(sigma): return quad(lambda x: g(x, mu, sigma) * f(x - a),
                          mu - 10 * sigma, mu + 10 * sigma)[0]


print("f(mu - a) = {:.2E}".format(f(mu - a)))

for sigma in sigma_set:
    ans = I(sigma)
    print("""
  _{}
 /
 | g(x, mu={}, sigma={})f(x - a={}) * dx  =  {:.2E}   (delta = {:.2E})
_/
 {} """.format(mu - 10 * sigma, mu, sigma, a, ans,
               abs(ans - f(mu - a)), mu + 10 * sigma))

"""
OUTPUT:
------

f(mu - a) = 2.50E+01

  _-98
 /
 | g(x, mu=2, sigma=10)f(x - a=-3) * dx  =  1.25E+02   (delta = 1.00E+02)
_/
 102

  _-8
 /
 | g(x, mu=2, sigma=1)f(x - a=-3) * dx  =  2.60E+01   (delta = 1.00E+00)
_/
 12

  _1.0
 /
 | g(x, mu=2, sigma=0.1)f(x - a=-3) * dx  =  2.50E+01   (delta = 1.00E-02)
_/
 3.0

  _1.9
 /
 | g(x, mu=2, sigma=0.01)f(x - a=-3) * dx  =  2.50E+01   (delta = 1.00E-04)
_/
 2.1

  _1.99
 /
 | g(x, mu=2, sigma=0.001)f(x - a=-3) * dx  =  2.50E+01   (delta = 1.00E-06)
_/
 2.01
 """
