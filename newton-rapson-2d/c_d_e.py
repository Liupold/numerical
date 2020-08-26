# Name of the student: Rohn Chatterjee
# Newton Raphson ND

import numpy as np
import inspect


def partial(f, X, var, h=1E-15):
    f_x = f(*X)
    X[var] += h
    f_x_h = f(*X)
    return (f_x_h - f_x) / h


def Jacobian(F):
    n = len(F)

    def J(X):
        J_FLAT = []
        for f in F:
            for var in range(n):
                J_FLAT.append(partial(f, X, var))
        return np.array(J_FLAT).reshape(n, n)
    return J


def NewtonRaphson_ND(F, X, tol=1E-15):
    # N-Dimensional Newton Raphson
    J = Jacobian(F)
    n = len(F)
    # X as column matrix
    X = np.array(X).reshape(n, 1)

    def vec_F(X): return np.array([f(*X[:, 0]) for f in F]).reshape(n, 1)
    F_return = vec_F(X)

    while (np.abs(F_return) > tol).any():
        X -= np.dot(np.linalg.inv(J(np.copy(X[:, 0]))), F_return)
        F_return = vec_F(X)

    return X.reshape(n)


def rootChecker(F, root):
    for i, f in enumerate(F):
        print('f{}({}) = {}'.format(1 + i, root, f(*root)))
    return 0


def printFunctions(F):
    for f in F:
        print(inspect.getsource(f))


print('##########################################################################')
print('# c')


def f1(x, y): return (x ** 2) + (y ** 2) - 1
def f2(x, y): return (y ** 2) - (4 * x)


F = [f1, f2]
printFunctions(F)

Guess = [0.5, 0.5]
X = NewtonRaphson_ND(F, Guess)
print('guess =', Guess)
print('root =', X)
rootChecker(F, X)
print()

print('##########################################################################')
print('# d')

a = 1 + 1j
def f1(x, y): return np.real(((x + 1j * y) ** n) - a)
def f2(x, y): return np.imag(((x + 1j * y) ** n) - a)


F = [f1, f2]
printFunctions(F)

# for n = 2
n = 2
print('Analytical Solution(s) for n=2:')
print('\t2^0.25 * e^{1j * (pi/8)}   ~= +1.09868 + 1j * 0.45508')
print('\t2^0.25 * e^{1j * (9*pi/8)} ~= -1.09868 - 1j * 0.45508\n')

Guess = [+1, +0.4]
X = NewtonRaphson_ND(F, Guess)
print('guess =', Guess, '(n = {})'.format(n))
print('root =', X)
rootChecker(F, X)
print()

Guess = [-1, -0.4]
X = NewtonRaphson_ND(F, Guess)
print('guess =', Guess, '(n = {})'.format(n))
print('root =', X)
rootChecker(F, X)
print()

# for n = 3
n = 3
print('Analytical Solution(s) for n=3:')
print('\t2^(1/6) * e^{1j * (pi/12)}    ~= +1.08421 + 1j * 0.29051')
print('\t2^(1/6) * e^{1j * (9*pi/12)}  ~= -0.79370 + 1j * 0.79370')
print('\t2^(1/6) * e^{1j * (17*pi/12)} ~= -0.29051 - 1j * 1.08421\n')

Guess = [+1, +0.3]
X = NewtonRaphson_ND(F, Guess)
print('guess =', Guess, '(n = {})'.format(n))
print('root =', X)
rootChecker(F, X)
print()

Guess = [-0.7, +0.7]
X = NewtonRaphson_ND(F, Guess)
print('guess =', Guess, '(n = {})'.format(n))
print('root =', X)
rootChecker(F, X)
print()

Guess = [-0.3, -1]
X = NewtonRaphson_ND(F, Guess)
print('guess =', Guess, '(n = {})'.format(n))
print('root =', X)
rootChecker(F, X)
print()

print('##########################################################################')
print('# e')


def f1(x, y): return np.real(np.cosh(x + 1j * y))
def f2(x, y): return np.imag(np.cosh(x + 1j * y))


F = [f1, f2]
printFunctions(F)

print('Analytical Solution(s):')
print(
    '\tcosh(z) = 0 => e^{2z} = -1 => z = 0 + 1j * (pi/2 + k * pi); k is a int\n')

Guess = [0, 4.0]
X = NewtonRaphson_ND(F, Guess)
print('guess =', Guess)
print('root =', X)
rootChecker(F, X)
print()

Guess = [0, 7.0]
X = NewtonRaphson_ND(F, Guess)
print('guess =', Guess)
print('root =', X)
rootChecker(F, X)
print()

"""
OUTPUT
------

##########################################################################
# c
f1 = lambda x, y: (x ** 2) + (y ** 2) - 1

f2 = lambda x, y: (y ** 2) - (4 * x)

guess = [0.5, 0.5]
root = [0.23606798 0.97173654]
f1([0.23606798 0.97173654]) = 0.0
f2([0.23606798 0.97173654]) = 7.771561172376096e-16

##########################################################################
# d
f1 = lambda x, y: np.real(((x + 1j * y) ** n) - a)

f2 = lambda x, y: np.imag(((x + 1j * y) ** n) - a)

Analytical Solution(s) for n=2:
        2^0.25 * e^{1j * (pi/8)}   ~= +1.09868 + 1j * 0.45508
        2^0.25 * e^{1j * (9*pi/8)} ~= -1.09868 - 1j * 0.45508

guess = [1, 0.4] (n = 2)
root = [1.09868411 0.45508986]
f1([1.09868411 0.45508986]) = -6.661338147750939e-16
f2([1.09868411 0.45508986]) = -5.551115123125783e-16

guess = [-1, -0.4] (n = 2)
root = [-1.09868411 -0.45508986]
f1([-1.09868411 -0.45508986]) = 2.220446049250313e-16
f2([-1.09868411 -0.45508986]) = 0.0

Analytical Solution(s) for n=3:
        2^(1/6) * e^{1j * (pi/12)}    ~= +1.08421 + 1j * 0.29051
        2^(1/6) * e^{1j * (9*pi/12)}  ~= -0.79370 + 1j * 0.79370
        2^(1/6) * e^{1j * (17*pi/12)} ~= -0.29051 - 1j * 1.08421

guess = [1, 0.3] (n = 3)
root = [1.08421508 0.29051456]
f1([1.08421508 0.29051456]) = 0.0
f2([1.08421508 0.29051456]) = -3.3306690738754696e-16

guess = [-0.7, 0.7] (n = 3)
root = [-0.79370053  0.79370053]
f1([-0.79370053  0.79370053]) = -2.220446049250313e-16
f2([-0.79370053  0.79370053]) = -2.220446049250313e-16

guess = [-0.3, -1] (n = 3)
root = [-0.29051456 -1.08421508]
f1([-0.29051456 -1.08421508]) = -2.220446049250313e-16
f2([-0.29051456 -1.08421508]) = -1.1102230246251565e-16

##########################################################################
# e
f1 = lambda x, y: np.real(np.cosh(x + 1j * y))

f2 = lambda x, y: np.imag(np.cosh(x + 1j * y))

Analytical Solution(s):
        cosh(z) = 0 => e^{2z} = -1 => z = 0 + 1j * (pi/2 + k * pi); k is a int

guess = [0, 4.0]
root = [0.         4.71238898]
f1([0.         4.71238898]) = -1.8369701987210297e-16
f2([0.         4.71238898]) = -0.0

guess = [0, 7.0]
root = [0.         7.85398163]
f1([0.         7.85398163]) = -5.82016719913287e-16
f2([0.         7.85398163]) = 0.0

"""
