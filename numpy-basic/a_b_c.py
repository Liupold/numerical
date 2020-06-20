# Name of the student: Rohn Chatterjee
# Basic Numpy(linalg)

import numpy as np

print("-" * 30 + 'a' + '-' * 30)
"""
A = [(2, 2, 2),(2, 2, 2)]
A = [(5, 0, 0),(0, 5, 0),(0, 0, 5)]
C = [(0), (0), (0)]
"""

# create A
A = 2 * np.ones((2, 3))
print("Matrix A:"); print(A)

#create B
B = 5 * np.eye(3)
print("\nMatrix B:"); print(B)

#create C
C = np.zeros((3, 1))
print("\nMatrix C:"); print(C)

# i) A . B
print("\nA . B:"); print(np.dot(A, B))

# ii) D
D = np.vstack((A, np.transpose(C)))
print("\nD:"); print(D)

# iii) E
E = B * D
print("\nE:"); print(E)

# iv) Replace col of E with C
E[:, 1] = np.transpose(C)
print("\nE (after col replaced):"); print(E)

################################################################################
# b)
print("-" * 30 + 'b' + '-' * 30)

int_array = np.arange(100, 220, 10).reshape((6, 2))
print("\nArray (6 x 2):"); print(int_array)

print("\nSplited into 3"); \
        print("1:\n{}\n2:\n{}\n3:\n{}\n".format(*np.split(int_array, 3)))


################################################################################
#c)
print("-" * 30 + 'c' + '-' * 30)

dagger = lambda x: np.transpose(np.conj(x))

arr = np.array([3, 2j, 6, 7, 4, 5j, 2, 6, 3]).reshape((3, 3))
print("\nArray (3x3):"); print(arr)

M = (arr + dagger(arr)) / 2

print("\nHermitian matrix(M) from Array (3x3):"); print(M)
print("\nM == M^H (Hermitian check)", (M == dagger(M)).all())

U = np.linalg.qr(arr)[0]
print("\nUnitary Matrix (U) from Array:"); print(U)
print("\nU . U^H (Unitary check):"); print(np.round(np.dot(U, dagger(U)), 15))

"""
OUTPUT:
------

------------------------------a------------------------------
Matrix A:
[[2. 2. 2.]
 [2. 2. 2.]]

Matrix B:
[[5. 0. 0.]
 [0. 5. 0.]
 [0. 0. 5.]]

Matrix C:
[[0.]
 [0.]
 [0.]]

A . B:
[[10. 10. 10.]
 [10. 10. 10.]]

D:
[[2. 2. 2.]
 [2. 2. 2.]
 [0. 0. 0.]]

E:
[[10.  0.  0.]
 [ 0. 10.  0.]
 [ 0.  0.  0.]]

E (after col replaced):
[[10.  0.  0.]
 [ 0.  0.  0.]
 [ 0.  0.  0.]]
------------------------------b------------------------------

Array (6 x 2):
[[100 110]
 [120 130]
 [140 150]
 [160 170]
 [180 190]
 [200 210]]

Splited into 3
1:
[[100 110]
 [120 130]]
2:
[[140 150]
 [160 170]]
3:
[[180 190]
 [200 210]]

------------------------------c------------------------------

Array (3x3):
[[3.+0.j 0.+2.j 6.+0.j]
 [7.+0.j 4.+0.j 0.+5.j]
 [2.+0.j 6.+0.j 3.+0.j]]

Hermitian matrix(M) from Array (3x3):
[[3. +0.j  3.5+1.j  4. +0.j ]
 [3.5-1.j  4. +0.j  3. +2.5j]
 [4. +0.j  3. -2.5j 3. +0.j ]]

M == M^H (Hermitian check) True

Unitary Matrix (U) from Array:
[[-0.38100038+0.j          0.3556715 -0.31417649j -0.68041382+0.40824829j]
 [-0.88900089-0.j          0.09484573+0.12448502j  0.40824829-0.13608276j]
 [-0.25400025-0.j         -0.86546731+0.03556715j -0.40824829-0.13608276j]]

U . U^H (Unitary check):
[[ 1.+0.j -0.+0.j  0.-0.j]
 [-0.-0.j  1.+0.j -0.+0.j]
 [ 0.+0.j -0.-0.j  1.+0.j]]
"""
