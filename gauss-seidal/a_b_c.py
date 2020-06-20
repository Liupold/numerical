# Author: Rohn Chatterjee
# Gauss Seidal (with automatic coeff)

import numpy as np
import sys
from itertools import permutations

# A = np.array([[2, 2, 1], [4, 2, 3], [1, 1, 1]], dtype=float)
# B = np.array([6, 4, 0], dtype=float)
# G = ""

def solve_by_g_seidel(A, B, G=None, ACCURACY_UPTO=6):
    tol = 10 ** -(ACCURACY_UPTO + 1)
    N = len(A)
    if not G:
        G = np.zeros(N)
    guess_array = G.reshape((N, 1))
    prev_guess = np.zeros(N).reshape((N, 1))

    #   determine which eqn to solve from which row
    x_to_solve = list(range(N))
    max_pos_map = np.zeros((N, N))

    for i, row in enumerate(A):
        max_pos_map[i] = (row == row.max())

    for x_to_solve in permutations(x_to_solve):
        all_possible = True
        for i, ind in enumerate(x_to_solve):
            all_possible = all_possible and max_pos_map[i, ind]
        if all_possible:
            break
    else:
        print("Cannot determine the required coeff condition")
        sys.exit()

    while True:
        prev_guess = np.copy(guess_array)
        for i, row in enumerate(A):
            n = x_to_solve[i]
            x_n = (B[i] - np.dot(np.delete(row, n), \
                    np.delete(guess_array, n))) / row[n]
            guess_array[n] = x_n
        if (prev_guess == guess_array).all():
            break
        if ((guess_array - prev_guess) < tol).all():
            guess_array = np.round(guess_array, ACCURACY_UPTO)
            break
    return guess_array.reshape(1, N)[0]


def get_inv(A, G=None):
    N = len(A)
    ident = np.identity(N)
    inv_matrix = np.zeros((N, N))

    for i, row in enumerate(ident):
        inv_matrix[:, i] = \
                solve_by_g_seidel(A, row, G=G)

    return inv_matrix


def pivot_sort(A, i):
    changed = False
    col = list(A[:,i][i:])
    col = [abs(i) for i in col]
    swp_index = (col.index(max(col))) + i
    A[[i, swp_index]] = A[[swp_index, i]]
    if i != swp_index:
        changed =True
    return A, changed


def upper_transform(A):
    N = len(A)
    pcount = 0
    for i in range(0, N - 1):
        A, is_changed = pivot_sort(A, i)
        if is_changed:
            pcount += 1
        for j in range(i + 1, N):
            factor = A[j][i] / A[i][i]
            A[j] = A[j] - (factor * A[i])
    return A, pcount


def get_det(matrix, pcount):
    N = len(matrix)
    det = (-1) ** pcount
    for i in range(N):
        det *= matrix[i][i]
    return det

if __name__ == '__main__':
    A = np.array(eval(input("Input Matrix (A) of AX=B: ")), dtype=float)
    B = np.array(eval(input("Input Matrix (B) of AX=B: ")), dtype=float)
    G = input("Input Initial Guess array (delfault=[0, 0, ..., 0]): ")
    if G != "":
        G = np.array(eval(G))
    else:
        G = None

    ##### a ######
    print(f'matrix A:\n{A}, \n\nMatrix B:\n {B}')
    det_A = get_det(*upper_transform(np.copy(A)))
    print('\nDeterminant of A(coeff matrix) :', det_A)

    if det_A == 0:
        print("Determinant is zero can't calculate any further")
        sys.exit()

    x_array = solve_by_g_seidel(A, B, G)
    ###### b #######
    print('[x1, x2, x3, ... xn]:', x_array)
    print("\nInverse of the matrix A (Gauss-Seidal):")
    inv_A = get_inv(A, G)
    print(inv_A)
    print("\nInverse of the matrix A (numpy):")
    print(np.linalg.inv(A))

    ###### c ########
    det_inv_A = get_det(*upper_transform(inv_A))
    print('\nDeterminant of the inverse(A): ', det_inv_A)
    print('(1 / det (A)) == det(inv A):', (1 / det_A) == det_inv_A)

"""
OUTPUT
------

Input Matrix (A) of AX=B: [2, 2, 1], [4, 2, 3], [1, 1, 1]
Input Matrix (B) of AX=B: [6, 4, 0]
Input Initial Guess array (delfault=[0, 0, ..., 0]):
matrix A:
[[2. 2. 1.]
 [4. 2. 3.]
 [1. 1. 1.]],

Matrix B:
 [6. 4. 0.]

Determinant of A(coeff matrix) : -2.0
[x1, x2, x3, ... xn]: [ 5.  1. -6.]

Inverse of the matrix A (Gauss-Seidal):
[[ 0.5  0.5 -2. ]
 [ 0.5 -0.5  1. ]
 [-1.  -0.   2. ]]

Inverse of the matrix A (numpy):
[[ 0.5  0.5 -2. ]
 [ 0.5 -0.5  1. ]
 [-1.   0.   2. ]]

Determinant of the inverse(A):  -0.5
(1 / det (A)) == det(inv A): True
"""
