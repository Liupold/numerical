# Name of the program
# Gauss Elimination (with partial pivot)
import numpy as np

# A = np.array([[2, 2, 1], [4, 2, 3], [1, 1, 1]])
# B = np.array([6, 4, 0])


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
            A[j] = A[j] - factor * A[i]
    return A, pcount


def get_x_array(C):
    N = len(A)
    x_arr = np.zeros(N)
    x_arr[N - 1] = C[N - 1][N] / C[N - 1][N - 1]
    for i in range(N - 2, -1, -1):
        x_arr[i] = (C[i][N] - np.sum(C[i][i + 1:N] * x_arr[i + 1:N])) / C[i][i]
    return x_arr

def get_det(matrix, pcount):
    N = len(matrix)
    det = (-1) ** pcount
    for i in range(N):
        det *= matrix[i][i]
    return det

def SolveByG_Elemination(matrix_A, matrix_B):
    N = len(matrix_A)
    C = np.zeros((N, N + 1))
    C[:, 0:N], C[:, N] = matrix_A, matrix_B
    C, pcount = upper_transform(C)
    det = get_det(C[:, 0:N], pcount)
    if det != 0:
        X_array = get_x_array(C)
    else:
        X_array = []
    return X_array, C, det

if __name__ == '__main__':
    A = np.array(eval(input("Enter A matrix: ")))
    B = np.array(eval(input("Enter B matrix: ")))
    x_arr, c_matrix, det = SolveByG_Elemination(A, B)
    print("Determinant: {}".format(det))
    print("Determinant (from numpy): {}".format(np.linalg.det(A)))
    print("\nArgument matrix:")
    print(c_matrix)
    print('\n[x1, x2, x3 ... xn]:', x_arr)


"""
OUTPUT:
------

Enter A matrix: [2, 2, 1], [4, 2, 3], [1, 1, 1]
Enter B matrix: [6, 4, 0]
Determinant: -2.0
Determinant (from numpy): -2.0

Argument matrix:
[[ 4.   2.   3.   4. ]
 [ 0.   1.  -0.5  4. ]
 [ 0.   0.   0.5 -3. ]]

[x1, x2, x3 ... xn]: [ 5.  1. -6.]
"""
