# Name of the student: Rohn chatterjee
# Name of the program: Gauss Jordan

import numpy as np
np.ALLOW_THREADS = 1

# A = np.array([[2, 4, -6], [1, 3, 1], [2, -4, -2]])
# B = np.array([-8, 10, -12])


def pivot_sort(A, i):
    changed = False
    col = list(A[:, i][i:])
    col = [abs(i) for i in col]
    swp_index = (col.index(max(col))) + i
    A[[i, swp_index]] = A[[swp_index, i]]
    if i != swp_index:
        changed = True
    return A, changed


def diagonalize(A):
    N = len(A)
    pcount = 0
    for i in range(0, N):
        A, changed = pivot_sort(A, i)
        if changed:
            pcount += 1
        for j in range(0, N):
            if j != i:
                factor = A[j][i] / A[i][i]
                A[j] = A[j] - factor * A[i]
    return A, pcount


def get_x_array(A):
    # from diagonalize matrix
    N = len(A)
    x_arr = []
    for i, row in enumerate(A):
        x_arr.append(row[N] / row[i])
    return x_arr


def get_det(matrix, pcount):
    N = len(matrix)
    det = (-1) ** pcount
    for i in range(N):
        det *= matrix[i][i]
    return det


def get_inverse(A):
    N = len(A)
    arg_mat = np.zeros((N, 2 * N))
    for i in range(N):
        arg_mat[i, N + i] = 1
    arg_mat[:, 0:N] = A
    arg_mat, _ = diagonalize(arg_mat)

    for i in range(N):
        arg_mat[i] = arg_mat[i] / arg_mat[i, i]

    return arg_mat[:, N:2 * N]


def SolveByGaussJordan(matrix_A, matrix_B):
    N = len(A)
    C = np.zeros((N, N + 1))
    C[:, 0:N], C[:, N] = matrix_A, matrix_B
    C, pcount = diagonalize(C)
    det = get_det(C[:, 0:N], pcount)
    if det != 0:
        X_array = get_x_array(C)
    else:
        print("Determinant is zero can't solve this system")
        X_array = []
    return X_array, C, det


if __name__ == '__main__':
    A = np.array(eval(input("Enter Matrix A: ")))
    B = np.array(eval(input("Enter Matrix B: ")))
    x_arr, diag_matrix, det = SolveByGaussJordan(A, B)
    print("Determinant: {}".format(det))
    print("Determinant (from numpy): {}".format(np.linalg.det(A)))
    print("\nDiagonal matrix:")
    print(diag_matrix)
    print('\n[x1, x2, x3 ... xn]:', x_arr)
    if det != 0:
        inv_A = get_inverse(A)
        print("\n inverse matrix")
        print(inv_A)
        print("\n inverse matrix (Numpy)")
        print(np.linalg.inv(A))

"""
OUTPUT:
------

Enter Matrix A: [2, 4, -6], [1, 3, 1], [2, -4, -2]
Enter Matrix B: -8, 10, -12
Determinant: 72.0
Determinant (from numpy): 72.0

Diagonal matrix:
[[  2.    0.    0.    2. ]
 [  0.   -8.    0.  -16. ]
 [  0.    0.    4.5  13.5]]

[x1, x2, x3 ... xn]: [1.0, 2.0, 3.0]

 inverse matrix
[[-0.02777778  0.44444444  0.30555556]
 [ 0.05555556  0.11111111 -0.11111111]
 [-0.13888889  0.22222222  0.02777778]]

 inverse matrix (Numpy)
[[-0.02777778  0.44444444  0.30555556]
 [ 0.05555556  0.11111111 -0.11111111]
 [-0.13888889  0.22222222  0.02777778]]
"""
