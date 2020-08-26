# Name of the student: Rohn chatterjee

import numpy as np
import sys


def eigen_using_power(A, eigen_vec, accuracy=0.1):
    def check_array(x): return x if isinstance(x, np.ndarray) \
        else np.array(x, dtype=float)
    A = check_array(A)
    eigen_vec = check_array(eigen_vec)
    eigen_val = np.inf
    delta = np.inf
    iter_count = 0

    while delta >= accuracy:
        eigen_vec = np.dot(A, eigen_vec)
        new_eigen_val = max(eigen_vec)
        delta = abs(eigen_val - new_eigen_val) \
            if eigen_val != np.inf else eigen_val  # init on the 0th iter
        eigen_val = new_eigen_val
        eigen_vec /= eigen_val  # scaling
        iter_count += 1

    return eigen_val, eigen_vec


A = [[1, 2, 0], [-2, 1, 2], [1, 3, 1]]
X = (1, 1, 1)

if __name__ == "__main__":
    while True:
        input_ = input("Enter array OR filename: ")
        sys.exit() if input_ == "exit" else 0
        try:
            A = np.loadtxt(input_)
        except OSError:
            A = np.array(eval(input_))

        print("Recived Array:")
        print(A)

        X = eval(input("Enter eigen vec (initial): "))
        lambda_, Xn = eigen_using_power(A, X, accuracy=1E-15)
        print(lambda_, Xn, '\n\n')

"""
OUTPUT
------
Enter array OR filename:  [9, 10, 8], [10, 5, -1], [8, -1, 3]
Recived Array:
[[ 9 10  8]
 [10  5 -1]
 [ 8 -1  3]]
Enter eigen vec (initial): (1, 1, 1)
19.286080513046528 [1.         0.66847086 0.45017149]


Enter array OR filename: matrix.txt
Recived Array:
[[1. 2. 0.]
 [2. 1. 0.]
 [0. 0. 1.]]
Enter eigen vec (initial): (1, 1, 1)
3.0 [1.         1.         0.11111111]


Enter array OR filename: exit
"""
