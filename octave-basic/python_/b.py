import numpy as np

A = np.random.rand(5, 5)
print("A = \n")
print(A)
B = 0.5 * (A + A.T)
print("B = \n")
print(B)

eig_value, eig_vec = np.linalg.eig(B)
print("\nEigenValues = ")
print(eig_value)
print("\nEigenVectors = ")
print(eig_vec)
