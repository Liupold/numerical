# Name of the student: Rohn Chatterjee
# Name of the Program GramSchmit Orthogonalisation (From input file)

import numpy as np

def VecNorm(vector: np.ndarray):
    return vector / np.linalg.norm(vector)

def GramSchmit(Rn: np.ndarray):
    """
    Rn is a n dimensional array.
    Where each row is a linearly independent vector.

    And return a orthogonal matrix
    """
    VecSet = []

    for Vec_ in Rn:
        Vec = np.copy(Vec_)
        for qVec in VecSet:
            Vec -= np.dot(Vec, qVec) * qVec # / np.linalg.norm(qVec)
        Vec = VecNorm(Vec)
        VecSet.append(Vec)

    return np.array(VecSet)


if __name__ == "__main__":
    file_ = input("Enter File name: ")
    orthogonal_arrays = np.loadtxt(file_)
    Q = GramSchmit(orthogonal_arrays)
    print("\nGiven Vectors (rows):")
    print(orthogonal_arrays)

    print("\nOrthogonal Matrix (Q):")
    print(Q)

    print("\n Q · Q^T:")
    print(np.dot(Q, np.transpose(Q)))


"""
OUTPUT:
------
Enter File name: data.dat

Given Vectors (rows):
[[ 1. -1.  1.  1.]
 [ 1.  0.  1.  0.]
 [ 0.  1.  0.  1.]]

Orthogonal Matrix (Q):
[[ 0.5        -0.5         0.5         0.5       ]
 [ 0.5         0.5         0.5        -0.5       ]
 [ 0.          0.70710678  0.          0.70710678]]

 Q · Q^T:
[[1. 0. 0.]
 [0. 1. 0.]
 [0. 0. 1.]]
"""