# Name of the student: Rohn Chatterjee
# Name of the Program GramSchmit Orthogonalisation (User Input)

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
            Vec -= np.dot(Vec, qVec) * qVec  # / np.linalg.norm(qVec)
        Vec = VecNorm(Vec)
        VecSet.append(Vec)

    return np.array(VecSet)


if __name__ == "__main__":
    orthogonal_arrays = eval(input("Enter linearly independent vector(S): "))
    orthogonal_arrays = np.array(orthogonal_arrays, dtype=float)
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

Enter linearly independent vector(S): [1, 2, 0], [8, 1, -6], [0, 0, 1]

Given Vectors (rows):
[[ 1.  2.  0.]
 [ 8.  1. -6.]
 [ 0.  0.  1.]]

Orthogonal Matrix (Q):
[[ 0.4472136   0.89442719  0.        ]
 [ 0.66666667 -0.33333333 -0.66666667]
 [ 0.59628479 -0.2981424   0.74535599]]

 Q · Q^T:
[[1. 0. 0.]
 [0. 1. 0.]
 [0. 0. 1.]]
"""
