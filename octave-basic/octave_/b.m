A = rand(5, 5)
B = 0.5 * (A + A')
[eigen_vec, eigen_val] = eig(B);
EigenValues = diag(eigen_val);
EigenVectors = eigen_vec;

EigenValues
EigenVectors
