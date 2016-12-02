from __future__ import print_function
import numpy as np
import paralution_wrapper
import scipy.sparse

# Define a symmetric 8 x 8 dense upper triangular matrix first. 
# This matrix is part of the examples which come with Intel's MKL library
# and is used here for historical reasons.

# A:
#     7.0,       1.0,           2.0, 7.0,
#          -4.0, 8.0,      2.0,
#                1.0,                     5.0,
#                     7.0,           9.0,
#                          5.0, 1.0, 5.0,
#                              -1.0,      5.0,
#                                   11.0,
#                                         5.0

print("Test with double precision")
A = np.zeros((8, 8), dtype=np.float64)
A[0, 0] = 7.0
A[0, 2] = 1.0
A[0, 5] = 2.0
A[0, 6] = 7.0

A[1, 1] = -4.0
A[1, 2] = 8.0
A[1, 4] = 2.0

A[2, 2] = 1.0
A[2, 7] = 5.0

A[3, 3] = 7.0
A[3, 6] = 9.0

A[4, 4] = 5.0
A[4, 5] = 1.0
A[4, 6] = 5.0

A[5, 5] = -1.0
A[5, 7] = 5.0

A[6, 6] = 11.0

A[7, 7] = 5.0

# print "Dense matrix:"
print(A)
# Dense matrix to sparse matrix in CSR format
Acsr = scipy.sparse.csr_matrix(A)

print("Sparse upper triangular CSR matrix:")
print("values:  ", Acsr.data)
# Indices are 0 based
print("index:   ", Acsr.indices)
print("pointer: ", Acsr.indptr)

# Convert the upper triangular CSR matrix Acsr to 'full' CSR matrix Acsr_full
Acsr_full = Acsr + Acsr.T - scipy.sparse.diags(Acsr.diagonal())

print()
print("Sparse 'full' CSR matrix:")
print("values:  ", Acsr_full.data)
# Indices are 0 based
print("index:   ", Acsr_full.indices)
print("pointer: ", Acsr_full.indptr)

# initial guess for solution x
x = np.zeros(8, dtype=np.float64)
# right hand side
b = np.ones(8, dtype=np.float64)
y = np.zeros(8, dtype=np.float64)
print("dtype x: ", x.dtype)
print("dtype b: ", b.dtype)
print("dtype A: ", Acsr_full.data.dtype)
info = 1    # make Paralution more verbose
abstol = 1e-6  # convergence tolerance
reltol = 5e-4
divtol = 1e5
max_iter = 10000 # maximum number of iterations

paralution_wrapper.solution(Acsr_full.data, Acsr_full.indices, Acsr_full.indptr,
                            x, b, info, abstol, reltol, divtol, max_iter)

# check solution x with original dense matrix A first
# convert upper triangular matrix AA to 'full' matrix
y = (A + A.T - np.eye(A.shape[0]) * A.diagonal()).dot(x)
assert (np.allclose(b, y))

# check solution  with sparse matrix Acsr_full
y = Acsr_full.dot(x)
assert (np.allclose(b, y))
print("Solution double x:")
print(x)
print()
print("A * x:")
print(y)
print("b:")
print(b)
print()
#stop
print("Test with single precision")
A = np.zeros((8, 8), dtype=np.float32)
A[0, 0] = 7.0
A[0, 2] = 1.0
A[0, 5] = 2.0
A[0, 6] = 7.0

A[1, 1] = -4.0
A[1, 2] = 8.0
A[1, 4] = 2.0

A[2, 2] = 1.0
A[2, 7] = 5.0

A[3, 3] = 7.0
A[3, 6] = 9.0

A[4, 4] = 5.0
A[4, 5] = 1.0
A[4, 6] = 5.0

A[5, 5] = -1.0
A[5, 7] = 5.0

A[6, 6] = 11.0

A[7, 7] = 5.0

# print "Dense matrix:"
print(A)
# Dense matrix to sparse matrix in CSR format
Acsr = scipy.sparse.csr_matrix(A)

print("Sparse upper triangular CSR matrix:")
print("values:  ", Acsr.data)
# Indices are 0 based
print("index:   ", Acsr.indices)
print("pointer: ", Acsr.indptr)

# Convert the upper triangular CSR matrix Acsr to 'full' CSR matrix Acsr_full
Acsr_full = Acsr + Acsr.T - scipy.sparse.diags(Acsr.diagonal())

print()
print("Sparse 'full' CSR matrix:")
print("values:  ", Acsr_full.data)
# Indices are 0 based
print("index:   ", Acsr_full.indices)
print("pointer: ", Acsr_full.indptr)

# initial guess for solution x
x = np.zeros(8, dtype=np.float32)
# right hand side
b = np.ones(8, dtype=np.float32)
y = np.zeros(8, dtype=np.float32)
print("dtype x: ", x.dtype)
print("dtype b: ", b.dtype)
print("dtype A: ", Acsr_full.data.dtype)
info = 1    # make Paralution more verbose
abstol = 1e-6  # convergence tolerance
reltol = 5e-4
divtol = 1e5
max_iter = 11111 # maximum number of iterations

paralution_wrapper.solution(Acsr_full.data, Acsr_full.indices, Acsr_full.indptr,
                            x, b, info, abstol, reltol, divtol, max_iter)

# check solution x with original dense matrix A first
# convert upper triangular matrix AA to 'full' matrix
y = (A + A.T - np.eye(A.shape[0]) * A.diagonal()).dot(x)

assert (np.allclose(b, y, rtol=reltol))

# check solution  with sparse matrix Acsr_full
y = Acsr_full.dot(x)
assert (np.allclose(b, y, rtol=reltol))
print("Solution float x:")
print(x)
print()
print("A * x:")
print(y)
print("b:")
print(b)
