#pylint: disable-all
import numpy as np

""" Solve Ax=B with the Jacobi method """
def jacobi(A, B, n, tol=1e-10):
# Initializes x with zeroes with same shape and type as B
    x = np.zeros_like(B)
    for it_count in range(n):
        x_new = np.zeros_like(x)
        for i in range(A.shape[0]):
            s1 = np.dot(A[i, :i], x[:i])
            s2 = np.dot(A[i, i + 1:], x[i + 1:])
            x_new[i] = (B[i] - s1 - s2) / A[i, i]
        if np.allclose(x, x_new, tol):
            break
        x = x_new
    return x

""" Solve Ax=B with the Gauss-Seidel method """

def gauss(A, B, n, tol=1e-10):
    L = np.tril(A) # Returns the lower triangular matrix of A
    U = A - L # Decompose A = L + U
    L_inv = np.linalg.inv(L)
    x = np.zeros_like(B)
    for i in range(n):
        Ux = np.dot(U, x)
        x_new = np.dot(L_inv, B - Ux)
        if np.allclose(x, x_new, tol):
            break
        x = x_new
    return x
if __name__ == "__main__":
    pass

