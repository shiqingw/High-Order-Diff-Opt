import numpy as np
import scipy as sp
import torch

def rimon_method_numpy(A, a, B, b):
    """
    Rimon's method for solving the smallest scaling between two ellipsoids.
    The method is based on the paper "Obstacle Collision Detection Using Best Ellipsoid Fit".
    Inputs:
    A (np.array, shape (n,n)): The matrix of the first ellipsoid.
    a (np.array, shape (n,)): The center of the first ellipsoid.
    B (np.array, shape (n,n)): The matrix of the second ellipsoid.
    b (np.array, shape (n,)): The center of the second ellipsoid.
    
    Outputs:
    x_rimon (np.array, shape (n,)): The point on the second ellipsoid where the ellipsoidal level
                surfaces surrounding the first ellipsoid first touch the second ellipsoid.
    """

    nv = A.shape[0]
    A_sqrt = np.linalg.cholesky(A) # A = L @ L.T 
    C = sp.linalg.solve_triangular(A_sqrt, B.T, lower=True) # C = inv(A_sqrt) @ B.T
    C = sp.linalg.solve_triangular(A_sqrt, C.T, lower=True) # C = inv(A_sqrt) @ B @ inv(A_sqrt).T
    c = A_sqrt.T @ (b-a)
    C_sqrt = np.linalg.cholesky(C)
    c_tilde = sp.linalg.solve_triangular(C_sqrt, c, lower=True)
    c_tilde = c_tilde[:,np.newaxis]
    C_sqrt_inv = sp.linalg.lapack.dtrtri(C_sqrt, lower=True)[0]
    C_tilde = C_sqrt_inv @ C_sqrt_inv.T # C_tilde = inv(C_sqrt) @ inv(C_sqrt).T
    M = np.block([[C_tilde, -np.eye(nv)],
                [-c_tilde @ c_tilde.T, C_tilde]])
    lambda_min = min(np.real(sp.linalg.eigvals(M)))
    x_rimon = np.linalg.solve(lambda_min*C - np.eye(nv), C @ c)
    x_rimon = a + lambda_min * sp.linalg.solve_triangular(A_sqrt.T, x_rimon, lower=False)
    return x_rimon


def rimon_method_pytorch(A, a, B, b):
    """
    Rimon's method for solving the smallest scaling between two ellipsoids using PyTorch.
    This function is designed to work with batch data.
    
    Inputs:
    A (torch.Tensor, shape (batch_size, n, n)): Batch of matrices for the first ellipsoid.
    a (torch.Tensor, shape (batch_size, n)): Batch of centers for the first ellipsoid.
    B (torch.Tensor, shape (batch_size, n, n)): Batch of matrices for the second ellipsoid.
    b (torch.Tensor, shape (batch_size, n)): Batch of centers for the second ellipsoid.
    
    Outputs:
    x_rimon (torch.Tensor, shape (batch_size, n)): Batch of points on the second ellipsoid where the
                ellipsoidal level surfaces surrounding the first ellipsoid first touch the second ellipsoid.
    """
    batch_size, nv, _ = A.shape

    # Compute the Cholesky decomposition of A for each batch
    A_sqrt = torch.linalg.cholesky(A, upper=False) # A = L @ L.T 

    # Solve for C using the Cholesky factor of A
    C = torch.linalg.solve_triangular(A_sqrt, B, left=True, upper=False) # C = inv(A_sqrt) @ B
    C = torch.linalg.solve_triangular(A_sqrt.transpose(-2, -1), C, left=False, upper=True) # C = inv(A_sqrt) @ B @ inv(A_sqrt).T

    # Compute the vector c for each batch
    c = torch.matmul(A_sqrt.transpose(-2, -1), (b - a).unsqueeze(-1)) # shape (batch_size, nv, 1)

    # Compute the Cholesky decomposition of C for each batch
    C_sqrt = torch.linalg.cholesky(C, upper=False) # A = L @ L.T 

    # Solve for c_tilde using the Cholesky factor of C
    c_tilde = torch.linalg.solve_triangular(C_sqrt, c, left=True, upper=False) # shape (batch_size, nv, 1)

    # Compute the inverse of C_sqrt
    C_sqrt_inv = torch.linalg.inv(C_sqrt)
    C_tilde = torch.matmul(C_sqrt_inv, C_sqrt_inv.transpose(-2, -1)) # C_tilde = inv(C_sqrt) @ inv(C_sqrt).T

    # Assemble the matrix M for each batch
    M = torch.cat([
        torch.cat([C_tilde, -torch.eye(nv).expand(batch_size, nv, nv)], dim=2),
        torch.cat([-torch.matmul(c_tilde, c_tilde.transpose(-2, -1)), C_tilde], dim=2)
    ], dim=1)

    # Compute the minimum eigenvalue of M for each batch
    lambda_min = torch.min(torch.real(torch.linalg.eigvals(M)), dim=1).values # shape (batch_size,)

    # Solve for x_rimon using the computed lambda_min for each batch
    I = torch.eye(nv).expand(batch_size, nv, nv)
    x_rimon = torch.linalg.solve(lambda_min.view(-1,1,1)*C - I, torch.matmul(C, c))
    x_rimon = torch.linalg.solve_triangular(A_sqrt.transpose(-2, -1), x_rimon, left=True, upper=True)
    x_rimon = a + lambda_min.view(-1,1) * x_rimon.squeeze(-1)

    return x_rimon