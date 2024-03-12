import torch

def ellipsoid_value(p, A, a):
    """
    Compute the ellipsoid function.
    p (torch.Tensor, shape (batch_size, dim(x))): input
    A (torch.Tensor, shape (batch_size, dim(x), dim(x))): real symmetric quadratic coefficient matrix
    a (torch.Tensor, shape (batch_size, dim(x))): center of the ellipsoid

    Returns:
    (torch.Tensor, shape (batch_size,)): ellipsoid function value
    """
    tmp = (p - a).unsqueeze(-1) # shape (batch_size, dim(x), 1)
    return torch.matmul(torch.matmul(tmp.transpose(-1,-2), A), tmp).squeeze() # seems faster than torch.einsum

def ellipsoid_gradient(p, A, a):
    """
    Compute the gradient of the ellipsoid function.
    p (torch.Tensor, shape (batch_size, dim(x))): input
    A (torch.Tensor, shape (batch_size, dim(x), dim(x))): real symmetric quadratic coefficient matrix
    a (torch.Tensor, shape (batch_size, dim(x))): center of the ellipsoid

    Returns:
    (torch.Tensor, shape (batch_size, dim(x))): gradient of the ellipsoid function
    """
    tmp = (p - a).unsqueeze(-1) # shape (batch_size, dim(x), 1)
    return 2 * torch.matmul(A, tmp).squeeze(-1)

def ellipsoid_hessian(A):
    """
    Compute the Hessian of the ellipsoid function.
    A (torch.Tensor, shape (batch_size, dim(x), dim(x))): real symmetric quadratic coefficient matrix

    Returns:
    (torch.Tensor, shape (batch_size, dim(x), dim(x))): Hessian of the ellipsoid function
    """

    return 2 * A