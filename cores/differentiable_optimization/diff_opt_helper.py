import torch
import numpy as np

def get_gradient_pytorch(dual_var, F1_dp, F2_dp, F1_dx, F2_dx, F1_dpdp, F2_dpdp, F1_dpdx, F2_dpdx):
    """
    Compute the gradient of the primal and dual variable w.r.t. x in a batch manner.
    dual_var (torch.Tensor, shape (batch_size, )): dual variables
    F1_dp (torch.Tensor, shape (batch_size, dim(p))): dF1/dp
    F2_dp (torch.Tensor, shape (batch_size, dim(p))): dF2/dp
    F1_dx (torch.Tensor, shape (batch_size, dim(x))): dF1/dx
    F2_dx (torch.Tensor, shape (batch_size, dim(x))): dF2/dx
    F1_dpdp (torch.Tensor, shape (batch_size, dim(p), dim(p))): d2F1/dpdp
    F2_dpdp (torch.Tensor, shape (batch_size, dim(p), dim(p))): d2F2/dpdp
    F1_dpdx (torch.Tensor, shape (batch_size, dim(p), dim(x))): d2F1/dpdx
    F2_dpdx (torch.Tensor, shape (batch_size, dim(p), dim(x))): d2F2/dpdx
    """
    # Construct the linear system
    batch_size = dual_var.shape[0]
    b1 = - F1_dpdx - torch.matmul(dual_var.unsqueeze(-1).unsqueeze(-1), F2_dpdx) # shape (batch_size, dim(p), dim(x))
    b2 = - F2_dx.unsqueeze(-2) # shape (batch_size, 1, dim(x))
    b = torch.cat([b1, b2], dim=1) # shape (batch_size, dim(p) + 1, dim(x))

    A11 = F1_dpdp + torch.matmul(dual_var.unsqueeze(-1).unsqueeze(-1), F2_dpdp) # shape (batch_size, dim(p), dim(p))
    A12 = F2_dp.unsqueeze(-1) # shape (batch_size, dim(p), 1)
    A21 = F2_dp.unsqueeze(-2) # shape (batch_size, 1, dim(p))
    A22 = torch.zeros((batch_size, 1, 1), dtype=F1_dpdp.dtype) # shape (batch_size, 1, 1)
    A1 = torch.cat([A11, A12], dim=2) # shape (batch_size, dim(p), dim(p) + 1)
    A2 = torch.cat([A21, A22], dim=2) # shape (batch_size, 1, dim(p) + 1)
    A = torch.cat([A1, A2], dim=1) # shape (batch_size, dim(p) + 1, dim(p) + 1)

    # Solve the linear system
    grad = torch.linalg.solve(A, b) # shape (batch_size, dim(p) + 1, dim(x))
    p_dx = grad[:, :-1, :] # shape (batch_size, dim(p), dim(x))
    # dual_dx = grad[:, -1, :] # shape (batch_size, dim(x)), not used

    alpha_dx = F1_dx.unsqueeze(-2) + torch.matmul(F1_dp.unsqueeze(-2), p_dx) # shape (batch_size, 1, dim(x))
    alpha_dx = alpha_dx.squeeze(1) # shape (batch_size, dim(x))
    return alpha_dx
