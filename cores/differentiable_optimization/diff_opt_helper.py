import torch
import numpy as np

def get_dual_variable_pytorch(F1_dp, F2_dp):
    """
    Compute the dual variable in a batch manner.
    F1_dp (torch.Tensor, shape (batch_size, dim(p))): dF1/dp
    F2_dp (torch.Tensor, shape (batch_size, dim(p))): dF2/dp

    Returns:
    dual_var (torch.Tensor, shape (batch_size, )): dual variables
    """
    return torch.linalg.norm(F1_dp, dim=1) / torch.linalg.norm(F2_dp, dim=1)

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

    Returns:
    alpha_dx (torch.Tensor, shape (batch_size, dim(x)): d(alpha)/dx
    """
    # Construct the linear system
    batch_size = dual_var.shape[0]
    device = F1_dpdx.device
    dtype = F1_dpdx.dtype
    b1 = - F1_dpdx - dual_var.view(-1, 1, 1) * F2_dpdx # shape (batch_size, dim(p), dim(x))
    b2 = - F2_dx.unsqueeze(-2) # shape (batch_size, 1, dim(x))
    b = torch.cat([b1, b2], dim=1) # shape (batch_size, dim(p) + 1, dim(x))

    A11 = F1_dpdp + dual_var.view(-1, 1, 1) * F2_dpdp # shape (batch_size, dim(p), dim(p))
    A12 = F2_dp.unsqueeze(-1) # shape (batch_size, dim(p), 1)
    A21 = F2_dp.unsqueeze(-2) # shape (batch_size, 1, dim(p))
    A22 = torch.zeros((batch_size, 1, 1), dtype=dtype, device=device) # shape (batch_size, 1, 1)
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

def get_gradient_and_hessian_pytorch(dual_var, F1_dp, F2_dp, F1_dx, F2_dx, F1_dpdp, F2_dpdp, F1_dpdx, F2_dpdx,
                                     F1_dxdx, F2_dxdx, F1_dpdpdp, F2_dpdpdp, F1_dpdpdx, F2_dpdpdx, F1_dpdxdx, F2_dpdxdx):
    """
    Compute the gradient and Hessian of the primal and dual variable w.r.t. x in a batch manner.
    dual_var (torch.Tensor, shape (batch_size, )): dual variables
    F1_dp (torch.Tensor, shape (batch_size, dim(p))): dF1/dp
    F2_dp (torch.Tensor, shape (batch_size, dim(p))): dF2/dp
    F1_dx (torch.Tensor, shape (batch_size, dim(x))): dF1/dx
    F2_dx (torch.Tensor, shape (batch_size, dim(x))): dF2/dx
    F1_dpdp (torch.Tensor, shape (batch_size, dim(p), dim(p))): d2F1/dpdp
    F2_dpdp (torch.Tensor, shape (batch_size, dim(p), dim(p))): d2F2/dpdp
    F1_dpdx (torch.Tensor, shape (batch_size, dim(p), dim(x))): d2F1/dpdx
    F2_dpdx (torch.Tensor, shape (batch_size, dim(p), dim(x))): d2F2/dpdx
    F1_dxdx (torch.Tensor, shape (batch_size, dim(x), dim(x))): d2F1/dxdx
    F2_dxdx (torch.Tensor, shape (batch_size, dim(x), dim(x))): d2F2/dxdx
    F1_dpdpdp (torch.Tensor, shape (batch_size, dim(p), dim(p), dim(p))): d3F1/dpdpdp
    F2_dpdpdp (torch.Tensor, shape (batch_size, dim(p), dim(p), dim(p))): d3F2/dpdpdp
    F1_dpdpdx (torch.Tensor, shape (batch_size, dim(p), dim(p), dim(x))): d3F1/dpdpdx
    F2_dpdpdx (torch.Tensor, shape (batch_size, dim(p), dim(p), dim(x))): d3F2/dpdpdx
    F1_dpdxdx (torch.Tensor, shape (batch_size, dim(p), dim(x), dim(x))): d3F1/dpdxdx
    F2_dpdxdx (torch.Tensor, shape (batch_size, dim(p), dim(x), dim(x))): d3F2/dpdxdx

    Returns:
    alpha_dx (torch.Tensor, shape (batch_size, dim(x)): d(alpha)/dx
    alpha_dxdx (torch.Tensor, shape (batch_size, dim(x), dim(x)): d2(alpha)/dxdx
    """

    # Construct the linear system
    batch_size, dim_p, dim_x = F1_dpdx.shape
    device = F1_dpdx.device
    dtype = F1_dpdx.dtype
    b1 = - F1_dpdx - dual_var.view(-1, 1, 1) * F2_dpdx # shape (batch_size, dim(p), dim(x))
    b2 = - F2_dx.unsqueeze(-2) # shape (batch_size, 1, dim(x))
    b = torch.cat([b1, b2], dim=1) # shape (batch_size, dim(p) + 1, dim(x))

    A11 = F1_dpdp + dual_var.view(-1, 1, 1) * F2_dpdp # shape (batch_size, dim(p), dim(p))
    A12 = F2_dp.unsqueeze(-1) # shape (batch_size, dim(p), 1)
    A21 = F2_dp.unsqueeze(-2) # shape (batch_size, 1, dim(p))
    A22 = torch.zeros((batch_size, 1, 1), dtype=dtype, device=device) # shape (batch_size, 1, 1)
    A1 = torch.cat([A11, A12], dim=2) # shape (batch_size, dim(p), dim(p) + 1)
    A2 = torch.cat([A21, A22], dim=2) # shape (batch_size, 1, dim(p) + 1)
    A = torch.cat([A1, A2], dim=1) # shape (batch_size, dim(p) + 1, dim(p) + 1)

    # Solve the linear system for the gradient
    grad = torch.linalg.solve(A, b) # shape (batch_size, dim(p) + 1, dim(x))
    p_dx = grad[:, :-1, :] # shape (batch_size, dim(p), dim(x))
    dual_dx = grad[:, -1, :] # shape (batch_size, dim(x))

    alpha_dx = F1_dx.unsqueeze(-2) + torch.matmul(F1_dp.unsqueeze(-2), p_dx) # shape (batch_size, 1, dim(x))
    alpha_dx = alpha_dx.squeeze(1) # shape (batch_size, dim(x))

    # Construct the linear system for the Hessian
    F1_dpdxdp = F1_dpdpdx.transpose(-1, -2) # shape (batch_size, dim(p), dim(x), dim(p))
    F2_dpdxdp = F2_dpdpdx.transpose(-1, -2) # shape (batch_size, dim(p), dim(x), dim(p))
    b1_dx_1 =  - F1_dpdxdx - torch.einsum('bijk,bkl->bijl', F1_dpdxdp, p_dx) # shape (batch_size, dim(p), dim(x), dim(x))
    tmp = F2_dpdxdx + torch.einsum('bijk,bkl->bijl', F2_dpdxdp, p_dx) # shape (batch_size, dim(p), dim(x), dim(x))
    b1_dx_2 = - torch.einsum('bij,bk->bijk', F1_dpdx, dual_dx) - torch.einsum('b,bijk->bijk', dual_var, tmp) # shape (batch_size, dim(p), dim(x), dim(x))
    b1_dx = b1_dx_1 + b1_dx_2 # shape (batch_size, dim(p), dim(x), dim(x))
    F2_dx_dp = F2_dpdx.transpose(-1, -2) # shape (batch_size, dim(x), dim(p))
    b2_dx = - F2_dxdx.unsqueeze(-3) - torch.matmul(F2_dx_dp, p_dx).unsqueeze(-3) # shape (batch_size, 1, dim(x), dim(x))
    b_dx = torch.cat([b1_dx, b2_dx], dim=1) # shape (batch_size, dim(p) + 1, dim(x), dim(x))

    A11_dx_1 = F1_dpdpdx + torch.einsum('bijk,bkl->bijl', F1_dpdpdp, p_dx) # shape (batch_size, dim(p), dim(p), dim(x))
    tmp = F2_dpdpdx + torch.einsum('bijk,bkl->bijl', F2_dpdpdp, p_dx) # shape (batch_size, dim(p), dim(p), dim(x))
    A11_dx_2 = torch.einsum('bij,bk->bijk', F2_dpdp, dual_dx) + torch.einsum('b,bijk->bijk', dual_var, tmp) # shape (batch_size, dim(p), dim(p), dim(x))
    A11_dx = A11_dx_1 + A11_dx_2 # shape (batch_size, dim(p), dim(p), dim(x))
    A21_dx = F2_dpdx.unsqueeze(-3) + torch.matmul(F2_dpdp, p_dx).unsqueeze(-3) # shape (batch_size, 1, dim(p), dim(x))
    A12_dx = A21_dx.transpose(-3, -2) # shape (batch_size, dim(p), 1, dim(x))
    A22_dx = torch.zeros((batch_size, 1, 1, dim_x), dtype=dtype, device=device) # shape (batch_size, 1, 1, dim(x))
    A1_dx = torch.cat([A11_dx, A12_dx], dim=2) # shape (batch_size, dim(p), dim(p) + 1, dim(x))
    A2_dx = torch.cat([A21_dx, A22_dx], dim=2) # shape (batch_size, 1, dim(p) + 1, dim(x))
    A_dx = torch.cat([A1_dx, A2_dx], dim=1) # shape (batch_size, dim(p) + 1, dim(p) + 1, dim(x))

    # Solve the linear system for the Hessian
    hessian = torch.zeros((batch_size, dim_p+1, dim_x, dim_x), dtype=dtype, device=device)
    for i in range(dim_x):
        tmp = torch.matmul(A_dx[:, :, :, i], grad) # shape (batch_size, dim(p) + 1, dim(x))
        tmp = torch.linalg.solve(A, tmp) # shape (batch_size, dim(p) + 1, dim(x))
        hessian[:, :, :, i] = - tmp + torch.linalg.solve(A, b_dx[:, :, :, i]) # shape (batch_size, dim(p) + 1, dim(x))

    F1_dxdp = F1_dpdx.transpose(-1, -2) # shape (batch_size, dim(x), dim(p))
    alpha_dxdx = F1_dxdx + torch.matmul(F1_dxdp, p_dx) + torch.matmul(torch.matmul(p_dx.transpose(-1, -2), F1_dpdp), p_dx) # shape (batch_size, dim(x), dim(x))
    p_dxdx = hessian[:, :-1, :, :] # shape (batch_size, dim(p), dim(x), dim(x))
    alpha_dxdx = alpha_dxdx + torch.einsum('bi,bijk->bjk', F1_dp, p_dxdx) # shape (batch_size, dim(x), dim(x))

    return alpha_dx, alpha_dxdx
