import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import numpy as np
import torch
import diffOptEllipsoidCpp as DOE
from cores.differentiable_optimization.diff_opt_helper import get_dual_variable_pytorch, get_gradient_pytorch, get_gradient_and_hessian_pytorch
from cores.configuration.configuration import Configuration
config = Configuration()

dim_x = 7
dim_p = 3

F1_dp = np.random.rand(dim_p).astype(config.np_dtype)
F2_dp = np.random.rand(dim_p).astype(config.np_dtype)
F1_dx = np.random.rand(dim_x).astype(config.np_dtype)
F2_dx = np.random.rand(dim_x).astype(config.np_dtype)
F1_dpdp = np.random.rand(dim_p, dim_p).astype(config.np_dtype)
F2_dpdp = np.random.rand(dim_p, dim_p).astype(config.np_dtype)
F1_dpdx = np.random.rand(dim_p, dim_x).astype(config.np_dtype)
F2_dpdx = np.random.rand(dim_p, dim_x).astype(config.np_dtype)
F1_dxdx = np.random.rand(dim_x, dim_x).astype(config.np_dtype)
F2_dxdx = np.random.rand(dim_x, dim_x).astype(config.np_dtype)
F1_dpdpdp = np.random.rand(dim_p, dim_p, dim_p).astype(config.np_dtype)
F2_dpdpdp = np.random.rand(dim_p, dim_p, dim_p).astype(config.np_dtype)
F1_dpdpdx = np.random.rand(dim_p, dim_p, dim_x).astype(config.np_dtype)
F2_dpdpdx = np.random.rand(dim_p, dim_p, dim_x).astype(config.np_dtype)
F1_dpdxdx = np.random.rand(dim_p, dim_x, dim_x).astype(config.np_dtype)
F2_dpdxdx = np.random.rand(dim_p, dim_x, dim_x).astype(config.np_dtype)

F1_dp_torch = torch.tensor(F1_dp, dtype=config.torch_dtype).unsqueeze(0)
F2_dp_torch = torch.tensor(F2_dp, dtype=config.torch_dtype).unsqueeze(0)
F1_dx_torch = torch.tensor(F1_dx, dtype=config.torch_dtype).unsqueeze(0)
F2_dx_torch = torch.tensor(F2_dx, dtype=config.torch_dtype).unsqueeze(0)
F1_dpdp_torch = torch.tensor(F1_dpdp, dtype=config.torch_dtype).unsqueeze(0)
F2_dpdp_torch = torch.tensor(F2_dpdp, dtype=config.torch_dtype).unsqueeze(0)
F1_dpdx_torch = torch.tensor(F1_dpdx, dtype=config.torch_dtype).unsqueeze(0)
F2_dpdx_torch = torch.tensor(F2_dpdx, dtype=config.torch_dtype).unsqueeze(0)
F1_dxdx_torch = torch.tensor(F1_dxdx, dtype=config.torch_dtype).unsqueeze(0)
F2_dxdx_torch = torch.tensor(F2_dxdx, dtype=config.torch_dtype).unsqueeze(0)
F1_dpdpdp_torch = torch.tensor(F1_dpdpdp, dtype=config.torch_dtype).unsqueeze(0)
F2_dpdpdp_torch = torch.tensor(F2_dpdpdp, dtype=config.torch_dtype).unsqueeze(0)
F1_dpdpdx_torch = torch.tensor(F1_dpdpdx, dtype=config.torch_dtype).unsqueeze(0)
F2_dpdpdx_torch = torch.tensor(F2_dpdpdx, dtype=config.torch_dtype).unsqueeze(0)
F1_dpdxdx_torch = torch.tensor(F1_dpdxdx, dtype=config.torch_dtype).unsqueeze(0)
F2_dpdxdx_torch = torch.tensor(F2_dpdxdx, dtype=config.torch_dtype).unsqueeze(0)

dual_var_torch = get_dual_variable_pytorch(F1_dp_torch, F2_dp_torch)
grad_torch = get_gradient_pytorch(dual_var_torch, F1_dp_torch, F2_dp_torch, F1_dx_torch,\
                            F2_dx_torch, F1_dpdp_torch, F2_dpdp_torch, F1_dpdx_torch, F2_dpdx_torch)

dual_var = DOE.getDualVariable(F1_dp, F2_dp)
grad = DOE.getGradient(dual_var, F1_dp, F2_dp, F1_dx, F2_dx, F1_dpdp, F2_dpdp, F1_dpdx, F2_dpdx)

# print(grad_torch.squeeze(0) - grad)

grad_torch, hess_torch = get_gradient_and_hessian_pytorch(dual_var_torch, F1_dp_torch, F2_dp_torch, F1_dx_torch, F2_dx_torch,\
                                        F1_dpdp_torch, F2_dpdp_torch, F1_dpdx_torch, F2_dpdx_torch,\
                                        F1_dxdx_torch, F2_dxdx_torch, F1_dpdpdp_torch, F2_dpdpdp_torch,\
                                        F1_dpdpdx_torch, F2_dpdpdx_torch, F1_dpdxdx_torch, F2_dpdxdx_torch)

grad, hess = DOE.getGradientAndHessian(dual_var, F1_dp, F2_dp, F1_dx, F2_dx, F1_dpdp, F2_dpdp, F1_dpdx, F2_dpdx,\
                                F1_dxdx, F2_dxdx, F1_dpdpdp, F2_dpdpdp, F1_dpdpdx, F2_dpdpdx, F1_dpdxdx, F2_dpdxdx)

print(grad_torch.squeeze(0) - grad)
print(hess_torch.squeeze(0) - hess)

