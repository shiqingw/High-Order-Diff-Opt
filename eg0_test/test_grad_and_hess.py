import time
import numpy as np
import timeit
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from cores.rimon_method_python.rimon_method import rimon_method_pytorch
import torch
from cores.configuration.configuration import Configuration
config = Configuration()
from cores.differentiable_optimization.diff_opt_helper import get_gradient_pytorch, get_dual_variable_pytorch, get_gradient_and_hessian_pytorch
from cores.scaling_functions.quadratics import ellipsoid_value, ellipsoid_gradient, ellipsoid_hessian

# Define parameters
print("==> Define parameters")
a = np.array([0.1, 0.2, 0.3], dtype=config.np_dtype)
A = np.array([[1.1, 0.1, 0.1],
              [0.1, 2, 0.1],
              [0.1, 0.1, 3]], dtype=config.np_dtype)

b = np.array([3.1, 3.2, 3.3], dtype=config.np_dtype)
B = np.array([[1, 0.1, 0.1],
              [0.1, 2, 0.1],
              [0.1, 0.1, 1]], dtype=config.np_dtype)

# Rimons method
print("==> Rimons method")
A_torch_original = torch.tensor(A, dtype=config.torch_dtype, requires_grad=True)
A_torch = A_torch_original.unsqueeze(0)
a_torch_original = torch.tensor(a, dtype=config.torch_dtype, requires_grad=True)
a_torch = a_torch_original.unsqueeze(0)
B_torch_original = torch.tensor(B, dtype=config.torch_dtype, requires_grad=True)
B_torch = B_torch_original.unsqueeze(0)
b_torch_original = torch.tensor(b, dtype=config.torch_dtype, requires_grad=True)
b_torch = b_torch_original.unsqueeze(0)

number = 1000
print("pytorch avg time: ", 
      timeit.timeit('rimon_method_pytorch(A_torch, a_torch, B_torch, b_torch)', globals=globals(), number=number)/number)
p_rimon = rimon_method_pytorch(A_torch, a_torch, B_torch, b_torch)
print("rimon p: ", p_rimon)
alpha = ellipsoid_value(p_rimon, A_torch, a_torch)
start = time.time()
alpha.backward()
end = time.time()
print("pytorch backward time: ", end-start)
print("pytorch gradients:")
print(A_torch_original.grad)
print(a_torch_original.grad)

start = time.time()
F1_dp = ellipsoid_gradient(p_rimon, A_torch, a_torch)
F2_dp = ellipsoid_gradient(p_rimon, B_torch, b_torch)
F1_dpdp = ellipsoid_hessian(A_torch)
F2_dpdp = ellipsoid_hessian(B_torch)
tmp1 = torch.matmul((p_rimon - a_torch).unsqueeze(-1), (p_rimon - a_torch).unsqueeze(-2)).view(1,-1) 
tmp2 = 2*torch.matmul(A_torch, (a_torch - p_rimon).unsqueeze(-1)).squeeze(-1)
F1_dx = torch.cat((tmp1, tmp2), dim=1)
F2_dx = torch.zeros_like(F1_dx)
tmp1 = torch.zeros((1,3,9), dtype=config.torch_dtype)
tmp1[0,0,0:3] = 2*(p_rimon - a_torch).squeeze()
tmp1[0,1,3:6] = 2*(p_rimon - a_torch).squeeze()
tmp1[0,2,6:9] = 2*(p_rimon - a_torch).squeeze()
tmp2 = -2*A_torch
F1_dpdx = torch.cat((tmp1, tmp2), dim=2)
F2_dpdx = torch.zeros_like(F1_dpdx)
dual_vars = get_dual_variable_pytorch(F1_dp, F2_dp)
alpha_dx = get_gradient_pytorch(dual_vars, F1_dp, F2_dp, F1_dx, F2_dx, F1_dpdp, F2_dpdp, F1_dpdx, F2_dpdx)
end = time.time()
print("pytorch get_gradient time: ", end-start)
# print(alpha_dx.shape)
print(alpha_dx)

dim_x = 12
dim_p = 3
F1_dxdx = torch.zeros((1, dim_x, dim_x), dtype=config.torch_dtype)
F1_dxdx[0,0:3,9] = 2*(a_torch - p_rimon).squeeze()
F1_dxdx[0,3:6,10] = 2*(a_torch - p_rimon).squeeze()
F1_dxdx[0,6:9,11] = 2*(a_torch - p_rimon).squeeze()
F1_dxdx[0,9,0:3] = 2*(a_torch - p_rimon).squeeze()
F1_dxdx[0,10,3:6] = 2*(a_torch - p_rimon).squeeze()
F1_dxdx[0,11,6:9] = 2*(a_torch - p_rimon).squeeze()
F1_dxdx[0,9:12,9:12] = 2*A_torch.squeeze()
F2_dxdx = torch.zeros_like(F1_dxdx)
F1_dpdpdp = torch.zeros((1,dim_p,dim_p,dim_p), dtype=config.torch_dtype)
F2_dpdpdp = torch.zeros_like(F1_dpdpdp)
F1_dpdpdx = torch.zeros((1,dim_p,dim_p,dim_x), dtype=config.torch_dtype)
for i in range(9):
      F1_dpdpdx[0,i//3,i%3,i] += 1.0
      F1_dpdpdx[0,i%3,i//3,i] += 1.0
F2_dpdpdx = torch.zeros_like(F1_dpdpdx)
F1_dxdxdp = torch.zeros((1,dim_x,dim_x,dim_p), dtype=config.torch_dtype)
for i in range(3):
      F1_dxdxdp[0,9,0+i,i] += -2.0
      F1_dxdxdp[0,10,3+i,i] += -2.0
      F1_dxdxdp[0,11,6+i,i] += -2.0
      F1_dxdxdp[0,0+i,9,i] += -2.0
      F1_dxdxdp[0,3+i,10,i] += -2.0
      F1_dxdxdp[0,6+i,11,i] += -2.0
F1_dpdxdx = F1_dxdxdp.transpose(-1, -3)
F2_dpdxdx = torch.zeros_like(F1_dpdxdx)
start = time.time()
alpha_dx, alpha_dxdx = get_gradient_and_hessian_pytorch(dual_vars, F1_dp, F2_dp, F1_dx, F2_dx, F1_dpdp, F2_dpdp, F1_dpdx, F2_dpdx,
                                     F1_dxdx, F2_dxdx, F1_dpdpdp, F2_dpdpdp, F1_dpdpdx, F2_dpdpdx, F1_dpdxdx, F2_dpdxdx)
end = time.time()
print("pytorch get_gradient_and_hessian time: ", end-start)

perturb_a = 0.01*torch.rand(3, dtype=config.torch_dtype, requires_grad=True)
perturb_A = 0.01*torch.rand((3,3), dtype=config.torch_dtype, requires_grad=True)
perturb_A = torch.matmul(perturb_A, perturb_A.transpose(-1, -2))
A = A_torch_original + perturb_A
a = a_torch_original + perturb_a
A_torch = A.unsqueeze(0)
a_torch = a.unsqueeze(0)
p_rimon = rimon_method_pytorch(A_torch, a_torch, B_torch, b_torch)
F1_dp = ellipsoid_gradient(p_rimon, A_torch, a_torch)
F2_dp = ellipsoid_gradient(p_rimon, B_torch, b_torch)
F1_dpdp = ellipsoid_hessian(A_torch)
F2_dpdp = ellipsoid_hessian(B_torch)
tmp1 = torch.matmul((p_rimon - a_torch).unsqueeze(-1), (p_rimon - a_torch).unsqueeze(-2)).view(1,-1) 
tmp2 = 2*torch.matmul(A_torch, (a_torch - p_rimon).unsqueeze(-1)).squeeze(-1)
F1_dx = torch.cat((tmp1, tmp2), dim=1)
F2_dx = torch.zeros_like(F1_dx)
tmp1 = torch.zeros((1,3,9), dtype=config.torch_dtype)
tmp1[0,0,0:3] = 2*(p_rimon - a_torch).squeeze()
tmp1[0,1,3:6] = 2*(p_rimon - a_torch).squeeze()
tmp1[0,2,6:9] = 2*(p_rimon - a_torch).squeeze()
tmp2 = -2*A_torch
F1_dpdx = torch.cat((tmp1, tmp2), dim=2)
F2_dpdx = torch.zeros_like(F1_dpdx)
dual_vars = get_dual_variable_pytorch(F1_dp, F2_dp)
alpha_dx_new = get_gradient_pytorch(dual_vars, F1_dp, F2_dp, F1_dx, F2_dx, F1_dpdp, F2_dpdp, F1_dpdx, F2_dpdx)

perturb = torch.cat([perturb_A.view(-1), perturb_a.view(-1)]).view(1,12,1)
alpha_dx_anticipated = alpha_dx + torch.matmul(alpha_dxdx, perturb).squeeze(-1)
# print(torch.all(alpha_dxdx.transpose(-1, -2) == alpha_dxdx))
print(alpha_dxdx - alpha_dxdx.transpose(-1, -2))
print(perturb)
print(alpha_dx_anticipated)
print(alpha_dx_new)