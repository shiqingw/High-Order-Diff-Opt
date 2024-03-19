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
from cores.scaling_functions.ellipsoid import Ellipsoid
torch.manual_seed(0)
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

SF = Ellipsoid()
number = 1000
print("rimon_method avg time: ", 
      timeit.timeit('rimon_method_pytorch(A_torch, a_torch, B_torch, b_torch)', globals=globals(), number=number)/number)
p_rimon = rimon_method_pytorch(A_torch, a_torch, B_torch, b_torch)
print("rimon p: ", p_rimon)
alpha = SF.F(p_rimon, A_torch, a_torch)
start = time.time()
alpha.backward()
end = time.time()
print("pytorch backward time: ", end-start)
print("pytorch gradients:")
print(A_torch_original.grad)
print(a_torch_original.grad)

F1_dp = SF.F_dp(p_rimon, A_torch, a_torch)
F2_dp = SF.F_dp(p_rimon, B_torch, b_torch)
F1_dpdp = SF.F_dpdp(p_rimon, A_torch, a_torch)
F2_dpdp = SF.F_dpdp(p_rimon, B_torch, b_torch)
F1_dx = SF.F_dx(p_rimon, A_torch, a_torch)
F2_dx = torch.zeros_like(F1_dx)
F1_dpdx = SF.F_dpdx(p_rimon, A_torch, a_torch)
F2_dpdx = torch.zeros_like(F1_dpdx)
dual_vars = get_dual_variable_pytorch(F1_dp, F2_dp)
print("get_gradient_pytorch time: ", 
      timeit.timeit('get_gradient_pytorch(dual_vars, F1_dp, F2_dp, F1_dx, F2_dx, F1_dpdp, F2_dpdp, F1_dpdx, F2_dpdx)', globals=globals(), number=number)/number)
alpha_dx = get_gradient_pytorch(dual_vars, F1_dp, F2_dp, F1_dx, F2_dx, F1_dpdp, F2_dpdp, F1_dpdx, F2_dpdx)
print("alpha_dx:")
print(alpha_dx)

F1_dxdx = SF.F_dxdx(p_rimon, A_torch, a_torch)
F2_dxdx = torch.zeros_like(F1_dxdx)
F1_dpdpdp = SF.F_dpdpdp(p_rimon, A_torch, a_torch)
F2_dpdpdp = SF.F_dpdpdp(p_rimon, B_torch, b_torch)
F1_dpdpdx = SF.F_dpdpdx(p_rimon, A_torch, a_torch)
F2_dpdpdx = torch.zeros_like(F1_dpdpdx)
F1_dpdxdx = SF.F_dpdxdx(p_rimon, A_torch, a_torch)
F2_dpdxdx = torch.zeros_like(F1_dpdxdx)
print("get_gradient_and_hessian_pytorch time: ", 
      timeit.timeit('get_gradient_and_hessian_pytorch(dual_vars, F1_dp, F2_dp, F1_dx, F2_dx, F1_dpdp, F2_dpdp, F1_dpdx, F2_dpdx, F1_dxdx, F2_dxdx, F1_dpdpdp, F2_dpdpdp, F1_dpdpdx, F2_dpdpdx, F1_dpdxdx, F2_dpdxdx)', globals=globals(), number=number)/number)
alpha_dx, alpha_dxdx = get_gradient_and_hessian_pytorch(dual_vars, F1_dp, F2_dp, F1_dx, F2_dx, F1_dpdp, F2_dpdp, F1_dpdx, F2_dpdx,
                                     F1_dxdx, F2_dxdx, F1_dpdpdp, F2_dpdpdp, F1_dpdpdx, F2_dpdpdx, F1_dpdxdx, F2_dpdxdx)

perturb_a = 0.01*torch.rand(3, dtype=config.torch_dtype, requires_grad=True)
perturb_A = 0.1*torch.rand((3,3), dtype=config.torch_dtype, requires_grad=True)
perturb_A = torch.matmul(perturb_A, perturb_A.transpose(-1, -2))
A = A_torch_original + perturb_A
a = a_torch_original + perturb_a
A_torch = A.unsqueeze(0)
a_torch = a.unsqueeze(0)
p_rimon = rimon_method_pytorch(A_torch, a_torch, B_torch, b_torch)
F1_dp = SF.F_dp(p_rimon, A_torch, a_torch)
F2_dp = SF.F_dp(p_rimon, B_torch, b_torch)
F1_dpdp = SF.F_dpdp(p_rimon, A_torch, a_torch)
F2_dpdp = SF.F_dpdp(p_rimon, B_torch, b_torch)
F1_dx = SF.F_dx(p_rimon, A_torch, a_torch)
F2_dx = torch.zeros_like(F1_dx)
F1_dpdx = SF.F_dpdx(p_rimon, A_torch, a_torch)
F2_dpdx = torch.zeros_like(F1_dpdx)
dual_vars = get_dual_variable_pytorch(F1_dp, F2_dp)
alpha_dx_new = get_gradient_pytorch(dual_vars, F1_dp, F2_dp, F1_dx, F2_dx, F1_dpdp, F2_dpdp, F1_dpdx, F2_dpdx)

perturb = torch.cat([perturb_A.view(-1), perturb_a.view(-1)]).view(1,12,1)
alpha_dx_anticipated = alpha_dx + torch.matmul(alpha_dxdx, perturb).squeeze(-1)
# print(torch.all(alpha_dxdx.transpose(-1, -2) == alpha_dxdx))
# print(alpha_dxdx - alpha_dxdx.transpose(-1, -2))
print("perturbation:")
print(perturb)
print("alpha_dx:")
print(alpha_dx)
print("alpha_dx_anticipated:")
print(alpha_dx_anticipated)
print("alpha_dx_new:")
print(alpha_dx_new)