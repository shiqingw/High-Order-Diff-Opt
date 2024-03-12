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
from cores.differentiable_optimization.diff_opt_helper import get_gradient_pytorch, get_dual_variable_pytorch
from cores.scaling_functions.quadratics import ellipsoid_value, ellipsoid_gradient, ellipsoid_hessian

# Define parameters
print("==> Define parameters")
a = np.array([0, 0, 0], dtype=config.np_dtype)
A = np.array([[1, 0, 0],
              [0, 2, 0],
              [0, 0, 3]], dtype=config.np_dtype)

b = np.array([3, 3, 3], dtype=config.np_dtype)
B = np.array([[1, 0, 0],
              [0, 2, 0],
              [0, 0, 1]], dtype=config.np_dtype)

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
tmp1 = torch.matmul(p_rimon.unsqueeze(-1), p_rimon.unsqueeze(-2)).view(1,-1) 
tmp2 = 2*torch.matmul(A_torch, (a_torch - p_rimon).unsqueeze(-1)).squeeze(-1)
F1_dx = torch.cat((tmp1, tmp2), dim=1)
F2_dx = torch.zeros_like(F1_dx)
tmp1 = torch.zeros((1,3,9), dtype=config.torch_dtype)
tmp1[0,0,0:3] = 2*(p_rimon - a_torch).squeeze()
tmp1[0,1,3:6] = 2*(p_rimon - a_torch).squeeze()
tmp1[0,2,6:9] = 2*(p_rimon - a_torch).squeeze()
tmp2 = 2*A_torch
F1_dpdx = torch.cat((tmp1, tmp2), dim=2)
F2_dpdx = torch.zeros_like(F1_dpdx)
dual_vars = get_dual_variable_pytorch(F1_dp, F2_dp)
alpha_dx = get_gradient_pytorch(dual_vars, F1_dp, F2_dp, F1_dx, F2_dx, F1_dpdp, F2_dpdp, F1_dpdx, F2_dpdx)
end = time.time()
print("pytorch get_gradient time: ", end-start)
print(alpha_dx.shape)
print(alpha_dx)

