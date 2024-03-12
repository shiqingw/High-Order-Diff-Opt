from cvxpylayers.torch import CvxpyLayer
import cvxpy as cp
import numpy as np
import timeit
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from cores.rimon_method_python.rimon_method import rimon_method_pytorch
import torch
from cores.configuration.configuration import Configuration
config = Configuration()
from cores.differentiable_optimization.diff_opt_helper import get_gradient_pytorch
from cores.scaling_functions.quadratics import ellipsoid_value, ellipsoid_gradient, ellipsoid_hessian
# Define cvxpy problem
print("==> Define cvxpy problem")
nv = 2
_p = cp.Variable(nv)
_E1_Q_sqrt = cp.Parameter((nv,nv))
_E1_b = cp.Parameter(nv)
_E1_c = cp.Parameter()
_E2_Q_sqrt = cp.Parameter((nv,nv))
_E2_b = cp.Parameter(nv)
_E2_c = cp.Parameter()
obj = cp.Minimize(cp.sum_squares(_E1_Q_sqrt @ _p) + _E1_b.T @ _p + _E1_c)
cons = [cp.sum_squares(_E2_Q_sqrt @ _p) + _E2_b.T @ _p + _E2_c <= 1]

problem = cp.Problem(obj, cons)
assert problem.is_dcp()
assert problem.is_dpp()

# Define parameters
print("==> Define parameters")
A = np.array([[2, 0], [0, 1]], dtype=config.np_dtype)
a = np.array([0, 0], dtype=config.np_dtype)
B = np.array([[1, 0], [0, 2]], dtype=config.np_dtype)
b = np.array([2, 1], dtype=config.np_dtype)

A_sqrt = np.linalg.cholesky(A) # A = L @ L.T 
E1_Q_sqrt_np = A_sqrt
E1_Q_np = E1_Q_sqrt_np.T @ E1_Q_sqrt_np
E1_b_np = -2 * E1_Q_np @ a
E1_c_np = a.T @ E1_Q_np @ a

B_sqrt = np.linalg.cholesky(B)
E2_Q_sqrt_np = B_sqrt
E2_Q_np = E2_Q_sqrt_np.T @ E2_Q_sqrt_np
E2_b_np = -2 * E2_Q_np @ b
E2_c_np = b.T @ E2_Q_np @ b

# Solve problem
print("==> Solve problem")
_E1_Q_sqrt.value = E1_Q_sqrt_np
_E1_b.value = E1_b_np
_E1_c.value = E1_c_np
_E2_Q_sqrt.value = E2_Q_sqrt_np
_E2_b.value = E2_b_np
_E2_c.value = E2_c_np

problem.solve(solver=cp.ECOS)
number = 100
print("cvxpy avg time: ", timeit.timeit('problem.solve(solver=cp.ECOS)', globals=globals(), number=number)/number)
print("cvxpy p: ", _p.value)

cvxpylayer = CvxpyLayer(problem, parameters=[_E1_Q_sqrt, _E1_b, _E1_c, _E2_Q_sqrt, _E2_b, _E2_c], variables=[_p], gp=False)
solver_args = {"solve_method": "ECOS"}
_E1_Q_sqrt_val = torch.tensor(E1_Q_sqrt_np, device=config.device, requires_grad=True)
_E1_b_val = torch.tensor(E1_b_np, device=config.device, requires_grad=True)
_E1_c_val = torch.tensor(E1_c_np, device=config.device, requires_grad=True)
_E2_Q_sqrt_val = torch.tensor(E2_Q_sqrt_np, device=config.device, requires_grad=True)
_E2_b_val = torch.tensor(E2_b_np, device=config.device, requires_grad=True)
_E2_c_val = torch.tensor(E2_c_np, device=config.device, requires_grad=True)
p_sol = cvxpylayer(_E1_Q_sqrt_val, _E1_b_val, _E1_c_val, _E2_Q_sqrt_val, _E2_b_val, _E2_c_val, solver_args=solver_args)
p_sol = p_sol[0]
print("cvxpylayer sol:", p_sol)
alpha = p_sol.T @ _E1_Q_sqrt_val @ _E1_Q_sqrt_val.T @ p_sol + _E1_b_val.T @ p_sol + _E1_c_val
alpha.backward()
print("cvxpylayer grad:")
print("_E1_Q_sqrt_val.grad:", _E1_Q_sqrt_val.grad)
print("_E1_b_val.grad:", _E1_b_val.grad)
print("_E1_c_val.grad:", _E1_c_val.grad)
print("_E2_Q_sqrt_val.grad:", _E2_Q_sqrt_val.grad)
print("_E2_b_val.grad:", _E2_b_val.grad)
print("_E2_c_val.grad:", _E2_c_val.grad)

# Rimons method
print("==> Rimons method")
A_torch = torch.from_numpy(A).to(config.device).unsqueeze(0)
a_torch = torch.from_numpy(a).to(config.device).unsqueeze(0)
B_torch = torch.from_numpy(B).to(config.device).unsqueeze(0)
b_torch = torch.from_numpy(b).to(config.device).unsqueeze(0)
print("pytorch avg time: ", 
      timeit.timeit('rimon_method_pytorch(A_torch, a_torch, B_torch, b_torch)', globals=globals(), number=number)/number)
p_rimon = rimon_method_pytorch(A_torch, a_torch, B_torch, b_torch)
print("rimon p: ", p_rimon)

F1_dp = ellipsoid_gradient(p_rimon, A_torch, a_torch)
F2_dp = ellipsoid_gradient(p_rimon, B_torch, b_torch)
F1_dpdp = ellipsoid_hessian(A_torch)
F2_dpdp = ellipsoid_hessian(B_torch)
#F1_dx, F2_dx, F1_dpdx, F2_dpdx

