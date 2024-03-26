import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import numpy as np
from cores_cpp import diffOptCpp as DOC
import torch
import timeit
import time
from cores.differentiable_optimization.quat_diff_utils import Quaternion_RDRT
from cores.differentiable_optimization.ellipsoid_quat_pos import Ellipsoid_Quat_Pos
from cores.utils.rotation_utils import get_rot_matrix_from_quat, get_quat_from_rot_matrix, get_rot_matrix_from_euler_zyx
from cores.configuration.configuration import Configuration
from torch.profiler import profile, record_function, ProfilerActivity
config = Configuration()

DO = Ellipsoid_Quat_Pos()
euler = [0.01,0.02,0.03]
R = get_rot_matrix_from_euler_zyx(euler)
quat = get_quat_from_rot_matrix(R)
quat_torch_original = torch.tensor(quat, dtype=config.torch_dtype)
quat_torch = quat_torch_original.unsqueeze(0)
R_torch_original = torch.tensor(R, dtype=config.torch_dtype)
R_torch = R_torch_original.unsqueeze(0)
D = np.array([[1.0, 0.0, 0.0],
                [0.0, 2.0, 0.0],
                [0.0, 0.0, 3.0]], dtype=config.np_dtype)
D_torch_original = torch.tensor(D, dtype=config.torch_dtype)
D_torch = D_torch_original.unsqueeze(0)
a = np.array([0.1, 0.3, 0.2], dtype=config.np_dtype)
a_torch_original = torch.tensor(a, dtype=config.torch_dtype)
a_torch = a_torch_original.unsqueeze(0)
A_torch = torch.matmul(torch.matmul(R_torch, D_torch), R_torch.transpose(-1,-2))

b = np.array([3.1, 3.2, 3.3], dtype=config.np_dtype)
B = np.array([[1, 0.0, 0.0],
              [0.0, 2, 0.0],
              [0.0, 0.0, 1]], dtype=config.np_dtype)
B_torch_original = torch.tensor(B, dtype=config.torch_dtype, requires_grad=False)
B_torch = B_torch_original.unsqueeze(0)
b_torch_original = torch.tensor(b, dtype=config.torch_dtype, requires_grad=False)
b_torch = b_torch_original.unsqueeze(0)

N = 1
A_torch = A_torch.repeat(N,1,1)
a_torch = a_torch.repeat(N,1)
B_torch = B_torch.repeat(N,1,1)
b_torch = b_torch.repeat(N,1)
quat_torch = quat_torch.repeat(N,1)
D_torch = D_torch.repeat(N,1,1)


# Compute the gradient
# print("==> Compute the gradient")

number = 1000
# print("Avg time to compute the gradient pytorch: ",
#       timeit.timeit('DO.get_gradient(a_torch, quat_torch, D_torch, R_torch, B_torch, b_torch)', globals=globals(), number=number)/number)

# with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof_gradient:
#     with record_function("get_gradient"):
#         p_rimon, alpha_dx = DO.get_gradient(a_torch, quat_torch, D_torch, R_torch, B_torch, b_torch)

# traced_get_gradient = torch.jit.trace(DO.get_gradient, (a_torch, quat_torch, D_torch, R_torch, B_torch, b_torch))
# print("Avg time to compute the gradient using trace: ",
#       timeit.timeit('traced_get_gradient(a_torch, quat_torch, D_torch, R_torch, B_torch, b_torch)', globals=globals(), number=number)/number)

# p_rimon_torch, alpha_dx_torch = DO.get_gradient(a_torch, quat_torch, D_torch, R_torch, B_torch, b_torch)
# print("p_rimon_torch: ", p_rimon_torch)
# print("alpha_dx_torch: ", alpha_dx_torch)

# print("Avg time to compute the gradient C++: ",
#       timeit.timeit('DOE.getGradientEllipsoids(a, quat, D, R, B, b)', globals=globals(), number=number)/number)

# F, p_rimon, alpha_dx = DOE.getGradientEllipsoids(a, quat, D, R, B, b)
# print("F: ", F)
# print("p: ", p_rimon)
# print("alpha: ", alpha_dx)

# Compute the gradient and hessian
# print("==> Compute the gradient and hessian")

# print("Avg time to compute the gradient and hessian: ",
#         timeit.timeit('DO.get_gradient_and_hessian(a_torch, quat_torch, D_torch, R_torch, B_torch, b_torch)', globals=globals(), number=number)/number)

# with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof_gradient_hessian:
#     with record_function("get_gradient_and_hessian"):
#         p_rimon, alpha_dx, alpha_dxdx = DO.get_gradient_and_hessian(a_torch, quat_torch, D_torch, R_torch, B_torch, b_torch)

# traced_get_gradient_and_hessian = torch.jit.trace(DO.get_gradient_and_hessian, (a_torch, quat_torch, D_torch, R_torch, B_torch, b_torch))
# print("Avg time to compute the gradient using trace: ",
#       timeit.timeit('traced_get_gradient_and_hessian(a_torch, quat_torch, D_torch, R_torch, B_torch, b_torch)', globals=globals(), number=number)/number)

# print("Avg time to compute the gradient and hessian C++: ",
#         timeit.timeit('DOC.getGradientAndHessianEllipsoids(a, quat, D, R, B, b)', globals=globals(), number=number)/number)

# F, p_rimon, alpha_dx, alpha_dxdx = DOC.getGradientAndHessianEllipsoids(a, quat, D, R, B, b)

# print("F: ", F)
# print("p: ", p_rimon)
# print("alpha_dx: ", alpha_dx)
# print("alpha_dxdx: ", alpha_dxdx)

p_rimon_torch, alpha_dx_torch, alpha_dxdx_torch = DO.get_gradient_and_hessian(a_torch, quat_torch, D_torch, R_torch, B_torch, b_torch)
print("p_rimon_torch: ", p_rimon_torch)
print("alpha_dx_torch: ", alpha_dx_torch)
print("alpha_dxdx_torch: ", alpha_dxdx_torch)


# # euler = [0.01,0.01,0.01]
# # R_new = get_rot_matrix_from_euler_zyx(euler)
# # quat_new = get_quat_from_rot_matrix(R_new)
# # print(quat_new)
# # quat_new_torch_original = torch.tensor(quat_new, dtype=config.torch_dtype)
# # quat_new_torch = quat_new_torch_original.unsqueeze(0)
# # R_new_torch_original = torch.tensor(R_new, dtype=config.torch_dtype)
# # R_new_torch = R_new_torch_original.unsqueeze(0)
# # a_new = np.array([0.01, 0.01, 0.01], dtype=config.np_dtype)
# # a_new_torch_original = torch.tensor(a_new, dtype=config.torch_dtype)
# # a_new_torch = a_new_torch_original.unsqueeze(0)
# # A_new_torch = torch.matmul(torch.matmul(R_new_torch, D_torch), R_new_torch.transpose(-1,-2))
# # alpha_dx_new = DO.get_gradient(A_new_torch, a_new_torch, B_torch, b_torch, quat_new_torch, D_torch)
# # dx = torch.cat((quat_new_torch - quat_torch, a_new_torch - a_torch), dim=1)
# # alpha_dx_anticipated = alpha_dx + torch.matmul(alpha_dxdx, dx.unsqueeze(-1)).squeeze(-1)
# # print("dx:", dx)
# # print("alpha_dx:", alpha_dx)
# # print("alpha_dx_new:", alpha_dx_new)
# # print("alpha_dx_anticipated:", alpha_dx_anticipated)
# # prof_gradient.export_chrome_trace("trace_grdient.json")
# # prof_gradient_hessian.export_chrome_trace("trace_grdient_hessian.json")