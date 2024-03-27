import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import numpy as np
import cores_cpp.diffOptCpp as DOC
import time
from cores.symbolic.diff_opt_helper_no_simplify import DiffOptHelper
from cores.utils.rotation_utils import get_rot_matrix_from_quat, get_quat_from_rot_matrix, get_rot_matrix_from_euler_zyx, sp_get_rot_matrix_from_quat
from cores.configuration.configuration import Configuration
from scipy.spatial.transform import Rotation
import sympy as sp

config = Configuration()
np.random.seed(0)

qx, qy, qz, qw = sp.symbols('qx qy qz qw')
p1, p2, p3 = sp.symbols('p1 p2 p3')
d11, d22, d33 = sp.symbols('d11 d22 d33')
a1, a2, a3 = sp.symbols('a1 a2 a3')

p_vars = [p1, p2, p3]
theta_vars = [qx, qy, qz, qw, a1, a2, a3]
dummy_vars = [d11, d22, d33]
p_vec_sp = sp.Matrix([p1, p2, p3])
a_vec_sp = sp.Matrix([a1, a2, a3])
D_sp = sp.Matrix([[d11, 0, 0],
                [0, d22, 0],
                [0, 0, d33]])

n_v = 3
n_inq = 10
B_np = np.random.rand(n_inq, n_v).astype(config.np_dtype)
b_np = np.random.rand(n_inq).astype(config.np_dtype)
kappa = 10

# Convert A and b to sympy matrices for symbolic operations
B_sp = sp.Matrix(B_np.tolist())
b_sp = sp.Matrix(b_np.tolist())

# Perform the symbolic operation
exp_terms = [sp.exp(kappa * ((B_sp[i, :] @ p_vec_sp)[0,0] + b_sp[i])) for i in range(n_inq)]
F = sp.log(sum(exp_terms)/n_inq) +1 
F_dp = sp.Matrix([F.diff(var) for var in p_vars])
F_dp_func = sp.lambdify([p_vars], F_dp, 'numpy')

R_b_to_w = sp_get_rot_matrix_from_quat([qx, qy, qz, qw])
obj = (p_vec_sp-a_vec_sp).T @ R_b_to_w @ D_sp @ R_b_to_w.T @ (p_vec_sp-a_vec_sp) 
con = F
cons = [sp.simplify(obj)[0,0], sp.simplify(con)]


p_val = np.random.rand(n_v)
a_val = np.random.rand(n_v)
d11_val = np.random.rand()
d22_val = np.random.rand()
d33_val = np.random.rand()
D_val = np.array([[d11_val, 0, 0],
                [0, d22_val, 0],
                [0, 0, d33_val]])
rot = Rotation.random()
R_val = rot.as_matrix()
q_val = rot.as_quat()
print("p_val: ", p_val)
print("q_val: ", q_val)
print("a_val: ", a_val)
print("D_val: ", D_val)
print("R_val: ", R_val)

print("##############################################")
alpha, alpha_dx, alpha_dxdx = DOC.getGradientAndHessianEllipsoidAndLogSumExp(p_val, a_val, q_val, D_val, R_val, B_np, b_np, kappa)
print('alpha: ', alpha)
print('alpha_dx: ', alpha_dx)
print('alpha_dxdx: ', alpha_dxdx)

print("##############################################")
x_val = np.concatenate((q_val, a_val))
dummy_val = np.array([d11_val, d22_val, d33_val])
obj_dp = 2 * R_val @ D_val @ R_val.T @ (p_val-a_val)
F_dp_val = F_dp_func(p_val).squeeze()
dual_val = [np.linalg.norm(obj_dp)/np.linalg.norm(F_dp_val)]
print("p_val: ", p_val)
print("x_val: ", x_val)
print("dummy_val: ", dummy_val)
print("dual_val: ", dual_val)

print("##############################################")
time_start = time.time()
DO = DiffOptHelper(cons, p_vars, theta_vars, dummy_vars)
time_end = time.time()
print("Time to initialize the DO class: ", time_end-time_start)

print("##############################################")
grad_alpha, grad_p, grad_dual, heissian_alpha, hessian_p, hessian_dual = DO.get_gradient_and_hessian(p_val, x_val, dual_val, dummy_val)
print('grad_alpha: ', grad_alpha)
# print('grad_p: ', grad_p)
# print('grad_dual: ', grad_dual)
print('heissian_alpha: ', heissian_alpha)
# print('hessian_p: ', hessian_p)
# print('hessian_dual: ', hessian_dual)
