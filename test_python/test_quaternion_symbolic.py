import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import numpy as np
import torch
import timeit
from cores.symbolic.diff_opt_helper import DiffOptHelper
from cores.utils.rotation_utils import get_rot_matrix_from_quat, get_quat_from_rot_matrix, get_rot_matrix_from_euler_zyx, sp_get_rot_matrix_from_quat
from cores.configuration.configuration import Configuration
import sympy as sp
from cores.rimon_method_python.rimon_method import rimon_method_numpy
config = Configuration()

qx, qy, qz, qw = sp.symbols('qx qy qz qw')
p1, p2, p3 = sp.symbols('px py pz')
A11, A22, A33 = sp.symbols('A11 A22 A33')
B11, B22, B33 = sp.symbols('B11 B22 B33')
a1, a2, a3 = sp.symbols('a1 a2 a3')
b1, b2, b3 = sp.symbols('b1 b2 b3')

p_vars = [p1, p2, p3]
theta_vars = [qx, qy, qz, qw, a1, a2, a3]
dummy_vars = [A11, A22, A33, B11, B22, B33, b1, b2, b3]

p_vec_sp = sp.Matrix(p_vars)
a_vec_sp = sp.Matrix([a1, a2, a3])
A_sp = sp.Matrix([[A11, 0, 0], [0, A22, 0], [0, 0, A33]])
b_vec_sp = sp.Matrix([b1, b2, b3])
B_sp = sp.Matrix([[B11, 0, 0], [0, B22, 0], [0, 0, B33]])
quat_vars = [qx, qy, qz, qw]

R_b_to_w = sp_get_rot_matrix_from_quat(quat_vars)
obj = (p_vec_sp-a_vec_sp).T @ R_b_to_w @ A_sp @ R_b_to_w.T @ (p_vec_sp-a_vec_sp) 
con = (p_vec_sp-b_vec_sp).T @ B_sp @ (p_vec_sp-b_vec_sp) 
cons = [sp.simplify(obj)[0,0], sp.simplify(con)[0,0]]

DO = DiffOptHelper(cons, p_vars, theta_vars, dummy_vars)

euler = [0.01,0.02,0.03]
R = get_rot_matrix_from_euler_zyx(euler)
quat_val = get_quat_from_rot_matrix(R)
A_val = np.array([[1.0, 0.0, 0.0],
                [0.0, 2.0, 0.0],
                [0.0, 0.0, 3.0]], dtype=config.np_dtype)
a_val = np.array([0.1, 0.3, 0.2], dtype=config.np_dtype)
B_val = np.array([[1, 0.0, 0.0],
              [0.0, 2, 0.0],
              [0.0, 0.0, 1]], dtype=config.np_dtype)
b_val = np.array([3.1, 3.2, 3.3], dtype=config.np_dtype)

p_val = rimon_method_numpy(R @ A_val @ R.T, a_val, B_val, b_val)
theta_val = np.concatenate((quat_val, a_val))
dummy_val = np.concatenate((np.diag(A_val), np.diag(B_val), b_val))
dual_val = [np.linalg.norm(2 * R @ A_val @ R.T @ (p_val-a_val))/np.linalg.norm(2*B_val @ (p_val-b_val))]
print("p_val: ", p_val)
print("theta_val: ", theta_val)
print("dummy_val: ", dummy_val)
print("dual_val: ", dual_val)

grad_alpha, grad_p, grad_dual = DO.get_gradient(p_val, theta_val, dual_val, dummy_val)
print('grad_alpha: ', grad_alpha)
print('grad_p: ', grad_p)
print('grad_dual: ', grad_dual)

grad_alpha, grad_p, grad_dual, heissian_alpha, hessian_p, hessian_dual = DO.get_gradient_and_hessian(p_val, theta_val, dual_val, dummy_val)
print('grad_alpha: ', grad_alpha)
print('grad_p: ', grad_p)
print('grad_dual: ', grad_dual)
print('heissian_alpha: ', heissian_alpha)
print('hessian_p: ', hessian_p)
print('hessian_dual: ', hessian_dual)
