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
p1, p2, p3 = sp.symbols('p1 p2 p3')
v1, v2, v3 = sp.symbols('v1 v2 v3')
d11, d22, d33 = sp.symbols('d11 d22 d33')
a1, a2, a3 = sp.symbols('a1 a2 a3')
b1, b2, b3 = sp.symbols('b1 b2 b3')

r11, r12, r13, r21, r22, r23, r31, r32, r33 = sp.symbols('r11 r12 r13 r21 r22 r23 r31 r32 r33', real=True)
pxx, pxy, pxz, pxw, pyy, pyz, pyw, pzz, pzw, pww = sp.symbols('pxx pxy pxz pxw pyy pyz pyw pzz pzw pww', real=True)

substitution_expr = {qw**2+qx**2:  (r11+1)/2,
                    qx*qy-qw*qz: r12/2,
                    qx*qz+qw*qy: r13/2,
                    qx*qy+qw*qz: r21/2,
                    qw**2+qy**2: (r22+1)/2,
                    qy*qz-qw*qx: r23/2,
                    qx*qz-qw*qy: r31/2,
                    qy*qz+qw*qx: r32/2,
                    qw**2+qz**2: (r33+1)/2,
                    2*qw**2+2*qx**2-1: r11,
                    2*qw**2+2*qy**2-1: r22,
                    2*qw**2+2*qz**2-1: r33,
                    qx*qx: pxx,
                    qx*qy: pxy,
                    qx*qz: pxz,
                    qx*qw: pxw,
                    qy*qy: pyy,
                    qy*qz: pyz,
                    qy*qw: pyw,
                    qz*qz: pzz,
                    qz*qw: pzw,
                    qw*qw: pww,
                    p1-a1: v1,
                    p2-a2: v2,
                    p3-a3: v3,
                    a1-p1: -v1,
                    a2-p2: -v2,
                    a3-p3: -v3}


p_vars = [p1, p2, p3]
theta_vars = [qx, qy, qz, qw, a1, a2, a3]

p_vec_sp = sp.Matrix(p_vars)
a_vec_sp = sp.Matrix([a1, a2, a3])
D_sp = sp.Matrix([[d11, 0, 0], [0, d22, 0], [0, 0, d33]])
quat_vars = [qx, qy, qz, qw]

R_b_to_w = sp_get_rot_matrix_from_quat(quat_vars)
obj = (p_vec_sp-a_vec_sp).T @ R_b_to_w @ D_sp @ R_b_to_w.T @ (p_vec_sp-a_vec_sp) 

# obj_dx = sp.Matrix([obj.diff(var) for var in theta_vars])
# for i in range(len(theta_vars)):
#     print(f'F_dx[:,{i}] =', sp.simplify(obj_dx[i].subs(substitution_expr)))

# hessian = sp.Matrix([[obj.diff(var1).diff(var2) for var2 in theta_vars] for var1 in p_vars])
# for i in range(hessian.rows):
#     for j in range(hessian.cols):
#         print(f'F_dpdx[:,{i},{j}] =', sp.simplify(hessian[i,j].subs(substitution_expr)))
        # print()

# hessian = sp.Matrix([[obj.diff(var1).diff(var2) for var2 in theta_vars] for var1 in theta_vars])
# for i in range(hessian.rows):
#     for j in range(i, hessian.cols):
#         print(f'F_dxdx[:,{i},{j}] =', sp.simplify(sp.simplify(hessian[i,j].subs(substitution_expr)).subs(substitution_expr)))

# hessian = sp.Matrix([[[obj.diff(var1).diff(var2).diff(var3) for var3 in theta_vars] for var2 in p_vars] for var1 in p_vars])
# for i in range(hessian.rows):
#     for j in range(i, hessian.cols):
#         for k in range(len(hessian[i,j])):
#             print(f'F_dpdpdx[:,{i},{j},{k}] =', sp.simplify(hessian[i,j][k].subs(substitution_expr))[0,0])


hessian = sp.Matrix([[[obj.diff(var1).diff(var2).diff(var3) for var3 in theta_vars] for var2 in theta_vars] for var1 in p_vars])
for i in range(hessian.rows):
    for j in range(hessian.cols):
        for k in range(j,len(hessian[i,j])):
            print(f'F_dpdxdx[:,{i},{j},{k}] =', sp.simplify(sp.simplify(hessian[i,j][k].subs(substitution_expr))[0,0].subs(substitution_expr)))