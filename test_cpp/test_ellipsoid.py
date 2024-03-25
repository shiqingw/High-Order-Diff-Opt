import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from cores_cpp import diffOptCpp as DOC
import numpy as np
import sympy as sp
import timeit
import sys
from cores.utils.rotation_utils import sp_get_rot_matrix_from_quat, np_get_rot_matrix_from_quat
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from cores.configuration.configuration import Configuration
config = Configuration()

qx, qy, qz, qw = sp.symbols('qx qy qz qw')
p1, p2, p3 = sp.symbols('p1 p2 p3')
d11, d22, d33 = sp.symbols('d11 d22 d33')
a1, a2, a3 = sp.symbols('a1 a2 a3')
A11, A12, A13, A22, A23, A33 = sp.symbols('A11 A12 A13 A22 A23 A33')
p_vars = [p1, p2, p3]
y_vars = [A11, A12, A13, A22, A23, A33, a1, a2, a3]

p = sp.Matrix([p1, p2, p3])
a = sp.Matrix([a1, a2, a3])
D = sp.Matrix([[d11, 0, 0], [0, d22, 0], [0, 0, d33]])
R = sp_get_rot_matrix_from_quat([qx, qy, qz, qw])
A = sp.Matrix([[A11, A12, A13], [A12, A22, A23], [A13, A23, A33]])
F = sp.simplify((p-a).T @ A @ (p-a))[0,0]

theta_np = np.random.rand()
p_np = np.random.rand(3)
a_np = np.random.rand(3)
d11_np = np.random.rand()
d22_np = np.random.rand()
d33_np = np.random.rand()
D_np = np.diag([d11_np, d22_np, d33_np])
q_np = np.random.rand(4)
R_np = np_get_rot_matrix_from_quat(q_np)
A_np = R_np @ D_np @ R_np.T

substitution_pairs = {p1: p_np[0],
                      p2: p_np[1],
                      p3: p_np[2],
                      a1: a_np[0],
                      a2: a_np[1],
                      a3: a_np[2],
                      d11: d11_np,
                      d22: d22_np,
                      d33: d22_np,
                      qx: q_np[0],
                      qy: q_np[1],
                      qz: q_np[2],
                      qw: q_np[3],
                      A11: A_np[0,0],
                      A12: A_np[0,1],
                      A13: A_np[0,2],
                      A22: A_np[1,1],
                      A23: A_np[1,2],
                      A33: A_np[2,2]
                      }

# F_np = DOC.ellipsoid_F(p_np, a_np, A_np)
# print(F.subs(substitution_pairs))
# print(F_np)

# F_dp_np = DOC.ellipsoid_dp(p_np, a_np, A_np)
# F_dp = sp.Matrix([F.diff(var1) for var1 in p_vars]).subs(substitution_pairs)
# print(F_dp_np)
# print(F_dp)

# F_dpdp_np = DOC.ellipsoid_dpdp(A_np)
# F_dpdp = sp.Matrix([[F.diff(var1).diff(var2) for var1 in p_vars] for var2 in p_vars]).subs(substitution_pairs)
# print(F_dpdp_np)
# print(F_dpdp)

# F_dpdpdp_np = DOC.ellipsoid_dpdpdp()
# F_dpdpdp = [sp.Matrix([[F.diff(var1).diff(var2).diff(var3) for var1 in p_vars] for var2 in p_vars]).subs(substitution_pairs) for var3 in p_vars]
# print(F_dpdpdp_np)
# print(F_dpdpdp)

# F_dy_np = DOC.ellipsoid_dy(p_np, a_np, A_np)
# F_dy = sp.Matrix([F.diff(var1) for var1 in y_vars]).subs(substitution_pairs)
# print(F_dy_np)
# print(F_dy)

# F_dpdy_np = DOC.ellipsoid_dpdy(p_np, a_np, A_np)
# F_dpdy = sp.Matrix([[F.diff(var1).diff(var2) for var1 in y_vars] for var2 in p_vars]).subs(substitution_pairs)
# print(F_dpdy_np)
# print(F_dpdy)

# F_dydy_np = DOC.ellipsoid_dydy(p_np, a_np, A_np)
# F_dydy = sp.Matrix([[F.diff(var1).diff(var2) for var1 in y_vars] for var2 in y_vars]).subs(substitution_pairs)
# print(F_dydy_np)
# print(F_dydy)

# F_dpdpdy_np = DOC.ellipsoid_dpdpdy(A_np)
# F_dpdpdy = [sp.Matrix([[F.diff(var3).diff(var2).diff(var1) for var1 in y_vars] for var2 in p_vars]).subs(substitution_pairs) for var3 in p_vars]
# print(F_dpdpdy_np)
# print(F_dpdpdy)

F_dpdydy_np = DOC.ellipsoid_dpdydy(A_np)
F_dpdydy = [sp.Matrix([[F.diff(var3).diff(var2).diff(var1) for var1 in y_vars] for var2 in y_vars]).subs(substitution_pairs) for var3 in p_vars]
print(F_dpdydy_np)
print(F_dpdydy)