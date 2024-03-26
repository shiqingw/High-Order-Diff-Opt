import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import numpy as np
from scipy.spatial.transform import Rotation

from cores_cpp import diffOptCpp as DOC
import torch
from cores.differentiable_optimization.ellipsoid_quat_pos import Ellipsoid_Quat_Pos

np.random.seed(0)
rot = Rotation.random()
R = rot.as_matrix()
q = rot.as_quat()
d = np.random.rand(3)
D = np.diag(d**2) + np.eye(3)

B = np.random.rand(3, 3)
B = B @ B.T + np.eye(3)

a = np.random.rand(3)
b = np.random.rand(3) + 5

alpha, p_rimon, alpha_dx, alpha_dxdx = DOC.getGradientAndHessianEllipsoids(a,q,D,R,B,b)
# print(alpha_dxdx)
print()

euler = np.array([0.01, 0.01, 0.01])
rot_perturb = Rotation.from_euler('zyx', euler)
R_perturb = rot_perturb.as_matrix()
R_new = R_perturb @ R 
q_new = Rotation.from_matrix(R_new).as_quat()
q_perturb = q_new - q
a_new = a + np.random.rand(3)*0.1
a_perturb = a_new - a
alpha_new, p_rimon_new, alpha_dx_new, alpha_dxdx_new = DOC.getGradientAndHessianEllipsoids(a_new,q_new,D,R_new,B,b)

delta = np.concatenate([q_perturb, a_perturb])
alpha_approx = alpha_dx @ delta + alpha
alpha_dx_approx = alpha_dxdx @ delta + alpha_dx 

print("delta:", delta)
print("alpha:", alpha)
print("alpha_new:", alpha_new)
print("alpha_approx:", alpha_approx)
print()

print("alpha_dx:", alpha_dx)
print("alpha_dx_new:", alpha_dx_new)
print("alpha_dx_approx:", alpha_dx_approx)
