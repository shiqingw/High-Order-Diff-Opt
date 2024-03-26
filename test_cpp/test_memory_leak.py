import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import numpy as np
from scipy.spatial.transform import Rotation

from cores_cpp import diffOptCpp as DOC
import torch
from cores.differentiable_optimization.ellipsoid_quat_pos import Ellipsoid_Quat_Pos

np.random.seed(0)
theta = np.random.rand()
R = np.array([[np.cos(theta), -np.sin(theta)],
              [np.sin(theta), np.cos(theta)]])
d = np.random.rand(2)
D = np.diag(d**2) + np.eye(2)

B = np.random.rand(2, 2)
B = B @ B.T + np.eye(2)

a = np.random.rand(2)
b = np.random.rand(2) + 5

alpha, p_rimon, alpha_dx, alpha_dxdx = DOC.getGradientAndHessianEllipses(a,theta,D,R,B,b)
print(alpha_dxdx)
print()

delta_theta = np.random.rand()*0.1
theta_new = theta + delta_theta
R_new = np.array([[np.cos(theta_new), -np.sin(theta_new)],
                   [np.sin(theta_new), np.cos(theta_new)]])
delta_a = np.random.rand(2)*0.1
a_new = a + delta_a
alpha_new, p_rimon_new, alpha_dx_new, alpha_dxdx_new = DOC.getGradientAndHessianEllipses(a_new,theta_new,D,R_new,B,b)

delta = np.concatenate([np.array([delta_theta]), delta_a])
alpha_dx_approx = alpha_dxdx @ delta + alpha_dx

print("delta:", delta)
print("alpha_dx:", alpha_dx)
print("alpha_dx_new:", alpha_dx_new)
print("alpha_dx_approx:", alpha_dx_approx)
