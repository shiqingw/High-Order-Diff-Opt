import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import numpy as np
from cores_cpp import diffOptCpp as DOC
from cores.utils.rotation_utils import get_rot_matrix_from_quat, get_quat_from_rot_matrix, get_rot_matrix_from_euler_zyx
from cores.differentiable_optimization.quat_diff_utils import Quaternion_RDRT_Symmetric
from cores.configuration.configuration import Configuration
config = Configuration()

import sympy as sp

theta = sp.symbols('theta', real=True)
d11, d22 = sp.symbols('d11 d22', real=True)

R = sp.Matrix([[sp.cos(theta), -sp.sin(theta)],
                      [sp.sin(theta), sp.cos(theta)]])
D = sp.Matrix([[d11, 0], [0, d22]])
A = R @ D @ R.T

theta_np = np.random.rand()
d11_np = np.random.rand()
d22_np = np.random.rand()
D_np = np.array([[d11_np, 0], [0, d22_np]])

substitution_pairs = {theta: theta_np,
                      d11: d11_np,
                      d22: d22_np}

# print(A.diff(theta).subs(substitution_pairs))
# print(DOC.ellipse_RDRT_dtheta(theta_np, D_np))
# print()

print(A.diff(theta).diff(theta).subs(substitution_pairs))
print(DOC.ellipse_RDRT_dthetadtheta(theta_np, D_np))
