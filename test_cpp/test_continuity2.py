import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import numpy as np
from scipy.spatial.transform import Rotation

import torch
from cores.differentiable_optimization.ellipsoid_quat_pos import Ellipsoid_Quat_Pos

rot = Rotation.random()
R = rot.as_matrix()
q = rot.as_quat()
d = np.random.rand(3)
D = np.diag(d**2) + np.eye(3)

B = np.random.rand(3, 3)
B = B @ B.T + np.eye(3)

a = np.random.rand(3)
b = np.random.rand(3) + 5

DO = Ellipsoid_Quat_Pos()
a_torch = torch.tensor(a).unsqueeze(0)
q_torch = torch.tensor(q).unsqueeze(0)
D_torch = torch.tensor(D).unsqueeze(0)
R_torch = torch.tensor(R).unsqueeze(0)
B_torch = torch.tensor(B).unsqueeze(0)
b_torch = torch.tensor(b).unsqueeze(0)
p_rimon_torch, alpha_dx_torch, alpha_dxdx_torch = DO.get_gradient_and_hessian(a_torch, q_torch, D_torch, R_torch, B_torch, b_torch)
