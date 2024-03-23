import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import numpy as np
import torch
from cores_cpp import diffOptCpp as DOC
from cores.utils.rotation_utils import get_rot_matrix_from_quat, get_quat_from_rot_matrix, get_rot_matrix_from_euler_zyx
from cores.differentiable_optimization.quat_diff_utils import Quaternion_RDRT_Symmetric
from cores.configuration.configuration import Configuration
config = Configuration()

euler = [0.01,0.02,0.6]
R = get_rot_matrix_from_euler_zyx(euler)
q = get_quat_from_rot_matrix(R)
D = np.array([[1.0, 0.0, 0.0],
                [0.0, 2.0, 0.0],
                [0.0, 0.0, 3.0]], dtype=config.np_dtype)

R_torch = torch.tensor(R, dtype=config.torch_dtype).unsqueeze(0)
D_torch = torch.tensor(D, dtype=config.torch_dtype).unsqueeze(0)
q_torch = torch.tensor(q, dtype=config.torch_dtype).unsqueeze(0)

RDRT = Quaternion_RDRT_Symmetric()
print(RDRT.RDRT_dq(q_torch, D_torch, R_torch).squeeze(0)-DOC.RDRT_dq(q, D, R))
print()
print(RDRT.RDRT_dqdq(q_torch, D_torch, R_torch).squeeze(0)-DOC.RDRT_dqdq(q, D))
print()