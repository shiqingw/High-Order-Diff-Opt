import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import numpy as np
import torch
from cores.differentiable_optimization.quat_diff_utils import Quaternion_RDRT
from cores.utils.rotation_utils import get_rot_matrix_from_quat, get_quat_from_rot_matrix
from cores.configuration.configuration import Configuration
config = Configuration()

QT = Quaternion_RDRT()
R = np.array([[1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0]], dtype=config.np_dtype)
quat = get_quat_from_rot_matrix(R)
quat_torch_original = torch.tensor(quat, dtype=config.torch_dtype)
quat_torch = quat_torch_original.unsqueeze(0)
D = np.array([[1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0]], dtype=config.np_dtype)
D_torch_original = torch.tensor(D, dtype=config.torch_dtype)
D_torch = D_torch_original.unsqueeze(0)
RDRT_dq = QT.RDRT_dq(quat_torch, D_torch)
print(RDRT_dq.shape)
RDRT_dqdq = QT.RDRT_dqdq(quat_torch, D_torch)
print(RDRT_dqdq.shape)