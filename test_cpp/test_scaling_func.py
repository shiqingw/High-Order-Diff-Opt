import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from cores_cpp import diffOptCpp as DOC
import numpy as np
import scipy as sp
import timeit
import sys
import torch
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from cores.configuration.configuration import Configuration
config = Configuration()
from cores.scaling_functions.ellipsoid import Ellipsoid_Symmetric
from cores.rimon_method_python.rimon_method import rimon_method_numpy

a = np.array([0.1, 0.2, 0.3], dtype=config.np_dtype)
A = np.array([[2, 0.1, 0.3],
              [0.1, 2, 0.2],
              [0.3, 0.2, 3]], dtype=config.np_dtype)
# print(sp.linalg.eigvals(A))

b = np.array([3.3, 3.2, 3.1], dtype=config.np_dtype)
B = np.array([[1, 0.01, 0.03],
              [0.01, 3, 0.02],
              [0.03, 0.02, 1]], dtype=config.np_dtype)
# print(sp.linalg.eigvals(B))

p_rimon = rimon_method_numpy(A, a, B, b)

a_torch = torch.tensor(a, dtype=config.torch_dtype).unsqueeze(0)
A_torch = torch.tensor(A, dtype=config.torch_dtype).unsqueeze(0)
b_torch = torch.tensor(b, dtype=config.torch_dtype).unsqueeze(0)
B_torch = torch.tensor(B, dtype=config.torch_dtype).unsqueeze(0)
p_rimon_torch = torch.tensor(p_rimon, dtype=config.torch_dtype).unsqueeze(0)

SF = Ellipsoid_Symmetric()
p_rimon_xtensor = DOC.rimonMethodXtensor(A, a, B, b)
print(p_rimon)
print(p_rimon_xtensor)
print()

# print(SF.F(p_rimon_torch, A_torch, a_torch))
# print(DOC.F(p_rimon_xtensor, a, A))
# print()

# print(SF.F_dp(p_rimon_torch, A_torch, a_torch))
# print(DOC.F_dp(p_rimon_xtensor, a, A))
# print()

# print(SF.F_dpdp(p_rimon_torch, A_torch, a_torch))
# print(DOC.F_dpdp(A))
# print()

# print(SF.F_dpdpdp(p_rimon_torch, A_torch, a_torch))
# print(DOC.F_dpdpdp())
# print()

# print(SF.F_dx(p_rimon_torch, A_torch, a_torch))
# print(DOC.F_dy(p_rimon_xtensor, a, A))
# print()

# print(SF.F_dxdx(p_rimon_torch, A_torch, a_torch))
# print(DOE.F_dydy(p_rimon_xtensor, a, A))
# print()

# print(SF.F_dpdx(p_rimon_torch, A_torch, a_torch))
# print(DOC.F_dpdy(p_rimon_xtensor, a, A))
# print()

# print(SF.F_dpdpdx(p_rimon_torch, A_torch, a_torch))
# print(DOC.F_dpdpdy(A))
# print()

# print(SF.F_dpdxdx(p_rimon_torch, A_torch, a_torch))
# print(DOC.F_dpdydy(A))