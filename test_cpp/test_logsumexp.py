import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import numpy as np
from cores_cpp import diffOptCpp as DOC
import sympy as sp
import timeit
from cores.configuration.configuration import Configuration
config = Configuration()

p1, p2, p3 = sp.symbols('px py pz')
p_vars = [p1, p2, p3]
p_vec = sp.Matrix([p1, p2, p3])

n_v = 3
n_inq = 10
B_np = np.random.rand(n_inq, n_v).astype(config.np_dtype)
b_np = np.random.rand(n_inq).astype(config.np_dtype)
kappa = 10


# Convert A and b to sympy matrices for symbolic operations
A_sp = sp.Matrix(B_np.tolist())
b_sp = sp.Matrix(b_np.tolist())

# Perform the symbolic operation
# Note: The matrix-vector product and addition here are element-wise, adjust if necessary
exp_terms = [sp.exp(kappa * ((A_sp[i, :] @ p_vec)[0,0] + b_sp[i])) for i in range(n_inq)]
F = sp.log(sum(exp_terms)/n_inq) +1 

p_np = np.random.rand(3)
subs_dict = {p1: p_np[0], p2: p_np[1], p3: p_np[2]}

# F_val, F_dp, F_dpdp, F_dpdpdp = DOC.getLogSumExpDerivatives(p_np, B_np, b_np, kappa)

# print(F.subs(subs_dict))
# print(F_val)

# grad = sp.Matrix([F.diff(var) for var in p_vars])
# print(grad.subs(subs_dict))
# print(F_dp)

# hessian = sp.Matrix([[F.diff(var1).diff(var2) for var1 in p_vars] for var2 in p_vars])
# print(hessian.subs(subs_dict))
# print(F_dpdp)

# for i in range(n_v):
#     layer = hessian.diff(p_vars[i])
#     print(layer.subs(subs_dict))
#     print(F_dpdpdp[:,:,i])

# Ellipsoid parameters
D_np = np.random.rand(n_v, n_v).astype(config.np_dtype)
a_np = np.random.rand(n_v).astype(config.np_dtype)
q_np = np.random.rand(4).astype(config.np_dtype)
R_np = np.random.rand(n_v, n_v).astype(config.np_dtype)

number = 1000
print("Avg time to compute the gradient and hessian C++: ",
        timeit.timeit('DOC.getGradientAndHessianEllipsoidAndLogSumExp(p_np, a_np, q_np, D_np, R_np, B_np, b_np, kappa)', globals=globals(), number=number)/number)

alpha, alpha_dx, alpha_dxdx = DOC.getGradientAndHessianEllipsoidAndLogSumExp(p_np, a_np, q_np, D_np, R_np, B_np, b_np, kappa)
print(alpha)
print(alpha_dx.shape)
print(alpha_dxdx.shape)