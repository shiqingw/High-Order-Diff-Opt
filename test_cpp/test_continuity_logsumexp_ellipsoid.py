import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import cvxpy as cp
import numpy as np
import cores_cpp.diffOptCpp as DOC
from cores.configuration.configuration import Configuration
config = Configuration()

kappa_obs = 1
n_cons_obs = 4
n_v = 2
x_max_np = 10

_A_obs = cp.Parameter((n_cons_obs,n_v))
_b_obs = cp.Parameter(n_cons_obs)
_p = cp.Variable(n_v)
_ellipse_Q_sqrt = cp.Parameter((n_v,n_v))
_ellipse_b = cp.Parameter(n_v)
_ellipse_c = cp.Parameter()
obj = cp.Minimize(cp.sum_squares(_ellipse_Q_sqrt @ _p) + _ellipse_b.T @ _p + _ellipse_c)
cons = [cp.log_sum_exp(kappa_obs*(_A_obs @ _p + _b_obs) - x_max_np) - np.log(n_cons_obs) + x_max_np<= 0]
problem = cp.Problem(obj, cons)
assert problem.is_dcp()
assert problem.is_dpp()

theta_np = 0.1
a_np = np.array([3,3], dtype=config.np_dtype)
D_np = np.diag([1,1.1]).astype(config.np_dtype)
ellipse_coef_sqrt_np = np.linalg.cholesky(D_np)
R_np = np.array([[np.cos(theta_np), -np.sin(theta_np)],
                            [np.sin(theta_np), np.cos(theta_np)]])
ellipse_Q_sqrt_np = ellipse_coef_sqrt_np @ R_np.T
ellipse_Q_np = ellipse_Q_sqrt_np.T @ ellipse_Q_sqrt_np
_ellipse_Q_sqrt.value = ellipse_Q_sqrt_np
_ellipse_b.value = -2 * ellipse_Q_np @ a_np
_ellipse_c.value = a_np.T @ ellipse_Q_np @ a_np

A_obs_np = np.array([[1,0],
                     [0,-1],
                     [-1,0],
                     [0,1]], dtype=config.np_dtype)
b_obs_np = np.array([-1,-1,-1,-1], dtype=config.np_dtype)
_A_obs.value = A_obs_np
_b_obs.value = b_obs_np

problem.solve(solver=cp.ECOS)
p_np = _p.value

alpha, alpha_dx, alpha_dxdx = DOC.getGradientAndHessianEllipseAndLogSumExp(p_np, a_np, theta_np, D_np, R_np, A_obs_np, b_obs_np, kappa_obs)

delta_theta = np.random.rand()*0.01
delta_a = np.random.rand(2)*0.01

theta_np_new = theta_np + delta_theta
a_np_new = a_np + delta_a
R_np_new = np.array([[np.cos(theta_np_new), -np.sin(theta_np_new)],
                            [np.sin(theta_np_new), np.cos(theta_np_new)]])
ellipse_Q_sqrt_np = ellipse_coef_sqrt_np @ R_np_new.T
ellipse_Q_np = ellipse_Q_sqrt_np.T @ ellipse_Q_sqrt_np
_ellipse_Q_sqrt.value = ellipse_Q_sqrt_np
_ellipse_b.value = -2 * ellipse_Q_np @ a_np_new
_ellipse_c.value = a_np_new.T @ ellipse_Q_np @ a_np_new
problem.solve(solver=cp.ECOS)
p_np_new = _p.value
alpha_new, alpha_dx_new, alpha_dxdx_new = DOC.getGradientAndHessianEllipseAndLogSumExp(p_np_new, a_np_new, theta_np_new, D_np, R_np_new, A_obs_np, b_obs_np, kappa_obs)

delta = np.concatenate([np.array([delta_theta]), delta_a])
alpha_dx_approx = alpha_dx + alpha_dxdx @ delta
alpha_approx = alpha + alpha_dx @ delta

print("delta:", delta)
print("alpha:", alpha)
print("alpha_new:", alpha_new)
print("alpha_approx:", alpha_approx)
print()

print("alpha_dx:", alpha_dx)
print("alpha_dx_new:", alpha_dx_new)
print("alpha_dx_approx:", alpha_dx_approx)
