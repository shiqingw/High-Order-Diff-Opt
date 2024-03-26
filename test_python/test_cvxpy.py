import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import cvxpy as cp
import numpy as np
import time
import timeit
from cores.configuration.configuration import Configuration
config = Configuration()

kappa_obs = 1
n_cons_obs = 4
n_v = 2
# A_obs_np = np.random.rand(n_cons_obs, n_v)
# b_obs_np = np.random.rand(n_cons_obs)
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

drone_ori_np = 0
drone_pos_np = np.array([3,3], dtype=config.np_dtype)
D_np = np.diag([1,1]).astype(config.np_dtype)
ellipse_coef_sqrt_np = np.linalg.cholesky(D_np)
R_b_to_w_np = np.array([[np.cos(drone_ori_np), -np.sin(drone_ori_np)],
                            [np.sin(drone_ori_np), np.cos(drone_ori_np)]])
ellipse_Q_sqrt_np = ellipse_coef_sqrt_np @ R_b_to_w_np.T
ellipse_Q_np = ellipse_Q_sqrt_np.T @ ellipse_Q_sqrt_np
_ellipse_Q_sqrt.value = ellipse_Q_sqrt_np
_ellipse_b.value = -2 * ellipse_Q_np @ drone_pos_np
_ellipse_c.value = drone_pos_np.T @ ellipse_Q_np @ drone_pos_np

A_obs_np = np.array([[1,0],
                     [0,-1],
                     [-1,0],
                     [0,1]], dtype=config.np_dtype)
b_obs_np = np.array([-1,-1,-1,-1], dtype=config.np_dtype)
_A_obs.value = A_obs_np
_b_obs.value = b_obs_np

N = 100
drone_pos = np.random.rand(N,2) + np.array([10,10], dtype=config.np_dtype)

def cvxpy_solve(solver):
    for i in range(N):
        drone_pos_np = drone_pos[i,:]
        _ellipse_Q_sqrt.value = ellipse_Q_sqrt_np
        _ellipse_b.value = -2 * ellipse_Q_np @ drone_pos_np
        _ellipse_c.value = drone_pos_np.T @ ellipse_Q_np @ drone_pos_np
        problem.solve(warm_start=True, solver=solver)
    time_cvxpy_end = time.time()

repeat = 100
print("Average time with SCS:", timeit.timeit("cvxpy_solve(cp.SCS)", globals=globals(), number=repeat)/N/repeat)
print("Average time with ECOS:", timeit.timeit("cvxpy_solve(cp.ECOS)", globals=globals(), number=repeat)/N/repeat)