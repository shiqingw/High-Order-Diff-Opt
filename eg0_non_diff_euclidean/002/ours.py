import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
import numpy as np
import scalingFunctionsHelperPy as sfh
import HOCBFHelperPy as hh
import cvxpy as cp
import matplotlib.pyplot as plt
import scipy.sparse as sparse
import pickle
import os

from cores.utils.utils import save_dict, load_dict
from cores.utils.osqp_utils import init_osqp


_A =  np.array([
        [ 1,  0],
        [-1,  0],
        [ 0,  1],
        [ 0, -1],
    ])
_b = np.array([-1, -1, -1, -1])
kappa = 10.0


r = 1.0
a = 2.0
x0, y0 = 0.0, 0.0
dt = 0.01
omega = 0.5
N = 1280
trajector_r = 2.3
Kp = 2.0
Kd = 2.0
t_list = np.arange(0, N * dt, dt)
state_list = np.zeros((N+1, 4))
state_list[0,:] = np.array([2.3, 0.0, 0.0, 0.0])
h_list = np.zeros(N)
u_list = np.zeros((N, 2))

cbf_qp = init_osqp(n_v=2, n_in=1)
gamma1 = 1.0
gamma2 = 1.0

SF_rob = sfh.Ellipsoid2d(True, np.eye(2), np.zeros(2))
SF_obs = sfh.LogSumExp2d(False, _A, _b, kappa)
probs = hh.Problem2dCollection(1)
vertices = np.array([[1.0, 1.0],
                     [-1.0, 1.0],
                     [-1.0, -1.0],
                     [1.0, -1.0]])
prob = hh.EllipsoidAndLogSumExp2dPrb(SF_rob, SF_obs, vertices)
probs.addProblem(prob, 0)

for i in range(0, N):
    x = state_list[i, :]
    p = x[0:2]
    v = x[2:4]
    t = t_list[i]
    xd = trajector_r * np.cos(omega * t)
    yd = trajector_r * np.sin(omega * t)
    xd_dt = -trajector_r * omega * np.sin(omega * t)
    yd_dt = trajector_r * omega * np.cos(omega * t)
    xd_dtdt = -trajector_r * omega**2 * np.cos(omega * t)
    yd_dtdt = -trajector_r * omega**2 * np.sin(omega * t)
    e = np.array([p[0] - xd, p[1] - yd])
    e_dt = np.array([v[0] - xd_dt, v[1] - yd_dt])
    u_nominal = np.array([xd_dtdt, yd_dtdt]) - Kp * e - Kd * e_dt

    # CBF
    all_P_np = np.zeros([1, 2])
    all_theta_np = np.zeros([1])
    all_dx = np.zeros([1, 3])

    all_P_np[0] = p.copy()
    all_theta_np[0] = 0
    all_dx[0,:] = np.array([v[0], v[1], 0])
    alpha0 = 1.0
    compensation = 0.0

    all_h_np, all_h_dx, all_h_dxdx, all_phi1_np, all_actuation_np, all_lb_np, all_ub_np = \
        probs.getCBFConstraints(all_P_np, all_theta_np, all_dx, alpha0, gamma1, gamma2, compensation)
    
    h = all_h_np[0]
    h_dx = all_h_dx[0, 0:2]
    h_dxdx = all_h_dxdx[0, 0:2, 0:2]

    h_dot = np.dot(h_dx, v)
    phi_1 = h + gamma1 * h_dot
    C = np.zeros((1, 2))
    C[0, :] = h_dx
    lb = np.zeros(1)
    lb[0] = - v @ h_dxdx @ v - gamma1 * h_dot - gamma2 * phi_1
    
    data = C.flatten()
    rows, cols = np.indices(C.shape)
    row_indices = rows.flatten()
    col_indices = cols.flatten()
    Ax = sparse.csc_matrix((data, (row_indices, col_indices)), shape=C.shape)
    g = -u_nominal
    ub = np.array([np.inf])
    cbf_qp.update(q=g, l=lb, u=ub, Ax=Ax.data)
    results = cbf_qp.solve()
    u = results.x
    # u = u_nominal

    state_list[i+1, 0:2] = p + dt * v
    state_list[i+1, 2:4] = v + dt * u
    u_list[i, :] = u
    h_list[i] = h


summary = {"times": t_list,
            "states": state_list,
            "controls": u_list}
exp_num = 2
results_dir = "{}/eg0_results/{:03d}".format(str(Path(__file__).parent.parent.parent), exp_num)
save_dict(summary, os.path.join(results_dir, 'our_summary.pkl'))

plt.figure()
plt.plot(state_list[:, 0], state_list[:, 1], label='Trajectory')
plt.show()

# plt.figure()
# plt.plot(t_list, state_list[1:,0:2], label='Trajectory')
# plt.show()

plt.figure()
plt.plot(t_list, u_list, label='Control')
plt.show()

plt.figure()
plt.plot(t_list, h_list, label='CBF')
plt.axhline(0, color='r', linestyle='--')
plt.show()