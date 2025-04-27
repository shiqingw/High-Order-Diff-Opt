import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
import numpy as np
import math
import matplotlib.pyplot as plt
import scipy.sparse as sparse
import pickle
import os

from cores.utils.utils import save_dict, load_dict
from cores.utils.osqp_utils import init_osqp

def distance_ball_square(x, y, r, x0, y0, a):
    """
    Shortest distance between a circle of radius r at (x,y)
    and an axis-aligned square of side d centered at (x0,y0).
    Overlap → returns 0.
    """
    half = a / 2.0
    # signed offsets from square center
    tx, ty = x - x0, y - y0
    # how far outside the square along each axis
    dx = max(abs(tx) - half, 0.0)
    dy = max(abs(ty) - half, 0.0)
    dist_center = math.hypot(dx, dy)
    return max(dist_center - r, 0.0)

def grad_distance_ball_square(x, y, r, x0, y0, a):
    """
    Gradient of the shortest-distance function w.r.t. (x, y).
    Returns (∂d/∂x, ∂d/∂y). Zeros if circle overlaps the square.
    """
    half = a / 2.0
    tx, ty = x - x0, y - y0
    adx, ady = abs(tx), abs(ty)
    dx = max(adx - half, 0.0)
    dy = max(ady - half, 0.0)
    dist_center = math.hypot(dx, dy)

    # If inside (dist_center ≤ r) or exactly inside square (dist_center=0), gradient = 0
    if dist_center <= r or dist_center == 0.0:
        return 0.0, 0.0

    # derivative of dx w.r.t x is sign(tx) when |tx|>half, else 0
    pdx = math.copysign(1.0, tx) if adx > half else 0.0
    # derivative of dy w.r.t y is sign(ty) when |ty|>half, else 0
    pdy = math.copysign(1.0, ty) if ady > half else 0.0

    # ∇√(dx²+dy²) = (dx·∂dx + dy·∂dy) / √(dx²+dy²)
    grad_x = (dx * pdx) / dist_center
    grad_y = (dy * pdy) / dist_center

    # since d = dist_center - r, ∇d = ∇dist_center
    return np.array([grad_x, grad_y], dtype=np.float64)

def hessian_distance_ball_square(x, y, r, x0, y0, a):
    """
    Hessian (2x2) of the shortest-distance function between:
      • circle of radius r at (x,y)
      • axis-aligned square of side d at (x0,y0)
    Returns a numpy array H = [[d²d/dx², d²d/dxdy],
                               [d²d/dydx, d²d/dy²]].
    """
    half = a / 2.0
    # offsets and “outside‐square” distances
    u, v = x - x0, y - y0
    ax, ay = abs(u) - half, abs(v) - half
    dx = ax if ax > 0.0 else 0.0
    dy = ay if ay > 0.0 else 0.0

    D = math.hypot(dx, dy)
    # zero‐Hessian whenever inside overlap (D ≤ r), at D=0, or on edges (dx=0 or dy=0)
    if D <= r or D == 0.0 or dx == 0.0 or dy == 0.0:
        return np.zeros((2, 2))

    # signs for the “corner” derivatives
    sgn_u = math.copysign(1.0, u)
    sgn_v = math.copysign(1.0, v)

    # build the Hessian of D
    H = np.zeros((2, 2))
    H[0, 0] =  1.0 / D - (dx**2) / (D**3)
    H[1, 1] =  1.0 / D - (dy**2) / (D**3)
    off = - (dx * dy * sgn_u * sgn_v) / (D**3)
    H[0, 1] = off
    H[1, 0] = off

    # since d = D - r in this region, ∇²d = ∇²D
    return H


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
    h = distance_ball_square(p[0], p[1], r, x0, y0, a)
    h_dx = grad_distance_ball_square(p[0], p[1], r, x0, y0, a)
    h_dxdx = hessian_distance_ball_square(p[0], p[1], r, x0, y0, a)

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
save_dict(summary, os.path.join(results_dir, 'euclidean_summary.pkl'))

plt.figure()
plt.plot(state_list[:, 0], state_list[:, 1], label='Trajectory')
plt.show()

# # plt.figure()
# # plt.plot(t_list, state_list[1:,0:2], label='Trajectory')
# # plt.show()

plt.figure()
plt.plot(t_list, u_list, label='Control')
plt.show()

# plt.figure()
# plt.plot(t_list, h_list, label='CBF')
# plt.axhline(0, color='r', linestyle='--')
# plt.show()

