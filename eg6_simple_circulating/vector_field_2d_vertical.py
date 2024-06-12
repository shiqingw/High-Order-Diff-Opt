import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import os

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
from cores.utils.proxsuite_utils import init_proxsuite_qp
import diffOptHelper as doh
from cores.configuration.configuration import Configuration

config = Configuration()

circulation = False

obstacle_center = np.array([0.0, -0.8])
obstacle_a = 2
obstacle_b = 1.5
obstacle_matrix = np.diag([1.0/obstacle_a**2, 1.0/obstacle_b**2])
obstacle_func = lambda x: (x - obstacle_center) @ obstacle_matrix @ (x - obstacle_center) - 1.0

robot_r = 0.5
robot_D = np.diag([1.0/robot_r**2, 1.0/robot_r**2])

target_point = np.array([0.0, 5.0])

cbf_qp = init_proxsuite_qp(2, 0, 2)
gamma1 = 1.0
gamma2 = 1.0
offset = 1.0
inclination = 1.0

dt = 0.01
t = 0
kp = 20
kd = 40
times = []
state = np.array([0.0, -5.0, 0.0, 0.0])
states = []

while np.linalg.norm(state[:2] - target_point) >= 1e-2 and t < 100:
    states.append(np.copy(state))
    times.append(t)

    p = state[:2]
    v = state[2:4]

    theta = 0
    R = np.eye(2)
    alpha, _, grad, hess = doh.getGradientAndHessianEllipses(p, theta, robot_D, R, obstacle_matrix, obstacle_center)
    
    h = alpha - 1.2
    h_dp = grad[1:3]
    h_dpdp = hess[1:3,1:3]

    h_dt = h_dp @ v
    phi1 = h_dt + gamma1 * h

    e = p - target_point
    e_dt = v
    u_nominal = - kp*e - kd*e_dt

    C = np.zeros((2,2))
    lb = np.zeros(2)
    ub = np.zeros(2)

    C[0,:] = h_dp
    lb[0] = - v @ h_dpdp @ v - gamma1 * h_dt - gamma2 * phi1
    ub[0] = np.inf

    if circulation:
        Omega = np.array([[0.0, -1.0],
                        [1.0, 0.0]])
        C[1,:] = Omega @ h_dp
        lb[1] = offset - inclination*h
        ub[1] = np.inf

    H = np.eye(2)
    g = - u_nominal

    cbf_qp.update(H=H, g=g, C=C, l=lb, u=ub)
    cbf_qp.solve()
    u = cbf_qp.results.x

    state[0] = state[0] + dt * state[2]
    state[1] = state[1] + dt * state[3]
    state[2] = state[2] + dt * u[0]
    state[3] = state[3] + dt * u[1]

    t += dt

x_min, x_max = -5, 5
y_min, y_max = -10, 10
spacing = 0.5
x = np.arange(x_min, x_max, spacing)
y = np.arange(y_min, y_max, spacing)
x, y = np.meshgrid(x, y) 

x_flatten = x.flatten()
y_flatten = y.flatten()

a_x = np.zeros_like(x_flatten)
a_y = np.zeros_like(x_flatten)

for i in range(len(x_flatten)):
    p = np.array([x_flatten[i], y_flatten[i]])

    if obstacle_func(p) <= 0:
        continue

    v = np.array([0.0, 0.0])
    theta = 0
    R = np.eye(2)
    alpha, _, grad, hess = doh.getGradientAndHessianEllipses(p, theta, robot_D, R, obstacle_matrix, obstacle_center)
    
    h = alpha - 1.2

    if h < 0:
        continue

    h_dp = grad[1:3]
    h_dpdp = hess[1:3,1:3]

    h_dt = h_dp @ v
    phi1 = h_dt + gamma1 * h

    e = p - target_point
    e_dt = v
    u_nominal = - kp*e - kd*e_dt

    C = np.zeros((2,2))
    lb = np.zeros(2)
    ub = np.zeros(2)

    C[0,:] = h_dp
    lb[0] = - v @ h_dpdp @ v - gamma1 * h_dt - gamma2 * phi1
    ub[0] = np.inf

    if circulation:
        Omega = np.array([[0.0, -1.0],
                        [1.0, 0.0]])
        C[1,:] = Omega @ h_dp
        lb[1] = offset - inclination*h
        ub[1] = np.inf

    H = np.eye(2)
    g = - u_nominal

    cbf_qp.update(H=H, g=g, C=C, l=lb, u=ub)
    cbf_qp.solve()
    u = cbf_qp.results.x

    a_x[i] = u[0]
    a_y[i] = u[1]

# post process the trajectory
states = np.array(states)

# post process the vector field
a_norm = np.sqrt(a_x**2 + a_y**2)
for i in range(len(a_norm)):
    if a_norm[i] > 1e-6:
        a_x[i] /= a_norm[i]
        a_y[i] /= a_norm[i]
a_x = a_x.reshape(x.shape)
a_y = a_y.reshape(y.shape)

# Draw plots
print("==> Draw plots")
plt.rcParams['font.family'] = 'serif'
plt.rcParams.update({"text.usetex": True,
                        "text.latex.preamble": r"\usepackage{amsmath}"})
plt.rcParams.update({'pdf.fonttype': 42})

label_fs = 20
tick_fs = 20
legend_fs = 30
linewidth = 5

fig, ax = plt.subplots(figsize=(5,6), dpi=config.dpi, frameon=True)
ax.quiver(x, y, a_x, a_y, color='#2F9E44', scale=6, scale_units='inches', width=0.005, zorder=0.1)
ax.axis('equal')

obstacle = Ellipse(obstacle_center, 2*obstacle_a, 2*obstacle_b, edgecolor='#1C7ED6', facecolor='#74C0FC', linewidth=2)
ax.add_patch(obstacle)

# target = plt.Circle(target_point, 0.1, color='r')
# ax.add_patch(target)

# start = plt.Circle(states[0,0:2], 0.1, color='b')
# ax.add_patch(start)

plt.scatter(states[0,0], states[0,1], s=40, marker="D", color='r', zorder=1.9)
plt.scatter(target_point[0], target_point[1], s=80, marker="*", color='r', zorder=1.9)

if circulation:
    plt.plot(states[:,0], states[:,1], color='#F76707', linestyle="--", lw=2, zorder=0.9)
    indices = [0]
    for i in range(1, len(states)):
        if np.linalg.norm(states[i,:2] - states[indices[-1],:2]) > 2*robot_r and i != len(states)-1:
            indices.append(i)
    indices.append(len(states)-1)

    for i in indices:
        state = states[i]
        robot = plt.Circle(state[:2], robot_r, edgecolor='#F76707', facecolor="None", linestyle='--', lw=1)
        ax.add_patch(robot)

else:
    plt.plot(states[:,0], states[:,1], color='#F76707', linestyle="--", lw=2, zorder=0.9)
    indices = [0]
    for i in range(1, len(states)):
        if np.linalg.norm(states[i,:2] - states[indices[-1],:2]) > 2*robot_r and i != len(states)-1:
            indices.append(i)
    indices.append(len(states)-1)

    for i in indices:
        state = states[i]
        robot = plt.Circle(state[:2], robot_r, edgecolor='#F76707', facecolor="None", linestyle='--', lw=1)
        ax.add_patch(robot)
# plt.legend()
# plt.legend(fontsize = legend_fs)
plt.xlabel('$x_1$', fontsize=label_fs)
plt.ylabel('$x_2$', fontsize=label_fs)
plt.xticks(fontsize = tick_fs)
plt.yticks(fontsize = tick_fs)
plt.xlim(-3, 3)
plt.ylim(-6, 6)
plt.tight_layout()
results_dir = "{}/eg6_results".format(str(Path(__file__).parent.parent))
if circulation:
    plt.savefig(os.path.join(results_dir, 'plot_x_y_w_circulation.pdf'))
else:
    plt.savefig(os.path.join(results_dir, 'plot_x_y_wo_circulation.pdf'))
plt.close(fig)

    


  

