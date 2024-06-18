import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import os

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
from cores.utils.proxsuite_utils import init_proxsuite_qp
import diffOptHelper2 as doh


plt.rcParams['font.family'] = 'serif'
plt.rcParams.update({"text.usetex": True,
                        "text.latex.preamble": r"\usepackage{amsmath}"})
plt.rcParams.update({'pdf.fonttype': 42})

obstacle_center = np.array([0.0, 0.0])
obstacle_r = 1
obstacle_matrix = np.diag([1.0/obstacle_r**2, 1.0/obstacle_r**2])
obs_SF = doh.Ellipsoid2d(False, obstacle_matrix, np.zeros(2))

robot_r = 0.5
robot_D = np.diag([1.0/robot_r**2, 1.0/robot_r**2])
robot_SF = doh.Ellipsoid2d(True, robot_D, np.zeros(2))

cbf_qp = init_proxsuite_qp(2, 0, 2)


alpha0 = 1.1
gamma1 = 1.0
gamma2 = 1.0

# fig, ax = plt.subplots(figsize=(5,5), frameon=True)
fig, ax = plt.subplots(frameon=True)

obstacle = Ellipse(obstacle_center, 2*obstacle_r, 2*obstacle_r, edgecolor='#1C7ED6', facecolor='#74C0FC', linewidth=2)
ax.add_patch(obstacle)

h_desired = 0
radius = robot_r*np.sqrt(h_desired + alpha0) + obstacle_r + 1
theta = 0
pos = radius*np.array([np.cos(theta), np.sin(theta)])
p_rimon = doh.rimonMethod2d(robot_SF, pos, 0, obs_SF, obstacle_center, 0)
alpha, alpha_dx, alpha_dxdx = doh.getGradientAndHessian2d(p_rimon, robot_SF, pos, 0, obs_SF, obstacle_center, 0)
h = alpha - alpha0
h_dpdp = alpha_dxdx[:2,:2]
h_dp = alpha_dx[:2]
v = np.zeros(2)
a = alpha_dx[:2]
c = np.array([a[1], -a[0]])
d = 1-h
# u_nominal = - c
u_nominal = np.array([-10, 0])

h_dt = h_dp @ v
phi1 = h_dt + gamma1 * h

H = np.eye(2)
g = - u_nominal
C = np.zeros((2,2))
lb = np.zeros(2)
ub = np.zeros(2)

C[0,:] = a
lb[0] = - v @ h_dpdp @ v - gamma1 * h_dt - gamma2 * phi1
ub[0] = np.inf
C[1,:] = c
lb[1] = d
ub[1] = np.inf

cbf_qp.update(H=H, g=g, C=C, l=lb, u=ub)
cbf_qp.solve()
u = cbf_qp.results.x
print(u)

ax.quiver(pos[0], pos[1], a[0], a[1], color='#2F9E44', scale=10, scale_units='inches', width=0.005, zorder=0.1)
ax.quiver(pos[0], pos[1], c[0], c[1], color='#E03131', scale=10, scale_units='inches', width=0.005, zorder=0.1)
ax.quiver(pos[0], pos[1], u_nominal[0], u_nominal[1], color='#343A40', scale=10, scale_units='inches', width=0.005, zorder=0.1)
ax.quiver(pos[0], pos[1], u[0], u[1], color='#F76707', scale=1, scale_units='inches', width=0.005, zorder=0.1)

robot = plt.Circle(pos, robot_r, edgecolor='#F76707', facecolor="None", linestyle='--', lw=1)
ax.add_patch(robot)

ax.axis('equal')
plt.show()