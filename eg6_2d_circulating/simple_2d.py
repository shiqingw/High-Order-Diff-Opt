import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt
import numpy as np
from cores.utils.proxsuite_utils import init_proxsuite_qp

circle_center = np.array([0.0, 2.0])
diag_matrix = np.diag([1.0, 1.0])

dt = 0.01
T = 100.0
times = np.arange(0, T, dt)
states = np.zeros((len(times)+1, 4))
controls = np.zeros((len(times), 2))
states[0,0:2] = np.array([0.01, 0.0])

cbf_qp = init_proxsuite_qp(2, 0, 2)
gamma1 = 1.0
gamma2 = 1.0
target_point = np.array([0.0, 4.0])

for i in range(0,len(times)):
    p = states[i,0:2]
    v = states[i,2:4]
    
    h = np.dot(p - circle_center, np.dot(diag_matrix, p - circle_center)) - 1.2
    h_dp = 2.0 * diag_matrix @ (p - circle_center)
    h_dpdp = 2.0 * diag_matrix

    h_dt = h_dp @ v
    phi1 = h_dt + gamma1 * h

    e = p - target_point
    e_dt = v
    u_nominal = - 10*e - 20*e_dt

    C = np.zeros((2,2))
    lb = np.zeros(2)
    ub = np.zeros(2)

    C[0,:] = h_dp
    lb[0] = - v @ h_dpdp @ v - gamma1 * h_dt - gamma2 * phi1
    ub[0] = np.inf

    # Omega = np.array([[0.0, -1.0],
    #                   [1.0, 0.0]])
    Omega = np.array([[0.0, 1.0],
                      [-1.0, 0.0]])
    C[1,:] = Omega @ h_dp
    lb[1] = 1.0- 10.0*h
    ub[1] = np.inf

    H = np.eye(2)
    g = - u_nominal

    cbf_qp.update(H=H, g=g, C=C, l=lb, u=ub)
    cbf_qp.solve()
    u = cbf_qp.results.x

    states[i+1,0:2] = p + dt * v
    states[i+1,2:4] = v + dt * u

# plot the circle
theta = np.linspace(0, 2*np.pi, 100)
x = circle_center[0] + np.cos(theta)
y = circle_center[1] + np.sin(theta)
plt.plot(x, y)
plt.plot(states[:,0], states[:,1])
plt.axis('equal')
plt.show()

