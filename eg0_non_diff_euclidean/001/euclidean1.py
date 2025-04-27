import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt

n_dim = 2
n_edges = 4
ball_center = np.array([-3, 0])
ball_radius = 1
_p_1 = cp.Variable(n_dim)
_p_2 = cp.Variable(n_dim)
_A = cp.Parameter((n_edges, n_dim))
_b = cp.Parameter(n_edges)

obj = cp.Minimize(cp.norm(_p_1 - _p_2, 2))
# _p_1 belongs to a ball and _p_2 belongs to a box
cons = [
    cp.sum_squares(_p_1 - ball_center) <= ball_radius ** 2,
    _A @ _p_2 + _b <= 0,
]
problem = cp.Problem(obj, cons)
assert problem.is_dpp()
assert problem.is_dcp(dpp = True)

thetas = np.linspace(0, 2 * np.pi, 1000)
min_dist = np.zeros_like(thetas)
for i, theta in enumerate(thetas):
    # rotation matrix
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta),  np.cos(theta)]])

    # the four axis‐aligned normals
    normals = np.array([
        [ 1,  0],
        [-1,  0],
        [ 0,  1],
        [ 0, -1],
    ])

    # rotate normals and form A, b
    A_val = normals @ R.T        # each row is (R(θ) n_i)^T
    b_val = -np.ones(n_edges)    # enforces (rotated normal)·p2 ≤ 1

    # assign to parameters
    _A.value = A_val
    _b.value = b_val

    # solve
    min_dist[i] = problem.solve(solver=cp.SCS, verbose=False)


# plot the minimum distance wrt theta
fig, ax = plt.subplots()
ax.plot(thetas, min_dist)
ax.set_xlabel(r"$\theta$")
ax.set_ylabel("Minimum distance")
plt.show()

# plot the derivative of the minimum distance wrt center_y
fig, ax = plt.subplots()
ax.plot(thetas, np.gradient(min_dist, thetas))
ax.set_xlabel(r"$y$")
ax.set_ylabel("Derivative of minimum distance")
plt.show()