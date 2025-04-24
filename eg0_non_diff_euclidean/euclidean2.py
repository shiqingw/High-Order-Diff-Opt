import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt

n_dim = 2
n_edges = 4
_p_1 = cp.Variable(n_dim)
_p_2 = cp.Variable(n_dim)
ball_center = cp.Parameter(n_dim)
ball_radius = 1
_A =  np.array([
        [ 1,  0],
        [-1,  0],
        [ 0,  1],
        [ 0, -1],
    ])
_b = np.array([-5, 3, -1, -1])

obj = cp.Minimize(cp.norm(_p_1 - _p_2, 2))
# _p_1 belongs to a ball and _p_2 belongs to a box
cons = [
    cp.sum_squares(_p_1 - ball_center) <= ball_radius ** 2,
    _A @ _p_2 + _b <= 0,
]
problem = cp.Problem(obj, cons)
assert problem.is_dpp()
assert problem.is_dcp(dpp = True)

center_x = 0.0
center_y_np = np.linspace(-2.5, 2.5, 1000)
min_dist = np.zeros_like(center_y_np)
for i, center_y in enumerate(center_y_np):
    ball_center.value = np.array([center_x, center_y])
    min_dist[i] = problem.solve(solver=cp.ECOS, verbose=False)


plt.rcParams['font.family'] = 'serif'
plt.rcParams.update({"text.usetex": True,
                    "text.latex.preamble": r"\usepackage{amsmath}"})
plt.rcParams.update({'pdf.fonttype': 42})
fontsize = 50
ticksize = 50

fig = plt.figure(figsize=(10, 8), dpi=50)
ax = fig.add_subplot(211)
ax.plot(center_y_np, min_dist, linewidth=4, color="tab:blue")
ax.tick_params(axis='both', which='major', labelsize=ticksize, grid_linewidth=10)
# ax.set_xlabel(r"$y_c$", fontsize=fontsize)

ax = fig.add_subplot(212)
ax.plot(center_y_np[2:], np.gradient(min_dist, center_y_np)[2:], linewidth=4, color="tab:orange")
ax.tick_params(axis='both', which='major', labelsize=ticksize, grid_linewidth=10)
ax.set_xlabel(r"$y_c$", fontsize=fontsize)

plt.tight_layout()
plt.savefig("eg0_results/euclidean.pdf", dpi=300)
plt.show()