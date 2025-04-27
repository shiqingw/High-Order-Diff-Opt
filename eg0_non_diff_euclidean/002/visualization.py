import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
import numpy as np
import pickle
import os

from cores.utils.utils import save_dict, load_dict

# euclidean data
exp_num = 2
results_dir = "{}/eg0_results/{:03d}".format(str(Path(__file__).parent.parent.parent), exp_num)
euclidean_summary = load_dict(os.path.join(results_dir, 'euclidean_summary.pkl'))

# our data
our_summary = load_dict(os.path.join(results_dir, 'our_summary.pkl'))

# plot
plt.rcParams['font.family'] = 'serif'
plt.rcParams.update({"text.usetex": True,
                    "text.latex.preamble": r"\usepackage{amsmath}"})
plt.rcParams.update({'pdf.fonttype': 42})
fontsize = 50
ticksize = 50
linewidth = 4


#####################Scenario#####################
fig = plt.figure(figsize=(8, 8), dpi=50)
ax = fig.add_subplot(111)

# obstacle
square = Rectangle(
    (-1,-1),
    2,
    2,
    facecolor="#8CE99A",
    edgecolor="#2F9E44",
    linewidth=linewidth
)
ax.add_patch(square)

# scaling function
kappa = 10.0
n_ineq = 4
aprox_exp = lambda x: np.exp(kappa*x)
aprox_square = lambda x,y : 1/kappa * np.log((aprox_exp(x-1) + aprox_exp(-x-1) + aprox_exp(-y-1) + aprox_exp(y-1))/n_ineq) + 1.0

xmin, xmax = -2, 2
ymin, ymax = -2, 2
n = 1000
x = np.linspace(xmin, xmax, n)
y = np.linspace(ymin, ymax, n)
X, Y = np.meshgrid(x, y)
values = aprox_square(X, Y)

# Boundary contour at level = 1.0
c  = ax.contour(
    X, Y, values,
    levels=[1.0],
    colors=['#2F9E44'],
    linewidths=[linewidth],
    linestyles = ["dashed"]
)

# desired trajectory
circle = plt.Circle((0.0, 0.0), 2.3, edgecolor="gray", fill=False, linewidth=linewidth, linestyle="dashed")
ax.add_artist(circle)

# # euclidean trajectory
# euclidean_state = euclidean_summary["states"]
# ax.plot(euclidean_state[:, 0], euclidean_state[:, 1], color="black", linewidth=linewidth, linestyle="solid", zorder=1)

# our trajectory
our_state = our_summary["states"]
ax.plot(our_state[:, 0], our_state[:, 1], color="black", linewidth=linewidth, linestyle="solid", zorder=1)

# robot
circle = plt.Circle((2.3, 0.0), 1, facecolor="#74C0FC", edgecolor="#1971C2", fill=True, linewidth=linewidth, linestyle="solid")
ax.add_artist(circle)

ax.set_xlim(-3.4, 3.4)
ax.set_ylim(-3.4, 3.4)
ax.set_aspect('equal', adjustable='box')
ax.set_xlabel(r"$x$", fontsize=fontsize)
ax.set_ylabel(r"$y$", fontsize=fontsize)
ax.tick_params(axis='both', which='major', labelsize=ticksize, grid_linewidth=10)
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'scenario.pdf'), dpi=50)
plt.close()

#####################Control#####################

fig = plt.figure(figsize=(8, 8), dpi=50)
ax = fig.add_subplot(211)
# euclidean trajectory
euclidean_controls = euclidean_summary["controls"]
eulcidean_time = euclidean_summary["times"]
ax.plot(eulcidean_time, euclidean_controls[:, 0], color="tab:orange", linewidth=linewidth, linestyle="solid")
ax.plot(eulcidean_time, euclidean_controls[:, 1], color="tab:blue", linewidth=linewidth, linestyle="solid")

# discontinuities
euclidean_controls_x_diff = np.diff(euclidean_controls[:, 0])
discontinuities_x = np.where(np.abs(euclidean_controls_x_diff) > 1e-1)[0]
ax.scatter(eulcidean_time[discontinuities_x], euclidean_controls[discontinuities_x, 0], 
           facecolors='none', color="red", s=100, marker="o", linewidth=linewidth, zorder=10)
euclidean_controls_y_diff = np.diff(euclidean_controls[:, 1])
discontinuities_y = np.where(np.abs(euclidean_controls_y_diff) > 1e-1)[0]
ax.scatter(eulcidean_time[discontinuities_y], euclidean_controls[discontinuities_y, 1], 
           facecolors='none', color="red", s=100, marker="o", linewidth=linewidth, zorder=10)

ax.set_xlim(0.0, eulcidean_time[-1])
ax.set_ylabel("Euclidean", fontsize=fontsize)
ax.tick_params(axis='both', which='major', labelsize=ticksize, grid_linewidth=10)

# our trajectory
ax = fig.add_subplot(212)
our_controls = our_summary["controls"]
our_time = our_summary["times"]
ax.plot(our_time, our_controls[:, 0], color="tab:orange", linewidth=linewidth, linestyle="dashed")
ax.plot(our_time, our_controls[:, 1], color="tab:blue", linewidth=linewidth, linestyle="dashed")

ax.set_xlim(0.0, our_time[-1])
ax.set_xlabel("Time [s]", fontsize=fontsize)
ax.set_ylabel("Ours", fontsize=fontsize)
ax.tick_params(axis='both', which='major', labelsize=ticksize, grid_linewidth=10)
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'controls.pdf'), dpi=50)
plt.close()