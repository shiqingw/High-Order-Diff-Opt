import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

plt.rcParams['font.family'] = 'serif'
plt.rcParams.update({"text.usetex": True,
                    "text.latex.preamble": r"\usepackage{amsmath}"})
plt.rcParams.update({'pdf.fonttype': 42})
fontsize = 50
ticksize = 50

#####################################################################################

# # Draw a ball with a radius of 1 centered at (-3, 3)
fig = plt.figure(figsize=(10, 8), dpi=50)
ax = fig.add_subplot(111)
linewidth = 4
circle = plt.Circle((0, -2.5), 1, facecolor="#74C0FC", edgecolor="#1971C2", fill=False, linewidth=linewidth, linestyle="dashed")
ax.add_artist(circle)

circle = plt.Circle((0, 2.5), 1, facecolor="#74C0FC", edgecolor="#1971C2", fill=True, linewidth=linewidth, linestyle="solid")
ax.add_artist(circle)

# Draw an arrow pointing from (-3, 0) to (0, 0)
ax.annotate("", xy=(0, 1), xytext=(0, -1),
            arrowprops=dict(arrowstyle="simple", lw=2, color="#1971C2"),
            fontsize=fontsize)

# Draw a box with corners at (-1, -1), (-1, 1), (1, 1), and (1, -1)
square = Rectangle(
    (3,-1),
    2,
    2,
    facecolor="#8CE99A",
    edgecolor="#2F9E44",
    linewidth=linewidth
)
ax.add_patch(square)


ax.set_xlim(-3, 7)
ax.set_ylim(-4, 4)
ax.set_aspect('equal', adjustable='box')
ax.set_xlabel(r"$x$", fontsize=fontsize)
ax.set_ylabel(r"$y$", fontsize=fontsize)
ax.tick_params(axis='both', which='major', labelsize=ticksize, grid_linewidth=10)
plt.tight_layout()
plt.savefig("eg0_results/euclidean_scenario.pdf", dpi=300)
plt.close()

#####################################################################################

fig = plt.figure(figsize=(10, 8), dpi=50)
ax = fig.add_subplot(111)
linewidth = 4

circle = plt.Circle((0, -2.5), 1, facecolor="#74C0FC", edgecolor="#1971C2", fill=False, linewidth=linewidth, linestyle="dashed")
ax.add_artist(circle)

circle = plt.Circle((0, 2.5), 1, facecolor="#74C0FC", edgecolor="#1971C2", fill=True, linewidth=linewidth, linestyle="solid")
ax.add_artist(circle)

# Draw an arrow pointing from (-3, 0) to (0, 0)
ax.annotate("", xy=(0, 1), xytext=(0, -1),
            arrowprops=dict(arrowstyle="simple", lw=2, color="#1971C2"),
            fontsize=fontsize)

# Draw scaling functions
kappa = 5.0
n_ineq = 4
aprox_exp = lambda x: np.exp(kappa*x)
aprox_square = lambda x,y : 1/kappa * np.log((aprox_exp(-x+3) + aprox_exp(x-5) + aprox_exp(-y-1) + aprox_exp(y-1))/n_ineq) + 1.0

xmin, xmax = 2, 6
ymin, ymax = -2, 2
n = 1000
x = np.linspace(xmin, xmax, n)
y = np.linspace(ymin, ymax, n)
X, Y = np.meshgrid(x, y)
values = aprox_square(X, Y)

cf = ax.contourf(
    X, Y, values,
    levels=[values.min(), 1.0],
    colors=['#8CE99A'],
    alpha=1.0
)

# Boundary contour at level = 1.0
c  = ax.contour(
    X, Y, values,
    levels=[1.0],
    colors=['#2F9E44'],
    linewidths=[linewidth]
)

# Draw a box with corners at (-1, -1), (-1, 1), (1, 1), and (1, -1)
square = Rectangle(
    (3,-1),
    2,
    2,
    fill=False,
    edgecolor="black",
    linewidth=linewidth,
    linestyle="dashed"
)
ax.add_patch(square)

ax.set_xlim(-3, 7)
ax.set_ylim(-4, 4)
ax.set_aspect('equal', adjustable='box')
ax.set_xlabel(r"$x$", fontsize=fontsize)
ax.set_ylabel(r"$y$", fontsize=fontsize)
ax.tick_params(axis='both', which='major', labelsize=ticksize, grid_linewidth=10)
plt.tight_layout()
# plt.show()
plt.savefig("eg0_results/ours_scenario.pdf", dpi=300)
# plt.close()