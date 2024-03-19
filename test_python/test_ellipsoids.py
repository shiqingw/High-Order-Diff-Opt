import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from cores.rimon_method_python.rimon_method import rimon_method

def create_ellipsoidal_points(center, coefficients):
    u = np.linspace(0, 2 * np.pi, 200)
    v = np.linspace(0, np.pi, 200)
    x = 1/np.sqrt(coefficients[0][0]) * np.outer(np.cos(u), np.sin(v)) + center[0]
    y = 1/np.sqrt(coefficients[1][1]) * np.outer(np.sin(u), np.sin(v)) + center[1]
    z = 1/np.sqrt(coefficients[2][2]) * np.outer(np.ones(np.size(u)), np.cos(v)) + center[2]
    return x, y, z


# Center and coefficients of the ellipsoid
a = np.array([0, 0, 0])
A = np.array([[1, 0, 0],
              [0, 2, 0],
              [0, 0, 3]])

b = np.array([3, 3, 3])
B = np.array([[1, 0, 0],
              [0, 2, 0],
              [0, 0, 1]])

# Plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x, y, z = create_ellipsoidal_points(a, A)
ax.plot_surface(x, y, z, rstride=4, cstride=4, color='b', edgecolors=None, alpha=0.5)

x, y, z = create_ellipsoidal_points(b, B)
ax.plot_surface(x, y, z, rstride=4, cstride=4, color='r', edgecolors=None, alpha=0.5)

# Rimon method
x_rimon = rimon_method(A, a, B, b)
ax.scatter(x_rimon[0], x_rimon[1], x_rimon[2], color='g', s=100)

# # Scaling
alpha = (x_rimon-a) @ A @ (x_rimon-a)
x, y, z = create_ellipsoidal_points(a, A/alpha)
ax.plot_surface(x, y, z, rstride=4, cstride=4, color='g', edgecolors=None, alpha=0.3)


# Axis labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

ax.axis('equal')

plt.show()
