import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from skimage import measure

# Define the function f(x,y,z)
def f(x, y, z):
    return x**2 + y**2 + z**2

# Create a 3D grid of points
x = np.linspace(-2, 2, 100)
y = np.linspace(-2, 2, 100)
z = np.linspace(-2, 2, 100)
x, y, z = np.meshgrid(x, y, z)

# Evaluate the function on the grid
values = f(x, y, z)

# Find the isosurface where f(x,y,z) = 1
verts, faces, normals, values = measure.marching_cubes(values, level=1)

# Adjust the vertices to match the original coordinate system
verts[:, 0] = verts[:, 0] * (4 / 99) - 2
verts[:, 1] = verts[:, 1] * (4 / 99) - 2
verts[:, 2] = verts[:, 2] * (4 / 99) - 2

# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the surface
ax.plot_trisurf(verts[:, 0], verts[:, 1], faces, verts[:, 2], cmap='Spectral', lw=1)

# Set labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Show the plot
plt.show()
