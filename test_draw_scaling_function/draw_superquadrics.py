import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3D
from skimage import measure


plt.rcParams['font.family'] = 'serif'
plt.rcParams.update({"text.usetex": True,
                        "text.latex.preamble": r"\usepackage{amsmath}"})
plt.rcParams.update({'pdf.fonttype': 42})

e1 = 1.5
e2 = 0.5
a1 = 1.0
a2 = 1.0
a3 = 1.0
superq = lambda x, y, z: np.power(np.power(np.abs(x/a1), 2/e2) + np.power(np.abs(y/a2), 2/e2), e2/e1) + np.power(np.abs(z/a3), 2/e1)

n = 40
xmin, xmax = -1.5, 1.5
ymin, ymax = -1.5, 1.5
zmin, zmax = -1.5, 1.5
x = np.linspace(xmin, xmax, n)
y = np.linspace(ymin, ymax, n)
z = np.linspace(zmin, zmax, n)
X,Y,Z = np.meshgrid(x,y,z)
values = superq(X,Y,Z)

# Create a 3D plot
fig = plt.figure(figsize=(5, 5), dpi=100)
ax = fig.add_subplot(111, projection='3d')

# Plot the approximation
verts, faces, normals, values = measure.marching_cubes(values, level=1, spacing=((xmax - xmin) / (n - 1), (ymax - ymin) / (n - 1), (zmax - zmin) / (n - 1)))
verts[:, 0] = verts[:, 0] + xmin
verts[:, 1] = verts[:, 1] + ymin
verts[:, 2] = verts[:, 2] + zmin
ax.plot_trisurf(verts[:, 0], verts[:, 1], faces, verts[:, 2], color='cornflowerblue', alpha=1.0, lw=0, shade=True)


# Add labels and title
# label_fontsize = 20
# ax.set_xlabel('X', fontsize=label_fontsize)
# ax.set_ylabel('Y', fontsize=label_fontsize)
# ax.set_zlabel('Z', fontsize=label_fontsize)
# ax.xaxis.labelpad=30
# ax.yaxis.labelpad=30
# ax.zaxis.labelpad=30

# Set the tick size for each axis
tickfontsize = 20
ax.xaxis.set_tick_params(size=tickfontsize)
ax.yaxis.set_tick_params(size=tickfontsize)
ax.zaxis.set_tick_params(size=tickfontsize)
ax.set_xlim(-1.0, 1.0)
ax.set_ylim(-1.0, 1.0)
ax.set_zlim(-1.0, 1.0)

# Optionally, set the tick label size as well
ax.tick_params(labelsize=tickfontsize)

# Set the view angle (elevation and azimuth)
ax.view_init(elev=20, azim=30)  # Adjust these angles

# Show the plot
plt.gca().set_aspect('equal')
plt.tight_layout()
# plt.show()
plt.savefig("superquadrics_e1eq{}_e2eq{}.pdf".format(e1,e2))

