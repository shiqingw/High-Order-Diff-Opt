import sys
import numpy as np
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import ConvexHull
from skimage import measure

from cores.obstacle_collections.polytope_collection import PolytopeCollection

dict_obs = {}
obs_1 = {'vertices': np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0],[0.5,0.5,1]])}
dict_obs['obs_1'] = obs_1
obs_2 = {'vertices': np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0],
                               [0,0,1],[1,0,1],[1,1,1],[0,1,1]])+np.array([2,2,0])}
dict_obs['obs_2'] = obs_2

polytope_collection = PolytopeCollection(3, len(dict_obs), dict_obs)

plt.rcParams['font.family'] = 'serif'
plt.rcParams.update({"text.usetex": True,
                        "text.latex.preamble": r"\usepackage{amsmath}"})
plt.rcParams.update({'pdf.fonttype': 42})

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

label_fs = 20
tick_fs = 20
legend_fs = 30
linewidth = 5

for (i, key) in enumerate(dict_obs.keys()):
    obs = dict_obs[key]
    vertices = np.array(obs["vertices"])
    hull = ConvexHull(vertices)

    # Plot the edges
    for simplex in hull.simplices:
        simplex = np.append(simplex, simplex[0])
        ax.plot(vertices[simplex, 0], vertices[simplex, 1], vertices[simplex, 2], 'k-')

    # Plot the vertices
    ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], c='r', marker='o')
    
    print(polytope_collection.face_equations[key])
    
    # Plot the scaling function
    kappa, n = 10, 60
    xmin, xmax = -0.5, 3.5
    ymin, ymax = -0.5, 3.5
    zmin, zmax = -0.5, 3.5
    x = np.linspace(xmin, xmax, n)
    y = np.linspace(ymin, ymax, n)
    z = np.linspace(zmin, zmax, n)
    X,Y,Z = np.meshgrid(x,y,z)
    X_flat = X.flatten()
    Y_flat = Y.flatten()
    Z_flat = Z.flatten()
    all_points = np.c_[X_flat, Y_flat, Z_flat] # shape (n_points, dim)
    A = polytope_collection.face_equations[key]["A"] # shape (n_faces, dim)
    b = polytope_collection.face_equations[key]["b"] # shape (n_faces,)
    values = all_points @ A.T + b # shape (n_points, n_faces)
    values = np.log(np.exp(kappa*values).sum(axis=1)) - np.log(len(b)) + 1.0 # shape (n_points,)
    values = values.reshape(X.shape)
    verts, faces, normals, values = measure.marching_cubes(values, level=1, spacing=((xmax - xmin) / (n - 1), (ymax - ymin) / (n - 1), (zmax - zmin) / (n - 1)))
    verts[:, 0] = verts[:, 0] + xmin
    verts[:, 1] = verts[:, 1] + ymin
    verts[:, 2] = verts[:, 2] + zmin
    ax.plot_trisurf(verts[:, 0], verts[:, 1], faces, verts[:, 2], cmap='magma', alpha=0.7, edgecolor='none', lw=0.0, shade=True)

# Add labels and title
ax.set_xlabel('X', fontsize=label_fs)
ax.set_ylabel('Y', fontsize=label_fs)
ax.set_zlabel('Z', fontsize=label_fs)
ax.xaxis.labelpad=30
plt.title(key, fontsize=legend_fs)
plt.show()