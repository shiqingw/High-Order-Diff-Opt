import numpy as np
from scipy.spatial import ConvexHull

# Define the vertices
points = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]])

# Compute the convex hull
hull = ConvexHull(points)

# Extract the equations of the facets
for simplex in hull.simplices:
    vertices = points[simplex]
    A = np.c_[vertices, np.ones(vertices.shape[0])]
    coeffs = np.linalg.svd(A)[-1][-1, :]
    equation = coeffs[:-1], coeffs[-1]
    print(f"Face defined by vertices {vertices} has equation: {equation[0]} * x + {equation[1]} = 0")
