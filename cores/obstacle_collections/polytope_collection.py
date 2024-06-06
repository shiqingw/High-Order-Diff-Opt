import numpy as np
from scipy.spatial import ConvexHull

class PolytopeCollection:
    def __init__(self, dim, n_polytopes, obs_dict):
        self.dim = dim
        self.n_obsctacles = n_polytopes
        self.polytope_equations = []

        for (i, key) in enumerate(obs_dict.keys()):
            obs = obs_dict[key]
            all_vertices = np.array(obs["vertices"]) # shape (n_vertices, dim)
            A, b = self.get_facial_equations(all_vertices)
            polytope = {}
            polytope["A"] = A
            polytope["b"] = b
            polytope["n_faces"] = len(A)
            self.polytope_equations.append(polytope)
    
    def get_facial_equations(self, vertices):
        """
        vertices: np.array of shape (n_vertices, dim)
        
        Returns:
        The matrices A and b such that A * x + b <= 0 defines the polytope
        """
        hull = ConvexHull(vertices)
        A = np.zeros((len(hull.simplices), self.dim))
        b = np.zeros(len(hull.simplices))

        for (i, simplex) in enumerate(hull.simplices):
            facial_vertices = vertices[simplex]
            homogeneous_coordinates = np.c_[facial_vertices, np.ones(facial_vertices.shape[0])]
            coeffs = np.linalg.svd(homogeneous_coordinates)[-1][-1, :]

            # normalize a to have a unit norm
            norm = np.linalg.norm(coeffs[:-1])
            A[i] = coeffs[:-1]/norm
            b[i] = coeffs[-1]/norm

        # Adjust the signs of the equations
        mean_point = np.mean(vertices, axis=0)
        for i in range(len(A)):
            if np.dot(A[i], mean_point) + b[i] > 0:
                A[i] = -A[i]
                b[i] = -b[i]
        return A, b


