import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

class PolytopeCollection:
    def __init__(self, dim, n_polytopes, obs_dict):
        self.dim = dim
        self.n_obsctacles = n_polytopes
        self.face_equations = {}

        for (i, key) in enumerate(obs_dict.keys()):
            obs = obs_dict[key]
            pos = np.array(obs["pos"])
            quat = np.array(obs["quat"])
            vertices = np.array(obs["vertices"])
            all_vertices_in_world = self.convert_vertices_to_world_3d(vertices, pos, quat) # shape (n_vertices, dim)
            A, b = self.get_facial_equations(all_vertices_in_world)
            polytope = {}
            polytope["A"] = A
            polytope["b"] = b
            polytope["n_faces"] = len(A)
            polytope["vertices_in_world"] = all_vertices_in_world
            self.face_equations[key] = polytope
    
    def convert_vertices_to_world_3d(self, vertices, pos, quat):
        quat = quat / np.linalg.norm(quat)
        R = Rotation.from_quat(quat).as_matrix()
        points = vertices @ R.T + pos
        return points

    
    def get_facial_equations(self, vertices):
        """
        vertices: np.array of shape (n_vertices, dim)
        
        Returns:
        The matrices A and b such that A * x + b <= 0 defines the polytope
        """
        hull = ConvexHull(vertices)
        tmp = np.zeros((len(hull.simplices), self.dim+1))

        for (i, simplex) in enumerate(hull.simplices):
            facial_vertices = vertices[simplex]
            homogeneous_coordinates = np.c_[facial_vertices, np.ones(facial_vertices.shape[0])]
            coeffs = np.linalg.svd(homogeneous_coordinates)[-1][-1, :]
            coeffs[np.abs(coeffs) < 1e-6] = 0
            tmp[i] = coeffs
        
        # Adjust the signs of the equations
        mean_point = np.mean(vertices, axis=0)
        for i in range(len(tmp)):
            if np.dot(tmp[i,0:self.dim], mean_point) + tmp[i,self.dim] > 0:
                tmp[i] = -tmp[i]

        # normalize to have a unit norm
        norm = np.linalg.norm(tmp[:,0:self.dim], axis=1)
        tmp = tmp / norm[:, np.newaxis]
        
        tmp = np.unique(tmp, axis=0)
        A = tmp[:,0:self.dim]
        b = tmp[:,self.dim]

        return A, b


