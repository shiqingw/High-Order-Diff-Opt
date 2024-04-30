import numpy as np
from scipy.spatial.transform import Rotation as R
import sympy as sp
from cores.configuration.configuration import Configuration
config = Configuration()

def get_quat_from_rot_matrix(rot_matrix):
    """
    The returned value is in scalar-first (x, y, z, w) format
    """
    r = R.from_matrix(rot_matrix)
    quat = r.as_quat()
    return quat

def get_rot_matrix_from_quat(quat):
    """
    The quaternion value is in scalar-first (x, y, z, w) format
    """
    r = R.from_quat(quat)
    rot_matrix = r.as_matrix()
    return rot_matrix

def get_rot_matrix_from_euler_zyx(euler):
    """
    The euler value is in extrinsic (z,y,x) format
    """
    r = R.from_euler('zyx', euler, degrees=False)
    rot_matrix = r.as_matrix()
    return rot_matrix

def get_Q_matrix_from_quat(quat):
    """
    Return the Q matrix for quaternion differentiation. 
    The quaternion value is in scalar-first (x, y, z, w) format.
    See: https://math.stackexchange.com/questions/189185/quaternion-differentiation
    """
    qx, qy, qz, qw = quat
    Q = np.array([[qw, qz, -qy],
                    [-qz, qw, qx],
                    [qy, -qx, qw],
                    [-qx, -qy, -qz]], dtype=config.np_dtype)
    return Q

def get_dQ_matrix(dquat):
    """
    Return the dQ matrix for second-order quaternion differentiation. 
    The quaternion value is in scalar-first (x, y, z, w) format
    """
    dx, dy, dz, dw = dquat
    dQ = np.array([[dw, dz, -dy],
                    [-dz, dw, dx],
                    [dy, -dx, dw],
                    [-dx, -dy, -dz]], dtype=config.np_dtype)
    return dQ

def sp_get_rot_matrix_from_quat(quat):
    """
    The quaternion value is in (qx, qy, qz, qw) format
    """
    qx, qy, qz, qw = quat
    rot_matrix = sp.Matrix([[2*(qw**2+qx**2)-1, 2*(qx*qy-qw*qz), 2*(qx*qz+qw*qy)],
                            [2*(qx*qy+qw*qz), 2*(qw**2+qy**2)-1, 2*(qy*qz-qw*qx)],
                            [2*(qx*qz-qw*qy), 2*(qy*qz+qw*qx), 2*(qw**2+qz**2)-1]])
    return rot_matrix

def np_get_rot_matrix_from_quat(quat):
    """
    The quaternion value is in (qx, qy, qz, qw) format
    """
    qx, qy, qz, qw = quat
    rot_matrix = np.array([[2*(qw**2+qx**2)-1, 2*(qx*qy-qw*qz), 2*(qx*qz+qw*qy)],
                            [2*(qx*qy+qw*qz), 2*(qw**2+qy**2)-1, 2*(qy*qz-qw*qx)],
                            [2*(qx*qz-qw*qy), 2*(qy*qz+qw*qx), 2*(qw**2+qz**2)-1]])
    return rot_matrix

def np_get_quat_qw_first(quat):
    """
    The original quaternion value is in (qx, qy, qz, qw) format
    """
    return np.array([quat[3], quat[0], quat[1], quat[2]])

def get_homogenous_matrix(R, p):
    """
    Returns a homogenous matrix with the rotation matrix R and the position vector p

    Args:
        R (np.array): rotation matrix, shape (3,3)
        p (np.array): position vector, shape (3,)
    
    Returns:
        H (np.array): homogenous matrix, shape (4,4)
    """
    H = np.eye(4)
    H[:3, :3] = R
    H[:3, 3] = p
    return H

def inverse_homogenous_matrix(H):
    """
    Returns the inverse of a homogenous matrix

    Args:
        H (np.array): homogenous matrix, shape (4,4)
    
    Returns:
        H_inv (np.array): inverse of homogenous matrix, shape (4,4)
    """
    H_inv = np.eye(4)
    H_inv[:3, :3] = H[:3, :3].T
    H_inv[:3, 3] = -H[:3, :3].T @ H[:3, 3]
    return H_inv