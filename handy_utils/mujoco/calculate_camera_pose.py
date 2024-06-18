import numpy as np
from scipy.spatial.transform import Rotation

def spherical_to_cartesian(distance, azimuth, elevation):
    """
    Convert spherical coordinates to Cartesian coordinates.
    """
    x = distance * np.cos(-np.radians(elevation)) * np.cos(np.radians(azimuth+180))
    y = distance * np.cos(-np.radians(elevation)) * np.sin(np.radians(azimuth+180))
    z = distance * np.sin(-np.radians(elevation))
    print(x, y, z)
    return np.array([x, y, z]).squeeze()

def calculate_camera_position_orientation(cam_distance, cam_azimuth, cam_elevation, cam_lookat):
    """
    Calculate the position and orientation of the camera.
    """
    # Convert spherical coordinates to Cartesian coordinates
    camera_pos = np.array(cam_lookat) + spherical_to_cartesian(cam_distance, cam_azimuth, cam_elevation)
    
    rot1 = Rotation.from_euler('z', cam_azimuth, degrees=True).as_matrix()
    rot2 = Rotation.from_euler('y', -cam_elevation, degrees=True).as_matrix()
    rot3 = np.array([[0, 0, 1], 
                     [-1, 0, 0], 
                     [0, -1, 0]])
    rot4 = np.diag([1, -1, -1]).T
    rot = Rotation.from_matrix(rot1 @ rot2 @ rot3 @ rot4)
    quat = rot.as_quat()

    # Put qw first
    quat = np.roll(quat, 1)
    # quat /= np.sign(quat[0])
    
    return camera_pos, quat

cam_distance = 10.0
cam_azimuth = 150
cam_elevation = -40
cam_lookat = np.array([0.0, 0.0, 2.0])

camera_pos, quat = calculate_camera_position_orientation(cam_distance, cam_azimuth, cam_elevation, cam_lookat)
print(f"Camera Position: {camera_pos}")
print(f"Quat: {quat}")

