from scipy.spatial.transform import Rotation
import numpy as np

# RPY angles in radians
roll = 0
pitch = np.pi
yaw = -np.pi/2

# Create a Rotation object from RPY
rotation = Rotation.from_euler('xyz', [roll, pitch, yaw])

# Convert to quaternion
quaternion = rotation.as_quat()
print("wxyz:", quaternion[3], quaternion[0], quaternion[1], quaternion[2])
# print("xyzw:", quaternion[0], quaternion[1], quaternion[2], quaternion[3])
