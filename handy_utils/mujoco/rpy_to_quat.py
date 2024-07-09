from scipy.spatial.transform import Rotation
import numpy as np

# RPY angles in radians
roll = 0
pitch = 0
yaw = np.pi/3 + np.pi/4

# Create a Rotation object from RPY
rotation = Rotation.from_euler('xyz', [roll, pitch, yaw])

# Convert to quaternion
quaternion = rotation.as_quat()
if quaternion[3] < 0:
    quaternion = -quaternion
print("wxyz:", quaternion[3], quaternion[0], quaternion[1], quaternion[2])
print("xyzw:", f"[{quaternion[0]}, {quaternion[1]}, {quaternion[2]}, {quaternion[3]}]")
