import numpy as np

width = 1280
height = 720
f = 910
fovy = np.arctan(height / (2 * f)) * 360 / np.pi
print(fovy)