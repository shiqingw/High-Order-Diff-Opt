import numpy as np

def find_orthogonal_points(point1, point2, d, h):
    x1, y1, z = point1
    x2, y2, _  = point2
    
    # Calculate the direction vector of the line connecting the points
    dx = x2 - x1
    dy = y2 - y1
    
    # Calculate the length of the direction vector
    length = np.sqrt(dx**2 + dy**2)
    
    # Normalize the direction vector to unit length
    dx /= length
    dy /= length
    
    # Calculate the orthogonal vectors
    orthogonal_dx = -dy
    orthogonal_dy = dx
    
    # Calculate the four points
    p1 = (x1 + d * orthogonal_dx, y1 + d * orthogonal_dy, z)
    p2 = (x1 - d * orthogonal_dx, y1 - d * orthogonal_dy, z)
    p3 = (x2 + d * orthogonal_dx, y2 + d * orthogonal_dy, z)
    p4 = (x2 - d * orthogonal_dx, y2 - d * orthogonal_dy, z)

    p1_ = (x1 + d * orthogonal_dx, y1 + d * orthogonal_dy, z-h)
    p2_ = (x1 - d * orthogonal_dx, y1 - d * orthogonal_dy, z-h)
    p3_ = (x2 + d * orthogonal_dx, y2 + d * orthogonal_dy, z-h)
    p4_ = (x2 - d * orthogonal_dx, y2 - d * orthogonal_dy, z-h)

    # keep only to 2 decimal places and print the coordinates without the parantheses
    p1 = np.round(p1, 3)
    p2 = np.round(p2, 3)
    p3 = np.round(p3, 3)
    p4 = np.round(p4, 3)
    p1_ = np.round(p1_, 3)
    p2_ = np.round(p2_, 3)
    p3_ = np.round(p3_, 3)
    p4_ = np.round(p4_, 3)

    s = ""
    l = []
    for p in [p1, p2, p3, p4, p1_, p2_, p3_, p4_]:
        s += f"{p[0]} {p[1]} {p[2]}  "
        l.append([p[0], p[1], p[2]])
    print(s)
    print(l)

P_A = np.array([0.23, 0.29, 0.28])
P_B = np.array([0.42, 0.54, 0.28])
P_C = np.array([0.05, 0.83, 0.28])
P_D = np.array([-0.14, 0.58, 0.28])

find_orthogonal_points(P_A, P_B, 0.005, 0.28)
find_orthogonal_points(P_B, P_C, 0.005, 0.28)
find_orthogonal_points(P_C, P_D, 0.005, 0.28)
find_orthogonal_points(P_D, P_A, 0.005, 0.28)
