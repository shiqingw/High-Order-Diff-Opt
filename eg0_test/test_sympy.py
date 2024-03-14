import sympy as sp

cx, cy, cz, qx, qy, qz, qw = sp.symbols('cx cy cz qx qy qz qw', real=True)
c11, c22, c33 = sp.symbols('c11 c22 c33', real=True)
px, py, pz = sp.symbols('px py pz', real=True)
p_vars = [px, py, pz]
# x_vars = [cx, cy, cz, qx, qy, qz, qw]
x_vars = [qx, qy, qz, qw]


R_b_to_w = sp.Matrix([[2*(qw**2+qx**2)-1, 2*(qx*qy-qw*qz), 2*(qx*qz+qw*qy)],
                            [2*(qx*qy+qw*qz), 2*(qw**2+qy**2)-1, 2*(qy*qz-qw*qx)],
                            [2*(qx*qz-qw*qy), 2*(qy*qz+qw*qx), 2*(qw**2+qz**2)-1]])
c_vec = sp.Matrix([cx, cy, cz])
p_vec = sp.Matrix([px, py, pz])
ellipse_coef_sp = sp.Matrix([[c11, 0, 0], [0, c22, 0], [0, 0, c33]])
f = (p_vec-c_vec).T @ R_b_to_w @ ellipse_coef_sp @ R_b_to_w.T @ (p_vec-c_vec) 

print(sp.diff(f,qx))