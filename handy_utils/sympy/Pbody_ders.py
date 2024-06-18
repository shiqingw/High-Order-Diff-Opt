import sympy as sp

qx, qy, qz, qw = sp.symbols('qx qy qz qw', real=True)
r11, r12, r13, r21, r22, r23, r31, r32, r33 = sp.symbols('r11 r12 r13 r21 r22 r23 r31 r32 r33', real=True)
dx, dy, dz = sp.symbols('dx dy dz', real=True)
px, py, pz = sp.symbols('px py pz', real=True)
qxx, qxy, qxz, qxw, qyy, qyz, qyw, qzz, qzw, qww = sp.symbols('qxx qxy qxz qxw qyy qyz qyw qzz qzw qww', real=True)

x_vars = [dx, dy, dz, qx, qy, qz, qw]
p_vars = [px, py, pz]

R = sp.Matrix([[2*(qw**2+qx**2)-1, 2*(qx*qy-qw*qz), 2*(qx*qz+qw*qy)],
                [2*(qx*qy+qw*qz), 2*(qw**2+qy**2)-1, 2*(qy*qz-qw*qx)],
                [2*(qx*qz-qw*qy), 2*(qy*qz+qw*qx), 2*(qw**2+qz**2)-1]])
p = sp.Matrix([px, py, pz])
d = sp.Matrix([dx, dy, dz])
P = R.T @ (p-d)

substitution_expr = {qw**2+qx**2: (r11+1)/2,
                    qx*qy-qw*qz: r12/2,
                    qx*qz+qw*qy: r13/2,
                    qx*qy+qw*qz: r21/2,
                    qw**2+qy**2: (r22+1)/2,
                    qy*qz-qw*qx: r23/2,
                    qx*qz-qw*qy: r31/2,
                    qy*qz+qw*qx: r32/2,
                    qw**2+qz**2: (r33+1)/2,
                    2*qw**2+2*qx**2-1: r11,
                    2*qw**2+2*qy**2-1: r22,
                    2*qw**2+2*qz**2-1: r33,
                    2*qx*qy-2*qw*qz: r12,
                    2*qx*qz+2*qw*qy: r13,
                    2*qx*qy+2*qw*qz: r21,
                    2*qy*qz-2*qw*qx: r23,
                    2*qx*qz-2*qw*qy: r31,
                    2*qy*qz+2*qw*qx: r32,
                    qx*qx: qxx,
                    qx*qy: qxy,
                    qx*qz: qxz,
                    qx*qw: qxw,
                    qy*qy: qyy,
                    qy*qz: qyz,
                    qy*qw: qyw,
                    qz*qz: qzz,
                    qz*qw: qzw,
                    qw*qw: qww}

# # P_dp
# P_dp = sp.simplify(sp.simplify(P.jacobian(p_vars)).subs(substitution_expr))
# print(P_dp)

# P_dx
dim_P = 3
dim_p = len(p_vars)
dim_x = len(x_vars)
for i in range(dim_P):
    for j in range(dim_x):
        result = P[i].diff(x_vars[j]).subs(substitution_expr)
        if result != 0:
            print(f'P_dx({i},{j}) = {result};')


# P_dpdx
# dim_P = 3
# dim_p = len(p_vars)
# dim_x = len(x_vars)
# for i in range(dim_P):
#     for j in range(dim_p):
#         for k in range(dim_x):
#             result = P[i].diff(p_vars[j]).diff(x_vars[k])
#             if result != 0:
#                 print(f'P_dpdx({i},{j},{k}) = {result};')

# P_dxdx
# dim_P = 3
# dim_p = len(p_vars)
# dim_x = len(x_vars)
# for i in range(dim_P):
#     for j in range(dim_x):
#         for k in range(dim_x):
#             result = P[i].diff(x_vars[j]).diff(x_vars[k])
#             if result != 0:
#                 print(f'P_dxdx({i},{j},{k}) = {result};')

# # P_dpdxdx
# dim_P = 3
# dim_p = len(p_vars)
# dim_x = len(x_vars)
# for i in range(dim_P):
#     for j in range(dim_p):
#         for k in range(dim_x):
#             for l in range(dim_x):
#                 result = P[i].diff(p_vars[j]).diff(x_vars[k]).diff(x_vars[l])
#                 if result != 0:
#                     print(f'P_dpdxdx({i},{j},{k},{l}) = {result};')