import sympy as sp

qx, qy, qz, qw = sp.symbols('qx qy qz qw', real=True)
a, b, c = sp.symbols('a b c', real=True)
r11, r12, r13, r21, r22, r23, r31, r32, r33 = sp.symbols('r11 r12 r13 r21 r22 r23 r31 r32 r33', real=True)
x_vars = [qx, qy, qz, qw]

R_b_to_w = sp.Matrix([[2*(qw**2+qx**2)-1, 2*(qx*qy-qw*qz), 2*(qx*qz+qw*qy)],
                            [2*(qx*qy+qw*qz), 2*(qw**2+qy**2)-1, 2*(qy*qz-qw*qx)],
                            [2*(qx*qz-qw*qy), 2*(qy*qz+qw*qx), 2*(qw**2+qz**2)-1]])
ellipse_coef_sp = sp.Matrix([[a, 0, 0], [0, b, 0], [0, 0, c]])
M = R_b_to_w @ ellipse_coef_sp @ R_b_to_w.T

substitution_expr = {qw**2+qx**2:  (r11+1)/2,
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
                    2*qw**2+2*qz**2-1: r33}
# Calculate and print Jacobian and Hessian for each element in M with respect to x_vars
for i in range(M.rows):
    for j in range(M.cols):
        # Current element
        element = M[i, j]

        # Jacobian
        # jacobian = sp.Matrix([element.diff(var) for var in x_vars])
        # print(f'Jacobian of M[{i+1}, {j+1}]:')
        # Mij_dq = sp.simplify(sp.simplify(jacobian).subs(substitution_expr))
        # print(Mij_dq[0])
        # print(Mij_dq[1])
        # print(Mij_dq[2])
        # print(Mij_dq[3])
        # print()
        
        # Hessian
        hessian = sp.Matrix([[element.diff(var1).diff(var2) for var2 in x_vars] for var1 in x_vars])
        print(f'Hessian of M[{i+1}, {j+1}]:')
        Mij_dqdq = sp.simplify(sp.simplify(hessian).subs(substitution_expr))
        print(f'M{i+1}{j+1}_dqdq_11 =', Mij_dqdq[0,0])
        print(f'M{i+1}{j+1}_dqdq_12 =',Mij_dqdq[0,1])
        print(f'M{i+1}{j+1}_dqdq_13 =',Mij_dqdq[0,2])
        print(f'M{i+1}{j+1}_dqdq_14 =',Mij_dqdq[0,3])
        print(f'M{i+1}{j+1}_dqdq_22 =',Mij_dqdq[1,1])
        print(f'M{i+1}{j+1}_dqdq_23 =',Mij_dqdq[1,2])
        print(f'M{i+1}{j+1}_dqdq_24 =',Mij_dqdq[1,3])
        print(f'M{i+1}{j+1}_dqdq_33 =',Mij_dqdq[2,2])
        print(f'M{i+1}{j+1}_dqdq_34 =',Mij_dqdq[2,3])
        print(f'M{i+1}{j+1}_dqdq_44 =',Mij_dqdq[3,3])
        print()