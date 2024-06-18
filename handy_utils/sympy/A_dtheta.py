import sympy as sp

theta = sp.symbols('theta', real=True)
a, b= sp.symbols('a b', real=True)
r11, r12, r21, r22 = sp.symbols('r11 r12 r21 r22', real=True)

R_b_to_w = sp.Matrix([[sp.cos(theta), -sp.sin(theta)],
                      [sp.sin(theta), sp.cos(theta)]])
ellipse_coef_sp = sp.Matrix([[a, 0], [0, b]])
M = R_b_to_w @ ellipse_coef_sp @ R_b_to_w.T

substitution_expr = {sp.cos(theta): r11,
                     sp.sin(theta): r21}
# Calculate and print Jacobian and Hessian for each element in M with respect to x_vars
for i in range(M.rows):
    for j in range(i, M.cols):
        # Current element
        element = M[i, j]

        # Jacobian
        # jacobian = element.diff(theta)
        # Mij_dq = sp.simplify(sp.simplify(jacobian).subs(substitution_expr))
        # print(element)
        # print(f'M{i+1}{j+1}_dtheta=',Mij_dq)
        # print()
        
        # Hessian
        hessian = element.diff(theta).diff(theta)
        print(f'Hessian of M[{i+1}, {j+1}]:')
        Mij_dqdq = sp.simplify(sp.simplify(hessian).subs(substitution_expr))
        print(f'M{i+1}{j+1}_dqdq_11 =', Mij_dqdq)
        print()