import numpy as np
import sympy
from sympy import lambdify, Matrix, hessian, diff, Function, zeros, simplify

class DiffOptHelper(object):
    def __init__(self, constraints_sympy, p_vars, theta_vars, dummy_vars):
        """
        Initialize the DiffOptHelper.\n
        Inputs:
            constraints_sympy: a list of sympy expressions corresponding to the constraints of cvxpy_prob.
                            They should be written using primal_vars and opt_params.
            p_vars: a list of sympy symbols corresponding to the p variables of cvxpy_prob
                        !!DO NOT include the alpha variable!!
            theta_vars: a list of sympy symbols corresponding to the external parameters of cvxpy_prob
            dummy_vars: a list of sympy symbols corresponding to the dummy parameters of cvxpy_prob
        """
        super(DiffOptHelper, self).__init__()
        if not isinstance(constraints_sympy, list) or not isinstance(constraints_sympy[0], sympy.Expr):
            raise TypeError("constraints_sympy must be a list of sympy expressions \
                            corresponding to the constraints of cvxpy_prob")
        self.constraints_sympy = constraints_sympy
        self.p_vars = p_vars
        self.theta_vars = theta_vars
        self.dummy_vars = dummy_vars
        self.implicit_dual_vars = [Function("lambda_" + str(i))(*self.theta_vars) for i in range(len(self.constraints_sympy)-1)]
        self.implicit_p_vars = [Function("p_" + str(i))(*self.theta_vars) for i in range(len(self.p_vars))]
        self.build_constraints_dict()
        self.build_matrices()
        self.build_alpha_derivative()

    def build_matrices(self):
        """
        Build the matrices N and b that are used to solve for the gradient of the primal and dual variables.\n
            N: a matrix of shape (dim(p_vars) + dim(constraints_sympy) - 1, dim(p_vars) + dim(constraints_sympy) - 1)\n
            b: a matrix of shape (dim(p_vars) + dim(constraints_sympy) - 1, dim(theta_vars))\n
            N_dtheta: a list of matrices of shape (dim(p_vars) + dim(constraints_sympy) - 1, dim(p_vars) + dim(constraints_sympy) - 1)\n
            b_dtheta: a list of matrices of shape (dim(p_vars) + dim(constraints_sympy) - 1, dim(theta_vars))\n
        """
        N = zeros(len(self.p_vars)+len(self.constraints_sympy)-1, len(self.p_vars)+len(self.constraints_sympy)-1)
        b = zeros(len(self.p_vars)+len(self.constraints_sympy)-1, len(self.theta_vars))
        Q = zeros(len(self.p_vars), len(self.p_vars))
        C = zeros(len(self.constraints_sympy)-1, len(self.p_vars))
        Phi = zeros(len(self.p_vars), len(self.theta_vars))
        Omega = zeros(len(self.constraints_sympy)-1, len(self.theta_vars))
        Q += hessian(self.constraints_sympy[0], self.p_vars)
        Phi += - Matrix([self.constraints_sympy[0]]).jacobian(self.p_vars).jacobian(self.theta_vars)

        for i in range(1, len(self.constraints_sympy)):
            Q += self.implicit_dual_vars[i-1] * hessian(self.constraints_sympy[i], self.p_vars)
            Phi += - self.implicit_dual_vars[i-1] * Matrix([self.constraints_sympy[i]]).jacobian(self.p_vars).jacobian(self.theta_vars)
            C[i-1,:] = Matrix([self.constraints_sympy[i]]).jacobian(self.p_vars)
            Omega[i-1,:] = - Matrix([self.constraints_sympy[i]]).jacobian(self.theta_vars)
        N[0:len(self.p_vars),0:len(self.p_vars)] = Q
        N[len(self.p_vars):, 0:len(self.p_vars)] = C
        N[0:len(self.p_vars), len(self.p_vars):] = C.T
        b[0:len(self.p_vars),:] = Phi
        b[len(self.p_vars):,:] = Omega
        substitution_pairs = [[a, b] for a, b in zip(self.p_vars, self.implicit_p_vars)]
        implicit_N = simplify(N.subs(substitution_pairs))
        implicit_b = simplify(b.subs(substitution_pairs))
        implicit_N_dtheta = Matrix([simplify(diff(implicit_N, theta_var)) for theta_var in self.theta_vars])
        implicit_b_dtheta = Matrix([simplify(diff(implicit_b, theta_var)) for theta_var in self.theta_vars])
        # implicit_N_dtheta = [diff(implicit_N, theta_var) for theta_var in self.theta_vars]
        # implicit_b_dtheta = [diff(implicit_b, theta_var) for theta_var in self.theta_vars]
        self.N_func = lambdify([self.implicit_p_vars, self.theta_vars, self.implicit_dual_vars, self.dummy_vars], implicit_N, "numpy")
        self.b_func = lambdify([self.implicit_p_vars, self.theta_vars, self.implicit_dual_vars, self.dummy_vars], implicit_b, "numpy")
        self.N_dtheta_func = lambdify([self.implicit_p_vars,
                                       self.theta_vars,
                                       self.implicit_dual_vars,
                                       self.dummy_vars,
                                       Matrix(self.implicit_p_vars).jacobian(self.theta_vars),
                                       Matrix(self.implicit_dual_vars).jacobian(self.theta_vars)],
                                       implicit_N_dtheta, "numpy")
        self.b_dtheta_func = lambdify([self.implicit_p_vars, 
                                       self.theta_vars, 
                                       self.implicit_dual_vars,
                                       self.dummy_vars,
                                       Matrix(self.implicit_p_vars).jacobian(self.theta_vars),
                                       Matrix(self.implicit_dual_vars).jacobian(self.theta_vars)],
                                       implicit_b_dtheta, "numpy")
    
    def build_alpha_derivative(self):
        """
        Build the function that calculates the derivative of alpha with respect to theta.\n
            alpha: a scalar.\n
            alpha_dtheta: a vector of shape (dim(theta_vars),)\n
            alpha_dthetadtheta: a matrix of shape (dim(theta_vars), dim(theta_vars))\n
        """
        substitution_pairs = [[a, b] for a, b in zip(self.p_vars, self.implicit_p_vars)]
        implicit_alpha = simplify(self.constraints_sympy[0].subs(substitution_pairs))
        implicit_alpha_dtheta = simplify(Matrix([implicit_alpha]).jacobian(self.theta_vars))
        implicit_alpha_dthetadtheta = simplify(hessian(implicit_alpha, self.theta_vars))
        self.alpha_func = lambdify([self.implicit_p_vars, self.theta_vars, self.dummy_vars], implicit_alpha, "numpy")
        self.alpha_dtheta_func = lambdify([self.implicit_p_vars, 
                                           self.theta_vars,
                                           self.dummy_vars,
                                           Matrix(self.implicit_p_vars).jacobian(self.theta_vars)],
                                           implicit_alpha_dtheta, "numpy")
        implicit_p_dthetadtheta = [hessian(self.implicit_p_vars[i], self.theta_vars) for i in range(len(self.implicit_p_vars))]
        self.alpha_dthetadtheta_func = lambdify([self.implicit_p_vars, 
                                                 self.theta_vars,
                                                 self.dummy_vars,
                                                 Matrix(self.implicit_p_vars).jacobian(self.theta_vars),
                                                 *implicit_p_dthetadtheta], implicit_alpha_dthetadtheta, "numpy")

    def build_constraints_dict(self):
        """
        Build a dictionary that stores the lambdified constraints and derivatives of the constraints.\n
        The value constraints_dict[i] is a dictionary that stores:\n
            value: dim(p_vars), dim(theta_vars) -> a scalar
            dp: dim(p_vars), dim(theta_vars) -> dim(p_vars)
            dpdp: dim(p_vars), dim(theta_vars) -> dim(p_vars) x dim(p_vars)
            dtetha: dim(p_vars), dim(theta_vars) -> dim(theta_vars)
            dpdtheta: dim(p_vars), dim(theta_vars) -> dim(p_vars) x dim(theta_vars)

        """
        constraints_dict = {}
        for i in range(len(self.constraints_sympy)):
            tmp_dict = {}
            tmp_dict["value"] = lambdify([self.p_vars, self.theta_vars, self.dummy_vars], self.constraints_sympy[i], 'numpy')
            # tmp_dict["dp"] = lambdify([self.p_vars, self.theta_vars, self.dummy_vars], Matrix([self.constraints_sympy[i]]).jacobian(self.p_vars), 'numpy')
            # tmp_dict["dpdp"] = lambdify([self.p_vars, self.theta_vars, self.dummy_vars], hessian(self.constraints_sympy[i], self.p_vars), 'numpy')
            # tmp_dict["dtheta"] = lambdify([self.p_vars, self.theta_vars, self.dummy_vars], Matrix([self.constraints_sympy[i]]).jacobian(self.theta_vars), 'numpy')
            # tmp_dict["dpdtheta"] = lambdify([self.p_vars, self.theta_vars, self.dummy_vars], Matrix([self.constraints_sympy[i]]).jacobian(self.p_vars).jacobian(self.theta_vars), 'numpy')
            # tmp_dict["dthetadp"] = lambdify([self.p_vars, self.theta_vars, self.dummy_vars], Matrix([self.constraints_sympy[i]]).jacobian(self.theta_vars).jacobian(self.p_vars), 'numpy')
            # tmp_dict["dthetadtheta"] = lambdify([self.p_vars, self.theta_vars, self.dummy_vars], Matrix([self.constraints_sympy[i]]).jacobian(self.theta_vars).jacobian(self.theta_vars), 'numpy')
            constraints_dict[i] = tmp_dict
        self.constraints_dict = constraints_dict
    
    def get_gradient(self, p_val, theta_val, dual_val, dummy_val):
        """
        Get the gradient of the primal and dual variables with respect to theta.\n
        Inputs:
            p_val: a numpy array of shape (dim(p_vars),)
            theta_val: a numpy array of shape (dim(theta_vars),)
            dual_val: a numpy array of shape (dim(dual_vars),)
            dummy_val: a numpy array of shape (dim(dummy_vars),)
        Outputs:
            grad_alpha: a numpy array of shape (1, dim(theta_vars))
            grad_p: a numpy array of shape (dim(p_vars), dim(theta_vars))
            grad_dual: a numpy array of shape (dim(dual_vars), dim(theta_vars))
        """
        threshhold = 1e-4
        constraint_values = np.array([self.constraints_dict[i]["value"](p_val, theta_val, dummy_val) for i in range(1, len(self.constraints_sympy))])
        active_set_B = np.where(np.abs(constraint_values - 1) <= threshhold)[0]
        keep_inds = np.concatenate((np.arange(len(self.p_vars)), active_set_B + len(self.p_vars)))
        N_total = self.N_func(p_val, theta_val, dual_val, dummy_val)
        b_total = self.b_func(p_val, theta_val, dual_val, dummy_val)
        N = N_total[keep_inds,:][:,keep_inds]
        b = b_total[keep_inds,:]
        X = np.linalg.pinv(N) @ b
        grad_p = X[0:len(p_val), :]
        grad_dual = np.zeros((len(dual_val), len(theta_val)))
        grad_dual[active_set_B,:] = X[len(p_val):, :]
        grad_alpha = self.alpha_dtheta_func(p_val, theta_val, dummy_val, grad_p.flatten())
        return grad_alpha, grad_p, grad_dual
    
    def get_gradient_and_hessian(self, p_val, theta_val, dual_val, dummy_val):
        """
        Get the gradient of the primal and dual variables with respect to theta.\n
        Inputs:
            alpha_val: a scalar
            p_val: a numpy array of shape (dim(p_vars),)
            theta_val: a numpy array of shape (dim(theta_vars),)
            dual_val: a numpy array of shape (dim(dual_vars),)
            dummy_val: a numpy array of shape (dim(dummy_vars),)
        Outputs:
            grad_alpha: a numpy array of shape (1, dim(theta_vars))
            grad_p: a numpy array of shape (dim(p_vars), dim(theta_vars))
            grad_dual: a numpy array of shape (dim(dual_vars), dim(theta_vars))
            hessian_alpha: a numpy array of shape (dim(theta_vars), dim(theta_vars))
            hessian_p: a list of numpy arrays of shape (dim(p_vars), dim(theta_vars), dim(theta_vars))
            hessian_dual: a list of numpy arrays of shape (dim(dual_vars), dim(theta_vars), dim(theta_vars))
        """
        threshhold = 1e-4
        constraint_values = np.array([self.constraints_dict[i]["value"](p_val, theta_val, dummy_val) for i in range(1, len(self.constraints_sympy))])
        active_set_B = np.where(np.abs(constraint_values - 1) <= threshhold)[0] 
        keep_inds = np.concatenate((np.arange(len(self.p_vars)), active_set_B + len(self.p_vars)))
        N_total = self.N_func(p_val, theta_val, dual_val, dummy_val)
        b_total = self.b_func(p_val, theta_val, dual_val, dummy_val)
        N = N_total[keep_inds,:][:,keep_inds]
        b = b_total[keep_inds,:]
        pinv_N = np.linalg.pinv(N)

        # Calculate the gradient
        X = pinv_N @ b
        grad_p = X[0:len(p_val), :]
        grad_dual = np.zeros((len(dual_val), len(theta_val)))
        grad_dual[active_set_B,:] = X[len(p_val):, :]
        grad_alpha = self.alpha_dtheta_func(p_val, theta_val, dummy_val, grad_p.flatten())

        # Calculate the hessian
        X_dtheta_total = np.zeros((X.shape[0], X.shape[1], len(theta_val)))
        N_total = self.N_func(p_val, theta_val, dual_val, dummy_val)
        N_dtheta_total = self.N_dtheta_func(p_val, theta_val, dual_val, dummy_val, grad_p.flatten(), grad_dual.flatten())
        N_dtheta_total = np.reshape(N_dtheta_total, (len(theta_val), -1, N_dtheta_total.shape[1]))
        b_dtheta_total = self.b_dtheta_func(p_val, theta_val, dual_val, dummy_val, grad_p.flatten(), grad_dual.flatten())
        b_dtheta_total = np.reshape(b_dtheta_total, (len(theta_val), -1, b_dtheta_total.shape[1]))
        for i in range(len(theta_val)):
            N_dtheta = N_dtheta_total[i][keep_inds,:][:,keep_inds]
            b_dtheta = b_dtheta_total[i][keep_inds,:]
            X_dtheta = - pinv_N @ N_dtheta @ X + pinv_N @ b_dtheta
            X_dtheta_total[:,:,i] = X_dtheta
        hessian_p = X_dtheta_total[0:len(p_val),:,:]
        hessian_dual = np.zeros((len(dual_val), len(theta_val), len(theta_val)))
        hessian_dual[active_set_B,:,:] = X_dtheta_total[len(p_val):,:,:]
        hessian_p_val = [hess.flatten() for hess in hessian_p]
        heissian_alpha = self.alpha_dthetadtheta_func(p_val, theta_val, dummy_val, grad_p.flatten(), *hessian_p_val)
        return grad_alpha.squeeze(), grad_p, grad_dual, heissian_alpha, hessian_p, hessian_dual