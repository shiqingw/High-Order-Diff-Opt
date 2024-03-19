from cores.scaling_functions.ellipsoid_brute_force import Ellipsoid_Brute_Force
from cores.differentiable_optimization.diff_opt_helper import get_dual_variable_pytorch,\
    get_gradient_pytorch, get_gradient_and_hessian_pytorch
from cores.differentiable_optimization.quat_diff_utils import Quaternion_RDRT_Symmetric
from cores.rimon_method_python.rimon_method import rimon_method_pytorch
import torch
import time


class Ellipsoid_Quat_Pos():
    """
    This class is used to compute the gradient and hessian of a differentiable optimization problem.
    More specifically, given two ellipsoids:
    F1(p) = (p-a)^T A (p-a) <= 1
    F2(p) = (p-b)^T B (p-b) <= 1
    where A, B are symmetric positive definite matrices, a, b are vectors in R^n, and p is the optimization variable.
    We use Rimon's method to find the point p_rimon on the second ellipsoid where the ellipsoidal level surfaces surrounding
    the first ellipsoid first touch the second ellipsoid.
    Define the scaling function as F1(p_rimon). Then we can compute the gradient and hessian of F1(p_rimon) with respect to
    the quaternion (which is q = [qx,qy,qz,qw]) and position (which is a) variables of F1.
    """

    def __init__(self):
        self.SF = Ellipsoid_Brute_Force()
        self.solver = rimon_method_pytorch
    
    def get_gradient(self, a_torch, q_torch, D_torch, R_torch, B_torch, b_torch):
        """
        Compute dF1(p_rimon)/dx where x = [qx, qy, qz, qw, ax, ay, az] 
        a_torch (np.array, shape (dim(p),)): The center of the first ellipsoid.
        q_torch (np.array, shape (4,)): The quaternion representing the rotation of the first ellipsoid.
        D_torch (np.array, shape (dim(p),dim(p))): The diagonal matrix representing the digonal elements of the first ellipsoid.
        R_torch (np.array, shape (dim(p),dim(p))): The rotation matrix representing the rotation of the first ellipsoid.
        B_torch (np.array, shape (dim(p),dim(p))): The matrix of the second ellipsoid.
        b_torch (np.array, shape (dim(p),)): The center of the second ellipsoid.
              
        Returns:
        (torch.Tensor, shape (batch_size, dim(p)): p_rimon
        (torch.Tensor, shape (batch_size, 4+dim(p)): dF1(p_rimon)/dx
        """
        
        batch_size, dim_p = a_torch.shape
        if dim_p != 3:
            raise NotImplementedError("dim_p must be 3")
        
        # x = [qx, qy, qz, qw, a1, a2, a3]
        A_torch = torch.matmul(torch.matmul(R_torch, D_torch), R_torch.transpose(-1,-2)) # shape (batch_size, dim(p), dim(p))
        p_rimon = self.solver(A_torch, a_torch, B_torch, b_torch)
        F1, F1_dp, F1_dx, F1_dpdp, F1_dpdx = self.SF.prepare_gradient(p_rimon, a_torch, q_torch, D_torch, R_torch, A_torch)
        F2_dp = self.SF.F_dp(p_rimon, B_torch, b_torch)
        F2_dpdp = self.SF.F_dpdp(p_rimon, B_torch, b_torch)
        F2_dx = torch.zeros_like(F1_dx)
        F2_dpdx = torch.zeros_like(F1_dpdx)
        dual_vars = get_dual_variable_pytorch(F1_dp, F2_dp)
        alpha_dx = get_gradient_pytorch(dual_vars, F1_dp, F2_dp, F1_dx, F2_dx, F1_dpdp, F2_dpdp,
                                        F1_dpdx, F2_dpdx) # shape (batch_size, 7)
        
        return F1, p_rimon, alpha_dx

    def get_gradient_and_hessian(self, a_torch, q_torch, D_torch, R_torch, B_torch, b_torch):
        """
        Compute dF1(p_rimon)/dx where x = [qx, qy, qz, qw, ax, ay, az] 
        a_torch (np.array, shape (dim(p),)): The center of the first ellipsoid.
        q_torch (np.array, shape (4,)): The quaternion representing the rotation of the first ellipsoid.
        D_torch (np.array, shape (dim(p),dim(p))): The diagonal matrix representing the digonal elements of the first ellipsoid.
        R_torch (np.array, shape (dim(p),dim(p))): The rotation matrix representing the rotation of the first ellipsoid.
        B_torch (np.array, shape (dim(p),dim(p))): The matrix of the second ellipsoid.
        b_torch (np.array, shape (dim(p),)): The center of the second ellipsoid.

        Returns:
        (torch.Tensor, shape (batch_size, dim(p)): p_rimon
        (torch.Tensor, shape (batch_size, 4+dim(p)): dF1(p_rimon)/dx
        (torch.Tensor, shape (batch_size, 4+dim(p), 4+dim(p)): d2F1(p_rimon)/dx2
        """

        batch_size, dim_p = a_torch.shape
        if dim_p != 3:
            raise NotImplementedError("dim_p must be 3")
        
        # x = [qx, qy, qz, qw, a1, a2, a3]
        A_torch = torch.matmul(torch.matmul(R_torch, D_torch), R_torch.transpose(-1,-2)) # shape (batch_size, dim(p), dim(p))
        p_rimon = self.solver(A_torch, a_torch, B_torch, b_torch)
        F1, F1_dp, F1_dx, F1_dpdp, F1_dpdx, F1_dxdx, F1_dpdpdp, F1_dpdpdx, F1_dpdxdx = self.SF.prepare_gradient_and_hessian(p_rimon, a_torch, q_torch, D_torch, R_torch, A_torch)
        F2_dp = self.SF.F_dp(p_rimon, B_torch, b_torch)
        F2_dpdp = self.SF.F_dpdp(p_rimon, B_torch, b_torch)
        F2_dx = torch.zeros_like(F1_dx)
        F2_dpdx = torch.zeros_like(F1_dpdx)
        dual_vars = get_dual_variable_pytorch(F1_dp, F2_dp)
        F2_dxdx = torch.zeros_like(F1_dxdx)
        F2_dpdpdp = self.SF.F_dpdpdp(p_rimon, B_torch, b_torch)
        F2_dpdpdx = torch.zeros_like(F1_dpdpdx)
        F2_dpdxdx = torch.zeros_like(F1_dpdxdx)
        alpha_dx, alpha_dxdx = get_gradient_and_hessian_pytorch(dual_vars, F1_dp, F2_dp, F1_dx, F2_dx, F1_dpdp, F2_dpdp, F1_dpdx, F2_dpdx,
                                     F1_dxdx, F2_dxdx, F1_dpdpdp, F2_dpdpdp, F1_dpdpdx, F2_dpdpdx, F1_dpdxdx, F2_dpdxdx)
        
        return F1, p_rimon, alpha_dx, alpha_dxdx