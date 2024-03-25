from cores.scaling_functions.ellipsoid import Ellipsoid_Symmetric
from cores.differentiable_optimization.diff_opt_helper import get_dual_variable_pytorch,\
    get_gradient_pytorch, get_gradient_and_hessian_pytorch
from cores.differentiable_optimization.quat_diff_utils import Quaternion_RDRT_Symmetric
from cores.rimon_method_python.rimon_method import rimon_method_pytorch
import torch


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
        self.SF = Ellipsoid_Symmetric()
        self.RDRT = Quaternion_RDRT_Symmetric()
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
        
        # y = [A11, A12, A13, A22, A23, A33, a1, a2, a3]
        A_torch = torch.matmul(torch.matmul(R_torch, D_torch), R_torch.transpose(-1,-2)) # shape (batch_size, dim(p), dim(p))
        p_rimon = self.solver(A_torch, a_torch, B_torch, b_torch)
        F1_dp = self.SF.F_dp(p_rimon, A_torch, a_torch)
        F2_dp = self.SF.F_dp(p_rimon, B_torch, b_torch)
        F1_dpdp = self.SF.F_dpdp(p_rimon, A_torch, a_torch)
        F2_dpdp = self.SF.F_dpdp(p_rimon, B_torch, b_torch)
        F1_dy = self.SF.F_dx(p_rimon, A_torch, a_torch)
        F2_dy = torch.zeros_like(F1_dy)
        F1_dpdy = self.SF.F_dpdx(p_rimon, A_torch, a_torch)
        F2_dpdy = torch.zeros_like(F1_dpdy)
        dual_vars = get_dual_variable_pytorch(F1_dp, F2_dp)
        alpha_dy = get_gradient_pytorch(dual_vars, F1_dp, F2_dp, F1_dy, F2_dy, F1_dpdp, F2_dpdp, F1_dpdy, F2_dpdy) # shape (batch_size, 9)
        
        # Compute the gradient wrt x = [qx, qy, qz, qw, a1, a2, a3]
        RDRT_dq = self.RDRT.RDRT_dq(q_torch, D_torch, R_torch) # shape (batch_size, 6, 4)
        dim_y = 9
        dim_x = 7
        y_dx = torch.zeros((batch_size,dim_y,dim_x), dtype=A_torch.dtype, device=A_torch.device)
        y_dx[:,0:6,0:4] = RDRT_dq
        y_dx[:,6:9,4:7] = torch.eye(3, dtype=A_torch.dtype, device=A_torch.device)
        alpha_dx = torch.matmul(alpha_dy.unsqueeze(1), y_dx).squeeze(1) # shape (batch_size, 4+dim(p))

        return p_rimon, alpha_dx

    def get_gradient_and_hessian(self, a_torch, q_torch, D_torch, R_torch, B_torch, b_torch):
        """
        Compute dF1(p_rimon)/dxdx where x = [qx, qy, qz, qw, ax, ay, az] 
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
        
        # y = [A11, A12, A13, A22, A23, A33, a1, a2, a3]
        A_torch = torch.matmul(torch.matmul(R_torch, D_torch), R_torch.transpose(-1,-2)) # shape (batch_size, dim(p), dim(p))
        p_rimon = self.solver(A_torch, a_torch, B_torch, b_torch)
        F1_dp = self.SF.F_dp(p_rimon, A_torch, a_torch)
        F2_dp = self.SF.F_dp(p_rimon, B_torch, b_torch)
        F1_dpdp = self.SF.F_dpdp(p_rimon, A_torch, a_torch)
        F2_dpdp = self.SF.F_dpdp(p_rimon, B_torch, b_torch)
        F1_dy = self.SF.F_dx(p_rimon, A_torch, a_torch)
        F2_dy = torch.zeros_like(F1_dy)
        F1_dpdy = self.SF.F_dpdx(p_rimon, A_torch, a_torch)
        F2_dpdy = torch.zeros_like(F1_dpdy)
        dual_vars = get_dual_variable_pytorch(F1_dp, F2_dp)
        F1_dydy = self.SF.F_dxdx(p_rimon, A_torch, a_torch)
        F2_dydy = torch.zeros_like(F1_dydy)
        F1_dpdpdp = self.SF.F_dpdpdp(p_rimon, A_torch, a_torch)
        F2_dpdpdp = self.SF.F_dpdpdp(p_rimon, B_torch, b_torch)
        F1_dpdpdy = self.SF.F_dpdpdx(p_rimon, A_torch, a_torch)
        F2_dpdpdy = torch.zeros_like(F1_dpdpdy)
        F1_dpdydy = self.SF.F_dpdxdx(p_rimon, A_torch, a_torch)
        F2_dpdydy = torch.zeros_like(F1_dpdydy)
        # alpha_dy: shape (batch_size, 9)
        # alpha_dydy: shape (batch_size, 9, 9)
        alpha_dy, alpha_dydy = get_gradient_and_hessian_pytorch(dual_vars, F1_dp, F2_dp, F1_dy, F2_dy, F1_dpdp, F2_dpdp, F1_dpdy, F2_dpdy,
                                     F1_dydy, F2_dydy, F1_dpdpdp, F2_dpdpdp, F1_dpdpdy, F2_dpdpdy, F1_dpdydy, F2_dpdydy)
        
        # Compute the gradient wrt x = [qx, qy, qz, qw, a1, a2, a3]
        RDRT_dq = self.RDRT.RDRT_dq(q_torch, D_torch, R_torch) # shape (batch_size, 6, 4)
        dim_y = 9
        dim_x = 7
        y_dx = torch.zeros((batch_size,dim_y,dim_x), dtype=A_torch.dtype, device=A_torch.device)
        y_dx[:,0:6,0:4] = RDRT_dq
        y_dx[:,6:9,4:7] = torch.eye(3, dtype=A_torch.dtype, device=A_torch.device)
        alpha_dx = torch.matmul(alpha_dy.unsqueeze(1), y_dx).squeeze(1) # shape (batch_size, 4+dim(p))

        # Compute the hessian wrt x = [qx, qy, qz, qw, a1, a2, a3]
        RDRT_dqdq = self.RDRT.RDRT_dqdq(q_torch, D_torch, R_torch)
        tmp1 = torch.matmul(torch.matmul(y_dx.transpose(-1, -2), alpha_dydy), y_dx) # shape (batch_size, 7, 7)
        y_dxdx = torch.zeros((batch_size, dim_y, dim_x, dim_x), dtype=A_torch.dtype, device=A_torch.device)
        y_dxdx[:,0:6,0:4,0:4] = RDRT_dqdq
        tmp2 = torch.einsum('bi,bijk->bjk', alpha_dy, y_dxdx) # shape (batch_size, 7, 7)
        alpha_dxdx = tmp1 + tmp2 # shape (batch_size, 7, 7)
        return p_rimon, alpha_dx, alpha_dxdx