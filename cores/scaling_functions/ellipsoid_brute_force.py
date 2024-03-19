import torch

class Ellipsoid_Brute_Force():
    """
    We consider F(p) = (p - a)^T @ R @ D @ R.T @ (p - a) where a is a vector and
    R = [[[2*(qw**2+qx**2)-1, 2*(qx*qy-qw*qz), 2*(qx*qz+qw*qy)],
        [2*(qx*qy+qw*qz), 2*(qw**2+qy**2)-1, 2*(qy*qz-qw*qx)],
        [2*(qx*qz-qw*qy), 2*(qy*qz+qw*qx), 2*(qw**2+qz**2)-1]].
    D = diag(D11, D22, D33).
    Note A = R @ D @ R.T for simplicity.
    The parameters of F are x = [qx, qy, qz, qw, ax, ay, az]. This class calculates
    various derivatives of F w.r.t. x and p.
    """

    @staticmethod
    def F_dp(p, A, a):
        """
        Compute dF/dp.
        p (torch.Tensor, shape (batch_size, dim(p))): input
        A (torch.Tensor, shape (batch_size, dim(p), dim(p))): real symmetric quadratic coefficient matrix
        a (torch.Tensor, shape (batch_size, dim(p))): center of the ellipsoid

        Returns:
        (torch.Tensor, shape (batch_size, dim(p))): dF/dp
        """
        tmp = (p - a).unsqueeze(-1) # shape (batch_size, dim(p), 1)

        return 2*torch.matmul(A, tmp).squeeze(-1)

    @staticmethod
    def F_dpdp(p, A, a):
        """
        Compute d^2F/dpdp.
        p (torch.Tensor, shape (batch_size, dim(p))): input
        A (torch.Tensor, shape (batch_size, dim(p), dim(p))): real symmetric quadratic coefficient matrix
        a (torch.Tensor, shape (batch_size, dim(p))): center of the ellipsoid

        Returns:
        (torch.Tensor, shape (batch_size, dim(p), dim(p))): d^2F/dpdp
        """

        return 2*A
    
    @staticmethod
    def F_dpdpdp(p, A, a):
        """
        Compute d^3F/dpdpdp.
        p (torch.Tensor, shape (batch_size, dim(p))): input
        A (torch.Tensor, shape (batch_size, dim(p), dim(p))): real symmetric quadratic coefficient matrix
        a (torch.Tensor, shape (batch_size, dim(p))): center of the ellipsoid

        Returns:
        (torch.Tensor, shape (batch_size, dim(p), dim(p), dim(p)): d^3F/dpdpdp
        """
        batch_size = A.shape[0]
        dim_p = A.shape[-1]

        return torch.zeros((batch_size, dim_p, dim_p, dim_p), dtype=A.dtype, device=A.device)
    
    @staticmethod
    def prepare_gradient(p, a, q, D, R, A):
        """
        Compute F, dF/dp, dF/dx, d^2F/dpdp, d^2F/dpdx, d^2F/dxdx.
        p (torch.Tensor, shape (batch_size, dim(p))): input
        a (torch.Tensor, shape (batch_size, dim(p))): center of the ellipsoid
        q (torch.Tensor, shape (batch_size, dim(q))): quaternion
        D (torch.Tensor, shape (batch_size, dim(p), dim(p))): diagonal matrix
        R (torch.Tensor, shape (batch_size, dim(p), dim(p))): rotation matrix
        A (torch.Tensor, shape (batch_size, dim(p), dim(p))): A = R @ D @ R.T

        Returns:
        (torch.Tensor, shape (batch_size,)): F
        (torch.Tensor, shape (batch_size, dim(p))): dF/dp
        (torch.Tensor, shape (batch_size, dim(q)+dim(p)): dF/dx
        (torch.Tensor, shape (batch_size, dim(p), dim(p))): d^2F/dpdp
        (torch.Tensor, shape (batch_size, dim(p), dim(q)+dim(p)): d^2F/dpdx
        (torch.Tensor, shape (batch_size, dim(q)+dim(p), dim(q)+dim(p)): d^2F/dxdx
        (torch.Tensor, shape (batch_size, dim(p), dim(p), dim(p)): d^3F/dpdpdp
        (torch.Tensor, shape (batch_size, dim(p), dim(p), dim(q)+dim(p)): d^3F/dpdpdx
        (torch.Tensor, shape (batch_size, dim(p), dim(q)+dim(p), dim(q)+dim(p)): d^3F/dpdxdx
        """
        qx, qy, qz, qw = q.unbind(-1)
        d11, d22, d33 = D[:,0,0], D[:,1,1], D[:,2,2]
        r11 = R[:,0,0]
        r12 = R[:,0,1]
        r13 = R[:,0,2]
        r21 = R[:,1,0]
        r22 = R[:,1,1]
        r23 = R[:,1,2]
        r31 = R[:,2,0]
        r32 = R[:,2,1]
        r33 = R[:,2,2]
        pxx = qx*qx
        pxy = qx*qy
        pxz = qx*qz
        pxw = qx*qw
        pyy = qy*qy
        pyz = qy*qz
        pyw = qy*qw
        pzz = qz*qz
        pzw = qz*qw
        pww = qw*qw
        v1 = p[:,0]-a[:,0]
        v2 = p[:,1]-a[:,1]
        v3 = p[:,2]-a[:,2]

        batch_size = p.shape[0]
        dim_p = 3
        dim_q = 4
        dim_x = dim_p + dim_q
        p_minus_a = (p - a).unsqueeze(-1) # shape (batch_size, dim(p), 1)

        F = torch.matmul(torch.matmul(p_minus_a.transpose(-1,-2), A), p_minus_a).flatten()# (batch_size,) 

        F_dp = 2*torch.matmul(A, p_minus_a).squeeze(-1) # shape (batch_size, dim(p))

        F_dx = torch.zeros((batch_size, dim_x), dtype=p.dtype, device=p.device)
        F_dx[:,0] = 2*v1*(2*d11*qx*(r11*v1 + 2*v2*(pxy + pzw) + 2*v3*(pxz - pyw)) + d11*r11*(2*qx*v1 + qy*v2 + qz*v3) + d22*qy*(r22*v2 + 2*v1*(pxy - pzw) + 2*v3*(pxw + pyz)) + 2*d22*(pxy - pzw)*(qw*v3 + qy*v1) + d33*qz*(r33*v3 + 2*v1*(pxz + pyw) - 2*v2*(pxw - pyz)) - 2*d33*(pxz + pyw)*(qw*v2 - qz*v1)) + 2*v2*(d11*qy*(r11*v1 + 2*v2*(pxy + pzw) + 2*v3*(pxz - pyw)) + 2*d11*(pxy + pzw)*(2*qx*v1 + qy*v2 + qz*v3) + d22*r22*(qw*v3 + qy*v1) - d33*qw*(r33*v3 + 2*v1*(pxz + pyw) - 2*v2*(pxw - pyz)) + 2*d33*(pxw - pyz)*(qw*v2 - qz*v1)) + 2*v3*(d11*qz*(r11*v1 + 2*v2*(pxy + pzw) + 2*v3*(pxz - pyw)) + 2*d11*(pxz - pyw)*(2*qx*v1 + qy*v2 + qz*v3) + d22*qw*(r22*v2 + 2*v1*(pxy - pzw) + 2*v3*(pxw + pyz)) + 2*d22*(pxw + pyz)*(qw*v3 + qy*v1) - d33*r33*(qw*v2 - qz*v1))
        F_dx[:,1] = 2*v1*(-d11*r11*(qw*v3 - qx*v2) + d22*qx*(r22*v2 + 2*v1*(pxy - pzw) + 2*v3*(pxw + pyz)) + 2*d22*(pxy - pzw)*(qx*v1 + 2*qy*v2 + qz*v3) + d33*qw*(r33*v3 + 2*v1*(pxz + pyw) - 2*v2*(pxw - pyz)) + 2*d33*(pxz + pyw)*(qw*v1 + qz*v2)) + 2*v2*(d11*qx*(r11*v1 + 2*v2*(pxy + pzw) + 2*v3*(pxz - pyw)) - 2*d11*(pxy + pzw)*(qw*v3 - qx*v2) + 2*d22*qy*(r22*v2 + 2*v1*(pxy - pzw) + 2*v3*(pxw + pyz)) + d22*r22*(qx*v1 + 2*qy*v2 + qz*v3) + d33*qz*(r33*v3 + 2*v1*(pxz + pyw) - 2*v2*(pxw - pyz)) - 2*d33*(pxw - pyz)*(qw*v1 + qz*v2)) + 2*v3*(-d11*qw*(r11*v1 + 2*v2*(pxy + pzw) + 2*v3*(pxz - pyw)) - 2*d11*(pxz - pyw)*(qw*v3 - qx*v2) + d22*qz*(r22*v2 + 2*v1*(pxy - pzw) + 2*v3*(pxw + pyz)) + 2*d22*(pxw + pyz)*(qx*v1 + 2*qy*v2 + qz*v3) + d33*r33*(qw*v1 + qz*v2))
        F_dx[:,2] = 2*v1*(d11*r11*(qw*v2 + qx*v3) - d22*qw*(r22*v2 + 2*v1*(pxy - pzw) + 2*v3*(pxw + pyz)) - 2*d22*(pxy - pzw)*(qw*v1 - qy*v3) + d33*qx*(r33*v3 + 2*v1*(pxz + pyw) - 2*v2*(pxw - pyz)) + 2*d33*(pxz + pyw)*(qx*v1 + qy*v2 + 2*qz*v3)) + 2*v2*(d11*qw*(r11*v1 + 2*v2*(pxy + pzw) + 2*v3*(pxz - pyw)) + 2*d11*(pxy + pzw)*(qw*v2 + qx*v3) - d22*r22*(qw*v1 - qy*v3) + d33*qy*(r33*v3 + 2*v1*(pxz + pyw) - 2*v2*(pxw - pyz)) - 2*d33*(pxw - pyz)*(qx*v1 + qy*v2 + 2*qz*v3)) + 2*v3*(d11*qx*(r11*v1 + 2*v2*(pxy + pzw) + 2*v3*(pxz - pyw)) + 2*d11*(pxz - pyw)*(qw*v2 + qx*v3) + d22*qy*(r22*v2 + 2*v1*(pxy - pzw) + 2*v3*(pxw + pyz)) - 2*d22*(pxw + pyz)*(qw*v1 - qy*v3) + 2*d33*qz*(r33*v3 + 2*v1*(pxz + pyw) - 2*v2*(pxw - pyz)) + d33*r33*(qx*v1 + qy*v2 + 2*qz*v3))
        F_dx[:,3] = 2*v1*(2*d11*qw*(r11*v1 + 2*v2*(pxy + pzw) + 2*v3*(pxz - pyw)) + d11*r11*(2*qw*v1 - qy*v3 + qz*v2) - d22*qz*(r22*v2 + 2*v1*(pxy - pzw) + 2*v3*(pxw + pyz)) + 2*d22*(pxy - pzw)*(2*qw*v2 + qx*v3 - qz*v1) + d33*qy*(r33*v3 + 2*v1*(pxz + pyw) - 2*v2*(pxw - pyz)) + 2*d33*(pxz + pyw)*(2*qw*v3 - qx*v2 + qy*v1)) + 2*v2*(d11*qz*(r11*v1 + 2*v2*(pxy + pzw) + 2*v3*(pxz - pyw)) + 2*d11*(pxy + pzw)*(2*qw*v1 - qy*v3 + qz*v2) + 2*d22*qw*(r22*v2 + 2*v1*(pxy - pzw) + 2*v3*(pxw + pyz)) + d22*r22*(2*qw*v2 + qx*v3 - qz*v1) - d33*qx*(r33*v3 + 2*v1*(pxz + pyw) - 2*v2*(pxw - pyz)) - 2*d33*(pxw - pyz)*(2*qw*v3 - qx*v2 + qy*v1)) + 2*v3*(-d11*qy*(r11*v1 + 2*v2*(pxy + pzw) + 2*v3*(pxz - pyw)) + 2*d11*(pxz - pyw)*(2*qw*v1 - qy*v3 + qz*v2) + d22*qx*(r22*v2 + 2*v1*(pxy - pzw) + 2*v3*(pxw + pyz)) + 2*d22*(pxw + pyz)*(2*qw*v2 + qx*v3 - qz*v1) + 2*d33*qw*(r33*v3 + 2*v1*(pxz + pyw) - 2*v2*(pxw - pyz)) + d33*r33*(2*qw*v3 - qx*v2 + qy*v1))
        F_dx[:,4:] = 2*torch.matmul(A, -p_minus_a).squeeze(-1)

        F_dpdp = 2*A # shape (batch_size, dim(p), dim(p))

        F_dpdx = torch.zeros((batch_size, dim_p, dim_x), dtype=p.dtype, device=p.device)
        F_dpdx[:,0,0] = 16*d11*pxy*qx*v2 + 16*d11*pxz*qx*v3 - 16*d11*pyw*qx*v3 + 16*d11*pzw*qx*v2 + 16*d11*qx*r11*v1 + 4*d11*qy*r11*v2 + 4*d11*qz*r11*v3 + 8*d22*pxw*qy*v3 + 8*d22*pxy*qw*v3 + 16*d22*pxy*qy*v1 + 8*d22*pyz*qy*v3 - 8*d22*pzw*qw*v3 - 16*d22*pzw*qy*v1 + 4*d22*qy*r22*v2 - 8*d33*pxw*qz*v2 - 8*d33*pxz*qw*v2 + 16*d33*pxz*qz*v1 - 8*d33*pyw*qw*v2 + 16*d33*pyw*qz*v1 + 8*d33*pyz*qz*v2 + 4*d33*qz*r33*v3
        F_dpdx[:,0,1] = -4*d11*qw*r11*v3 + 4*d11*qx*r11*v2 + 8*d22*pxw*qx*v3 + 16*d22*pxy*qx*v1 + 16*d22*pxy*qy*v2 + 8*d22*pxy*qz*v3 + 8*d22*pyz*qx*v3 - 16*d22*pzw*qx*v1 - 16*d22*pzw*qy*v2 - 8*d22*pzw*qz*v3 + 4*d22*qx*r22*v2 - 8*d33*pxw*qw*v2 + 16*d33*pxz*qw*v1 + 8*d33*pxz*qz*v2 + 16*d33*pyw*qw*v1 + 8*d33*pyw*qz*v2 + 8*d33*pyz*qw*v2 + 4*d33*qw*r33*v3
        F_dpdx[:,0,2] = 4*d11*qw*r11*v2 + 4*d11*qx*r11*v3 - 8*d22*pxw*qw*v3 - 16*d22*pxy*qw*v1 + 8*d22*pxy*qy*v3 - 8*d22*pyz*qw*v3 + 16*d22*pzw*qw*v1 - 8*d22*pzw*qy*v3 - 4*d22*qw*r22*v2 - 8*d33*pxw*qx*v2 + 16*d33*pxz*qx*v1 + 8*d33*pxz*qy*v2 + 16*d33*pxz*qz*v3 + 16*d33*pyw*qx*v1 + 8*d33*pyw*qy*v2 + 16*d33*pyw*qz*v3 + 8*d33*pyz*qx*v2 + 4*d33*qx*r33*v3
        F_dpdx[:,0,3] = 16*d11*pxy*qw*v2 + 16*d11*pxz*qw*v3 - 16*d11*pyw*qw*v3 + 16*d11*pzw*qw*v2 + 16*d11*qw*r11*v1 - 4*d11*qy*r11*v3 + 4*d11*qz*r11*v2 - 8*d22*pxw*qz*v3 + 16*d22*pxy*qw*v2 + 8*d22*pxy*qx*v3 - 16*d22*pxy*qz*v1 - 8*d22*pyz*qz*v3 - 16*d22*pzw*qw*v2 - 8*d22*pzw*qx*v3 + 16*d22*pzw*qz*v1 - 4*d22*qz*r22*v2 - 8*d33*pxw*qy*v2 + 16*d33*pxz*qw*v3 - 8*d33*pxz*qx*v2 + 16*d33*pxz*qy*v1 + 16*d33*pyw*qw*v3 - 8*d33*pyw*qx*v2 + 16*d33*pyw*qy*v1 + 8*d33*pyz*qy*v2 + 4*d33*qy*r33*v3
        F_dpdx[:,0,4] = -2*d11*r11**2 - 8*d22*(pxy - pzw)**2 - 8*d33*(pxz + pyw)**2
        F_dpdx[:,0,5] = -4*d11*r11*(pxy + pzw) - 4*d22*r22*(pxy - pzw) + 8*d33*(pxw - pyz)*(pxz + pyw)
        F_dpdx[:,0,6] = -4*d11*r11*(pxz - pyw) - 8*d22*(pxw + pyz)*(pxy - pzw) - 4*d33*r33*(pxz + pyw)
        F_dpdx[:,1,0] = 16*d11*pxy*qx*v1 + 16*d11*pxy*qy*v2 + 8*d11*pxy*qz*v3 + 8*d11*pxz*qy*v3 - 8*d11*pyw*qy*v3 + 16*d11*pzw*qx*v1 + 16*d11*pzw*qy*v2 + 8*d11*pzw*qz*v3 + 4*d11*qy*r11*v1 + 4*d22*qw*r22*v3 + 4*d22*qy*r22*v1 + 16*d33*pxw*qw*v2 - 8*d33*pxw*qz*v1 - 8*d33*pxz*qw*v1 - 8*d33*pyw*qw*v1 - 16*d33*pyz*qw*v2 + 8*d33*pyz*qz*v1 - 4*d33*qw*r33*v3
        F_dpdx[:,1,1] = -8*d11*pxy*qw*v3 + 16*d11*pxy*qx*v2 + 8*d11*pxz*qx*v3 - 8*d11*pyw*qx*v3 - 8*d11*pzw*qw*v3 + 16*d11*pzw*qx*v2 + 4*d11*qx*r11*v1 + 16*d22*pxw*qy*v3 + 16*d22*pxy*qy*v1 + 16*d22*pyz*qy*v3 - 16*d22*pzw*qy*v1 + 4*d22*qx*r22*v1 + 16*d22*qy*r22*v2 + 4*d22*qz*r22*v3 - 8*d33*pxw*qw*v1 - 16*d33*pxw*qz*v2 + 8*d33*pxz*qz*v1 + 8*d33*pyw*qz*v1 + 8*d33*pyz*qw*v1 + 16*d33*pyz*qz*v2 + 4*d33*qz*r33*v3
        F_dpdx[:,1,2] = 16*d11*pxy*qw*v2 + 8*d11*pxy*qx*v3 + 8*d11*pxz*qw*v3 - 8*d11*pyw*qw*v3 + 16*d11*pzw*qw*v2 + 8*d11*pzw*qx*v3 + 4*d11*qw*r11*v1 - 4*d22*qw*r22*v1 + 4*d22*qy*r22*v3 - 8*d33*pxw*qx*v1 - 16*d33*pxw*qy*v2 - 16*d33*pxw*qz*v3 + 8*d33*pxz*qy*v1 + 8*d33*pyw*qy*v1 + 8*d33*pyz*qx*v1 + 16*d33*pyz*qy*v2 + 16*d33*pyz*qz*v3 + 4*d33*qy*r33*v3
        F_dpdx[:,1,3] = 16*d11*pxy*qw*v1 - 8*d11*pxy*qy*v3 + 16*d11*pxy*qz*v2 + 8*d11*pxz*qz*v3 - 8*d11*pyw*qz*v3 + 16*d11*pzw*qw*v1 - 8*d11*pzw*qy*v3 + 16*d11*pzw*qz*v2 + 4*d11*qz*r11*v1 + 16*d22*pxw*qw*v3 + 16*d22*pxy*qw*v1 + 16*d22*pyz*qw*v3 - 16*d22*pzw*qw*v1 + 16*d22*qw*r22*v2 + 4*d22*qx*r22*v3 - 4*d22*qz*r22*v1 - 16*d33*pxw*qw*v3 + 16*d33*pxw*qx*v2 - 8*d33*pxw*qy*v1 - 8*d33*pxz*qx*v1 - 8*d33*pyw*qx*v1 + 16*d33*pyz*qw*v3 - 16*d33*pyz*qx*v2 + 8*d33*pyz*qy*v1 - 4*d33*qx*r33*v3
        F_dpdx[:,1,4] = -4*d11*r11*(pxy + pzw) - 4*d22*r22*(pxy - pzw) + 8*d33*(pxw - pyz)*(pxz + pyw)
        F_dpdx[:,1,5] = -8*d11*(pxy + pzw)**2 - 2*d22*r22**2 - 8*d33*(pxw - pyz)**2
        F_dpdx[:,1,6] = -8*d11*(pxy + pzw)*(pxz - pyw) - 4*d22*r22*(pxw + pyz) + 4*d33*r33*(pxw - pyz)
        F_dpdx[:,2,0] = 8*d11*pxy*qz*v2 + 16*d11*pxz*qx*v1 + 8*d11*pxz*qy*v2 + 16*d11*pxz*qz*v3 - 16*d11*pyw*qx*v1 - 8*d11*pyw*qy*v2 - 16*d11*pyw*qz*v3 + 8*d11*pzw*qz*v2 + 4*d11*qz*r11*v1 + 16*d22*pxw*qw*v3 + 8*d22*pxw*qy*v1 + 8*d22*pxy*qw*v1 + 16*d22*pyz*qw*v3 + 8*d22*pyz*qy*v1 - 8*d22*pzw*qw*v1 + 4*d22*qw*r22*v2 - 4*d33*qw*r33*v2 + 4*d33*qz*r33*v1
        F_dpdx[:,2,1] = -8*d11*pxy*qw*v2 - 16*d11*pxz*qw*v3 + 8*d11*pxz*qx*v2 + 16*d11*pyw*qw*v3 - 8*d11*pyw*qx*v2 - 8*d11*pzw*qw*v2 - 4*d11*qw*r11*v1 + 8*d22*pxw*qx*v1 + 16*d22*pxw*qy*v2 + 16*d22*pxw*qz*v3 + 8*d22*pxy*qz*v1 + 8*d22*pyz*qx*v1 + 16*d22*pyz*qy*v2 + 16*d22*pyz*qz*v3 - 8*d22*pzw*qz*v1 + 4*d22*qz*r22*v2 + 4*d33*qw*r33*v1 + 4*d33*qz*r33*v2
        F_dpdx[:,2,2] = 8*d11*pxy*qx*v2 + 8*d11*pxz*qw*v2 + 16*d11*pxz*qx*v3 - 8*d11*pyw*qw*v2 - 16*d11*pyw*qx*v3 + 8*d11*pzw*qx*v2 + 4*d11*qx*r11*v1 - 8*d22*pxw*qw*v1 + 16*d22*pxw*qy*v3 + 8*d22*pxy*qy*v1 - 8*d22*pyz*qw*v1 + 16*d22*pyz*qy*v3 - 8*d22*pzw*qy*v1 + 4*d22*qy*r22*v2 - 16*d33*pxw*qz*v2 + 16*d33*pxz*qz*v1 + 16*d33*pyw*qz*v1 + 16*d33*pyz*qz*v2 + 4*d33*qx*r33*v1 + 4*d33*qy*r33*v2 + 16*d33*qz*r33*v3
        F_dpdx[:,2,3] = -8*d11*pxy*qy*v2 + 16*d11*pxz*qw*v1 - 16*d11*pxz*qy*v3 + 8*d11*pxz*qz*v2 - 16*d11*pyw*qw*v1 + 16*d11*pyw*qy*v3 - 8*d11*pyw*qz*v2 - 8*d11*pzw*qy*v2 - 4*d11*qy*r11*v1 + 16*d22*pxw*qw*v2 + 16*d22*pxw*qx*v3 - 8*d22*pxw*qz*v1 + 8*d22*pxy*qx*v1 + 16*d22*pyz*qw*v2 + 16*d22*pyz*qx*v3 - 8*d22*pyz*qz*v1 - 8*d22*pzw*qx*v1 + 4*d22*qx*r22*v2 - 16*d33*pxw*qw*v2 + 16*d33*pxz*qw*v1 + 16*d33*pyw*qw*v1 + 16*d33*pyz*qw*v2 + 16*d33*qw*r33*v3 - 4*d33*qx*r33*v2 + 4*d33*qy*r33*v1
        F_dpdx[:,2,4] = -4*d11*r11*(pxz - pyw) - 8*d22*(pxw + pyz)*(pxy - pzw) - 4*d33*r33*(pxz + pyw)
        F_dpdx[:,2,5] = -8*d11*(pxy + pzw)*(pxz - pyw) - 4*d22*r22*(pxw + pyz) + 4*d33*r33*(pxw - pyz)
        F_dpdx[:,2,6] = -8*d11*(pxz - pyw)**2 - 8*d22*(pxw + pyz)**2 - 2*d33*r33**2
    
        return F, F_dp, F_dx, F_dpdp, F_dpdx
    
    @staticmethod
    def prepare_gradient_and_hessian(p, a, q, D, R, A):
        """
        Compute F, dF/dp, dF/dx, d^2F/dpdp, d^2F/dpdx, d^2F/dxdx.
        p (torch.Tensor, shape (batch_size, dim(p))): input
        a (torch.Tensor, shape (batch_size, dim(p))): center of the ellipsoid
        q (torch.Tensor, shape (batch_size, dim(q))): quaternion
        D (torch.Tensor, shape (batch_size, dim(p), dim(p))): diagonal matrix
        R (torch.Tensor, shape (batch_size, dim(p), dim(p))): rotation matrix
        A (torch.Tensor, shape (batch_size, dim(p), dim(p))): A = R @ D @ R.T

        Returns:
        (torch.Tensor, shape (batch_size,)): F
        (torch.Tensor, shape (batch_size, dim(p))): dF/dp
        (torch.Tensor, shape (batch_size, dim(q)+dim(p)): dF/dx
        (torch.Tensor, shape (batch_size, dim(p), dim(p))): d^2F/dpdp
        (torch.Tensor, shape (batch_size, dim(p), dim(q)+dim(p)): d^2F/dpdx
        (torch.Tensor, shape (batch_size, dim(q)+dim(p), dim(q)+dim(p)): d^2F/dxdx
        (torch.Tensor, shape (batch_size, dim(p), dim(p), dim(p)): d^3F/dpdpdp
        (torch.Tensor, shape (batch_size, dim(p), dim(p), dim(q)+dim(p)): d^3F/dpdpdx
        (torch.Tensor, shape (batch_size, dim(p), dim(q)+dim(p), dim(q)+dim(p)): d^3F/dpdxdx
        """
        qx, qy, qz, qw = q.unbind(-1)
        d11, d22, d33 = D[:,0,0], D[:,1,1], D[:,2,2]
        r11 = R[:,0,0]
        r12 = R[:,0,1]
        r13 = R[:,0,2]
        r21 = R[:,1,0]
        r22 = R[:,1,1]
        r23 = R[:,1,2]
        r31 = R[:,2,0]
        r32 = R[:,2,1]
        r33 = R[:,2,2]
        pxx = qx*qx
        pxy = qx*qy
        pxz = qx*qz
        pxw = qx*qw
        pyy = qy*qy
        pyz = qy*qz
        pyw = qy*qw
        pzz = qz*qz
        pzw = qz*qw
        pww = qw*qw
        v1 = p[:,0]-a[:,0]
        v2 = p[:,1]-a[:,1]
        v3 = p[:,2]-a[:,2]

        batch_size = p.shape[0]
        dim_p = 3
        dim_q = 4
        dim_x = dim_p + dim_q
        p_minus_a = (p - a).unsqueeze(-1) # shape (batch_size, dim(p), 1)

        F = torch.matmul(torch.matmul(p_minus_a.transpose(-1,-2), A), p_minus_a).flatten()# (batch_size,) 

        F_dp = 2*torch.matmul(A, p_minus_a).squeeze(-1) # shape (batch_size, dim(p))

        F_dx = torch.zeros((batch_size, dim_x), dtype=p.dtype, device=p.device)
        F_dx[:,0] = 2*v1*(2*d11*qx*(r11*v1 + 2*v2*(pxy + pzw) + 2*v3*(pxz - pyw)) + d11*r11*(2*qx*v1 + qy*v2 + qz*v3) + d22*qy*(r22*v2 + 2*v1*(pxy - pzw) + 2*v3*(pxw + pyz)) + 2*d22*(pxy - pzw)*(qw*v3 + qy*v1) + d33*qz*(r33*v3 + 2*v1*(pxz + pyw) - 2*v2*(pxw - pyz)) - 2*d33*(pxz + pyw)*(qw*v2 - qz*v1)) + 2*v2*(d11*qy*(r11*v1 + 2*v2*(pxy + pzw) + 2*v3*(pxz - pyw)) + 2*d11*(pxy + pzw)*(2*qx*v1 + qy*v2 + qz*v3) + d22*r22*(qw*v3 + qy*v1) - d33*qw*(r33*v3 + 2*v1*(pxz + pyw) - 2*v2*(pxw - pyz)) + 2*d33*(pxw - pyz)*(qw*v2 - qz*v1)) + 2*v3*(d11*qz*(r11*v1 + 2*v2*(pxy + pzw) + 2*v3*(pxz - pyw)) + 2*d11*(pxz - pyw)*(2*qx*v1 + qy*v2 + qz*v3) + d22*qw*(r22*v2 + 2*v1*(pxy - pzw) + 2*v3*(pxw + pyz)) + 2*d22*(pxw + pyz)*(qw*v3 + qy*v1) - d33*r33*(qw*v2 - qz*v1))
        F_dx[:,1] = 2*v1*(-d11*r11*(qw*v3 - qx*v2) + d22*qx*(r22*v2 + 2*v1*(pxy - pzw) + 2*v3*(pxw + pyz)) + 2*d22*(pxy - pzw)*(qx*v1 + 2*qy*v2 + qz*v3) + d33*qw*(r33*v3 + 2*v1*(pxz + pyw) - 2*v2*(pxw - pyz)) + 2*d33*(pxz + pyw)*(qw*v1 + qz*v2)) + 2*v2*(d11*qx*(r11*v1 + 2*v2*(pxy + pzw) + 2*v3*(pxz - pyw)) - 2*d11*(pxy + pzw)*(qw*v3 - qx*v2) + 2*d22*qy*(r22*v2 + 2*v1*(pxy - pzw) + 2*v3*(pxw + pyz)) + d22*r22*(qx*v1 + 2*qy*v2 + qz*v3) + d33*qz*(r33*v3 + 2*v1*(pxz + pyw) - 2*v2*(pxw - pyz)) - 2*d33*(pxw - pyz)*(qw*v1 + qz*v2)) + 2*v3*(-d11*qw*(r11*v1 + 2*v2*(pxy + pzw) + 2*v3*(pxz - pyw)) - 2*d11*(pxz - pyw)*(qw*v3 - qx*v2) + d22*qz*(r22*v2 + 2*v1*(pxy - pzw) + 2*v3*(pxw + pyz)) + 2*d22*(pxw + pyz)*(qx*v1 + 2*qy*v2 + qz*v3) + d33*r33*(qw*v1 + qz*v2))
        F_dx[:,2] = 2*v1*(d11*r11*(qw*v2 + qx*v3) - d22*qw*(r22*v2 + 2*v1*(pxy - pzw) + 2*v3*(pxw + pyz)) - 2*d22*(pxy - pzw)*(qw*v1 - qy*v3) + d33*qx*(r33*v3 + 2*v1*(pxz + pyw) - 2*v2*(pxw - pyz)) + 2*d33*(pxz + pyw)*(qx*v1 + qy*v2 + 2*qz*v3)) + 2*v2*(d11*qw*(r11*v1 + 2*v2*(pxy + pzw) + 2*v3*(pxz - pyw)) + 2*d11*(pxy + pzw)*(qw*v2 + qx*v3) - d22*r22*(qw*v1 - qy*v3) + d33*qy*(r33*v3 + 2*v1*(pxz + pyw) - 2*v2*(pxw - pyz)) - 2*d33*(pxw - pyz)*(qx*v1 + qy*v2 + 2*qz*v3)) + 2*v3*(d11*qx*(r11*v1 + 2*v2*(pxy + pzw) + 2*v3*(pxz - pyw)) + 2*d11*(pxz - pyw)*(qw*v2 + qx*v3) + d22*qy*(r22*v2 + 2*v1*(pxy - pzw) + 2*v3*(pxw + pyz)) - 2*d22*(pxw + pyz)*(qw*v1 - qy*v3) + 2*d33*qz*(r33*v3 + 2*v1*(pxz + pyw) - 2*v2*(pxw - pyz)) + d33*r33*(qx*v1 + qy*v2 + 2*qz*v3))
        F_dx[:,3] = 2*v1*(2*d11*qw*(r11*v1 + 2*v2*(pxy + pzw) + 2*v3*(pxz - pyw)) + d11*r11*(2*qw*v1 - qy*v3 + qz*v2) - d22*qz*(r22*v2 + 2*v1*(pxy - pzw) + 2*v3*(pxw + pyz)) + 2*d22*(pxy - pzw)*(2*qw*v2 + qx*v3 - qz*v1) + d33*qy*(r33*v3 + 2*v1*(pxz + pyw) - 2*v2*(pxw - pyz)) + 2*d33*(pxz + pyw)*(2*qw*v3 - qx*v2 + qy*v1)) + 2*v2*(d11*qz*(r11*v1 + 2*v2*(pxy + pzw) + 2*v3*(pxz - pyw)) + 2*d11*(pxy + pzw)*(2*qw*v1 - qy*v3 + qz*v2) + 2*d22*qw*(r22*v2 + 2*v1*(pxy - pzw) + 2*v3*(pxw + pyz)) + d22*r22*(2*qw*v2 + qx*v3 - qz*v1) - d33*qx*(r33*v3 + 2*v1*(pxz + pyw) - 2*v2*(pxw - pyz)) - 2*d33*(pxw - pyz)*(2*qw*v3 - qx*v2 + qy*v1)) + 2*v3*(-d11*qy*(r11*v1 + 2*v2*(pxy + pzw) + 2*v3*(pxz - pyw)) + 2*d11*(pxz - pyw)*(2*qw*v1 - qy*v3 + qz*v2) + d22*qx*(r22*v2 + 2*v1*(pxy - pzw) + 2*v3*(pxw + pyz)) + 2*d22*(pxw + pyz)*(2*qw*v2 + qx*v3 - qz*v1) + 2*d33*qw*(r33*v3 + 2*v1*(pxz + pyw) - 2*v2*(pxw - pyz)) + d33*r33*(2*qw*v3 - qx*v2 + qy*v1))
        F_dx[:,4:] = 2*torch.matmul(A, -p_minus_a).squeeze(-1)

        F_dpdp = 2*A # shape (batch_size, dim(p), dim(p))

        F_dpdx = torch.zeros((batch_size, dim_p, dim_x), dtype=p.dtype, device=p.device)
        F_dpdx[:,0,0] = 16*d11*pxy*qx*v2 + 16*d11*pxz*qx*v3 - 16*d11*pyw*qx*v3 + 16*d11*pzw*qx*v2 + 16*d11*qx*r11*v1 + 4*d11*qy*r11*v2 + 4*d11*qz*r11*v3 + 8*d22*pxw*qy*v3 + 8*d22*pxy*qw*v3 + 16*d22*pxy*qy*v1 + 8*d22*pyz*qy*v3 - 8*d22*pzw*qw*v3 - 16*d22*pzw*qy*v1 + 4*d22*qy*r22*v2 - 8*d33*pxw*qz*v2 - 8*d33*pxz*qw*v2 + 16*d33*pxz*qz*v1 - 8*d33*pyw*qw*v2 + 16*d33*pyw*qz*v1 + 8*d33*pyz*qz*v2 + 4*d33*qz*r33*v3
        F_dpdx[:,0,1] = -4*d11*qw*r11*v3 + 4*d11*qx*r11*v2 + 8*d22*pxw*qx*v3 + 16*d22*pxy*qx*v1 + 16*d22*pxy*qy*v2 + 8*d22*pxy*qz*v3 + 8*d22*pyz*qx*v3 - 16*d22*pzw*qx*v1 - 16*d22*pzw*qy*v2 - 8*d22*pzw*qz*v3 + 4*d22*qx*r22*v2 - 8*d33*pxw*qw*v2 + 16*d33*pxz*qw*v1 + 8*d33*pxz*qz*v2 + 16*d33*pyw*qw*v1 + 8*d33*pyw*qz*v2 + 8*d33*pyz*qw*v2 + 4*d33*qw*r33*v3
        F_dpdx[:,0,2] = 4*d11*qw*r11*v2 + 4*d11*qx*r11*v3 - 8*d22*pxw*qw*v3 - 16*d22*pxy*qw*v1 + 8*d22*pxy*qy*v3 - 8*d22*pyz*qw*v3 + 16*d22*pzw*qw*v1 - 8*d22*pzw*qy*v3 - 4*d22*qw*r22*v2 - 8*d33*pxw*qx*v2 + 16*d33*pxz*qx*v1 + 8*d33*pxz*qy*v2 + 16*d33*pxz*qz*v3 + 16*d33*pyw*qx*v1 + 8*d33*pyw*qy*v2 + 16*d33*pyw*qz*v3 + 8*d33*pyz*qx*v2 + 4*d33*qx*r33*v3
        F_dpdx[:,0,3] = 16*d11*pxy*qw*v2 + 16*d11*pxz*qw*v3 - 16*d11*pyw*qw*v3 + 16*d11*pzw*qw*v2 + 16*d11*qw*r11*v1 - 4*d11*qy*r11*v3 + 4*d11*qz*r11*v2 - 8*d22*pxw*qz*v3 + 16*d22*pxy*qw*v2 + 8*d22*pxy*qx*v3 - 16*d22*pxy*qz*v1 - 8*d22*pyz*qz*v3 - 16*d22*pzw*qw*v2 - 8*d22*pzw*qx*v3 + 16*d22*pzw*qz*v1 - 4*d22*qz*r22*v2 - 8*d33*pxw*qy*v2 + 16*d33*pxz*qw*v3 - 8*d33*pxz*qx*v2 + 16*d33*pxz*qy*v1 + 16*d33*pyw*qw*v3 - 8*d33*pyw*qx*v2 + 16*d33*pyw*qy*v1 + 8*d33*pyz*qy*v2 + 4*d33*qy*r33*v3
        F_dpdx[:,0,4] = -2*d11*r11**2 - 8*d22*(pxy - pzw)**2 - 8*d33*(pxz + pyw)**2
        F_dpdx[:,0,5] = -4*d11*r11*(pxy + pzw) - 4*d22*r22*(pxy - pzw) + 8*d33*(pxw - pyz)*(pxz + pyw)
        F_dpdx[:,0,6] = -4*d11*r11*(pxz - pyw) - 8*d22*(pxw + pyz)*(pxy - pzw) - 4*d33*r33*(pxz + pyw)
        F_dpdx[:,1,0] = 16*d11*pxy*qx*v1 + 16*d11*pxy*qy*v2 + 8*d11*pxy*qz*v3 + 8*d11*pxz*qy*v3 - 8*d11*pyw*qy*v3 + 16*d11*pzw*qx*v1 + 16*d11*pzw*qy*v2 + 8*d11*pzw*qz*v3 + 4*d11*qy*r11*v1 + 4*d22*qw*r22*v3 + 4*d22*qy*r22*v1 + 16*d33*pxw*qw*v2 - 8*d33*pxw*qz*v1 - 8*d33*pxz*qw*v1 - 8*d33*pyw*qw*v1 - 16*d33*pyz*qw*v2 + 8*d33*pyz*qz*v1 - 4*d33*qw*r33*v3
        F_dpdx[:,1,1] = -8*d11*pxy*qw*v3 + 16*d11*pxy*qx*v2 + 8*d11*pxz*qx*v3 - 8*d11*pyw*qx*v3 - 8*d11*pzw*qw*v3 + 16*d11*pzw*qx*v2 + 4*d11*qx*r11*v1 + 16*d22*pxw*qy*v3 + 16*d22*pxy*qy*v1 + 16*d22*pyz*qy*v3 - 16*d22*pzw*qy*v1 + 4*d22*qx*r22*v1 + 16*d22*qy*r22*v2 + 4*d22*qz*r22*v3 - 8*d33*pxw*qw*v1 - 16*d33*pxw*qz*v2 + 8*d33*pxz*qz*v1 + 8*d33*pyw*qz*v1 + 8*d33*pyz*qw*v1 + 16*d33*pyz*qz*v2 + 4*d33*qz*r33*v3
        F_dpdx[:,1,2] = 16*d11*pxy*qw*v2 + 8*d11*pxy*qx*v3 + 8*d11*pxz*qw*v3 - 8*d11*pyw*qw*v3 + 16*d11*pzw*qw*v2 + 8*d11*pzw*qx*v3 + 4*d11*qw*r11*v1 - 4*d22*qw*r22*v1 + 4*d22*qy*r22*v3 - 8*d33*pxw*qx*v1 - 16*d33*pxw*qy*v2 - 16*d33*pxw*qz*v3 + 8*d33*pxz*qy*v1 + 8*d33*pyw*qy*v1 + 8*d33*pyz*qx*v1 + 16*d33*pyz*qy*v2 + 16*d33*pyz*qz*v3 + 4*d33*qy*r33*v3
        F_dpdx[:,1,3] = 16*d11*pxy*qw*v1 - 8*d11*pxy*qy*v3 + 16*d11*pxy*qz*v2 + 8*d11*pxz*qz*v3 - 8*d11*pyw*qz*v3 + 16*d11*pzw*qw*v1 - 8*d11*pzw*qy*v3 + 16*d11*pzw*qz*v2 + 4*d11*qz*r11*v1 + 16*d22*pxw*qw*v3 + 16*d22*pxy*qw*v1 + 16*d22*pyz*qw*v3 - 16*d22*pzw*qw*v1 + 16*d22*qw*r22*v2 + 4*d22*qx*r22*v3 - 4*d22*qz*r22*v1 - 16*d33*pxw*qw*v3 + 16*d33*pxw*qx*v2 - 8*d33*pxw*qy*v1 - 8*d33*pxz*qx*v1 - 8*d33*pyw*qx*v1 + 16*d33*pyz*qw*v3 - 16*d33*pyz*qx*v2 + 8*d33*pyz*qy*v1 - 4*d33*qx*r33*v3
        F_dpdx[:,1,4] = -4*d11*r11*(pxy + pzw) - 4*d22*r22*(pxy - pzw) + 8*d33*(pxw - pyz)*(pxz + pyw)
        F_dpdx[:,1,5] = -8*d11*(pxy + pzw)**2 - 2*d22*r22**2 - 8*d33*(pxw - pyz)**2
        F_dpdx[:,1,6] = -8*d11*(pxy + pzw)*(pxz - pyw) - 4*d22*r22*(pxw + pyz) + 4*d33*r33*(pxw - pyz)
        F_dpdx[:,2,0] = 8*d11*pxy*qz*v2 + 16*d11*pxz*qx*v1 + 8*d11*pxz*qy*v2 + 16*d11*pxz*qz*v3 - 16*d11*pyw*qx*v1 - 8*d11*pyw*qy*v2 - 16*d11*pyw*qz*v3 + 8*d11*pzw*qz*v2 + 4*d11*qz*r11*v1 + 16*d22*pxw*qw*v3 + 8*d22*pxw*qy*v1 + 8*d22*pxy*qw*v1 + 16*d22*pyz*qw*v3 + 8*d22*pyz*qy*v1 - 8*d22*pzw*qw*v1 + 4*d22*qw*r22*v2 - 4*d33*qw*r33*v2 + 4*d33*qz*r33*v1
        F_dpdx[:,2,1] = -8*d11*pxy*qw*v2 - 16*d11*pxz*qw*v3 + 8*d11*pxz*qx*v2 + 16*d11*pyw*qw*v3 - 8*d11*pyw*qx*v2 - 8*d11*pzw*qw*v2 - 4*d11*qw*r11*v1 + 8*d22*pxw*qx*v1 + 16*d22*pxw*qy*v2 + 16*d22*pxw*qz*v3 + 8*d22*pxy*qz*v1 + 8*d22*pyz*qx*v1 + 16*d22*pyz*qy*v2 + 16*d22*pyz*qz*v3 - 8*d22*pzw*qz*v1 + 4*d22*qz*r22*v2 + 4*d33*qw*r33*v1 + 4*d33*qz*r33*v2
        F_dpdx[:,2,2] = 8*d11*pxy*qx*v2 + 8*d11*pxz*qw*v2 + 16*d11*pxz*qx*v3 - 8*d11*pyw*qw*v2 - 16*d11*pyw*qx*v3 + 8*d11*pzw*qx*v2 + 4*d11*qx*r11*v1 - 8*d22*pxw*qw*v1 + 16*d22*pxw*qy*v3 + 8*d22*pxy*qy*v1 - 8*d22*pyz*qw*v1 + 16*d22*pyz*qy*v3 - 8*d22*pzw*qy*v1 + 4*d22*qy*r22*v2 - 16*d33*pxw*qz*v2 + 16*d33*pxz*qz*v1 + 16*d33*pyw*qz*v1 + 16*d33*pyz*qz*v2 + 4*d33*qx*r33*v1 + 4*d33*qy*r33*v2 + 16*d33*qz*r33*v3
        F_dpdx[:,2,3] = -8*d11*pxy*qy*v2 + 16*d11*pxz*qw*v1 - 16*d11*pxz*qy*v3 + 8*d11*pxz*qz*v2 - 16*d11*pyw*qw*v1 + 16*d11*pyw*qy*v3 - 8*d11*pyw*qz*v2 - 8*d11*pzw*qy*v2 - 4*d11*qy*r11*v1 + 16*d22*pxw*qw*v2 + 16*d22*pxw*qx*v3 - 8*d22*pxw*qz*v1 + 8*d22*pxy*qx*v1 + 16*d22*pyz*qw*v2 + 16*d22*pyz*qx*v3 - 8*d22*pyz*qz*v1 - 8*d22*pzw*qx*v1 + 4*d22*qx*r22*v2 - 16*d33*pxw*qw*v2 + 16*d33*pxz*qw*v1 + 16*d33*pyw*qw*v1 + 16*d33*pyz*qw*v2 + 16*d33*qw*r33*v3 - 4*d33*qx*r33*v2 + 4*d33*qy*r33*v1
        F_dpdx[:,2,4] = -4*d11*r11*(pxz - pyw) - 8*d22*(pxw + pyz)*(pxy - pzw) - 4*d33*r33*(pxz + pyw)
        F_dpdx[:,2,5] = -8*d11*(pxy + pzw)*(pxz - pyw) - 4*d22*r22*(pxw + pyz) + 4*d33*r33*(pxw - pyz)
        F_dpdx[:,2,6] = -8*d11*(pxz - pyw)**2 - 8*d22*(pxw + pyz)**2 - 2*d33*r33**2

        F_dxdx = torch.zeros((batch_size, dim_x, dim_x), dtype=p.dtype, device=p.device)
        F_dxdx[:,0,0] = 4*v1*(4*d11*qx*(2*qx*v1 + qy*v2 + qz*v3) + d11*r11*v1 + d11*(r11*v1 + 2*v2*(pxy + pzw) + 2*v3*(pxz - pyw)) + 2*d22*qy*(qw*v3 + qy*v1) - 2*d33*qz*(qw*v2 - qz*v1)) + 8*v2*(d11*qy*(2*qx*v1 + qy*v2 + qz*v3) + d11*v1*(pxy + pzw) + d33*qw*(qw*v2 - qz*v1)) + 8*v3*(d11*qz*(2*qx*v1 + qy*v2 + qz*v3) + d11*v1*(pxz - pyw) + d22*qw*(qw*v3 + qy*v1))
        F_dxdx[:,0,1] = 8*d11*pxy*v2**2 + 8*d11*pxz*v2*v3 - 8*d11*pyw*v2*v3 + 8*d11*pzw*v2**2 - 16*d11*qw*qx*v1*v3 - 8*d11*qw*qy*v2*v3 - 8*d11*qw*qz*v3**2 + 16*d11*qx**2*v1*v2 + 8*d11*qx*qy*v2**2 + 8*d11*qx*qz*v2*v3 + 4*d11*r11*v1*v2 + 8*d22*pxw*v1*v3 + 8*d22*pxy*v1**2 + 8*d22*pyz*v1*v3 - 8*d22*pzw*v1**2 + 8*d22*qw*qx*v1*v3 + 16*d22*qw*qy*v2*v3 + 8*d22*qw*qz*v3**2 + 8*d22*qx*qy*v1**2 + 16*d22*qy**2*v1*v2 + 8*d22*qy*qz*v1*v3 + 4*d22*r22*v1*v2 - 8*d33*qw**2*v1*v2 + 8*d33*qw*qz*v1**2 - 8*d33*qw*qz*v2**2 + 8*d33*qz**2*v1*v2
        F_dxdx[:,0,2] = 8*d11*pxy*v2*v3 + 8*d11*pxz*v3**2 - 8*d11*pyw*v3**2 + 8*d11*pzw*v2*v3 + 16*d11*qw*qx*v1*v2 + 8*d11*qw*qy*v2**2 + 8*d11*qw*qz*v2*v3 + 16*d11*qx**2*v1*v3 + 8*d11*qx*qy*v2*v3 + 8*d11*qx*qz*v3**2 + 4*d11*r11*v1*v3 - 8*d22*qw**2*v1*v3 - 8*d22*qw*qy*v1**2 + 8*d22*qw*qy*v3**2 + 8*d22*qy**2*v1*v3 - 8*d33*pxw*v1*v2 + 8*d33*pxz*v1**2 + 8*d33*pyw*v1**2 + 8*d33*pyz*v1*v2 - 8*d33*qw*qx*v1*v2 - 8*d33*qw*qy*v2**2 - 16*d33*qw*qz*v2*v3 + 8*d33*qx*qz*v1**2 + 8*d33*qy*qz*v1*v2 + 16*d33*qz**2*v1*v3 + 4*d33*r33*v1*v3
        F_dxdx[:,0,3] = 32*d11*qw*qx*v1**2 + 16*d11*qw*qy*v1*v2 + 16*d11*qw*qz*v1*v3 - 16*d11*qx*qy*v1*v3 + 16*d11*qx*qz*v1*v2 - 8*d11*qy**2*v2*v3 + 8*d11*qy*qz*v2**2 - 8*d11*qy*qz*v3**2 + 8*d11*qz**2*v2*v3 + 8*d22*pxw*v3**2 + 8*d22*pxy*v1*v3 + 8*d22*pyz*v3**2 - 8*d22*pzw*v1*v3 + 16*d22*qw**2*v2*v3 + 8*d22*qw*qx*v3**2 + 16*d22*qw*qy*v1*v2 - 8*d22*qw*qz*v1*v3 + 8*d22*qx*qy*v1*v3 - 8*d22*qy*qz*v1**2 + 4*d22*r22*v2*v3 + 8*d33*pxw*v2**2 - 8*d33*pxz*v1*v2 - 8*d33*pyw*v1*v2 - 8*d33*pyz*v2**2 - 16*d33*qw**2*v2*v3 + 8*d33*qw*qx*v2**2 - 8*d33*qw*qy*v1*v2 + 16*d33*qw*qz*v1*v3 - 8*d33*qx*qz*v1*v2 + 8*d33*qy*qz*v1**2 - 4*d33*r33*v2*v3
        F_dxdx[:,0,4] = -16*d11*pxy*qx*v2 - 16*d11*pxz*qx*v3 + 16*d11*pyw*qx*v3 - 16*d11*pzw*qx*v2 - 16*d11*qx*r11*v1 - 4*d11*qy*r11*v2 - 4*d11*qz*r11*v3 - 8*d22*pxw*qy*v3 - 8*d22*pxy*qw*v3 - 16*d22*pxy*qy*v1 - 8*d22*pyz*qy*v3 + 8*d22*pzw*qw*v3 + 16*d22*pzw*qy*v1 - 4*d22*qy*r22*v2 + 8*d33*pxw*qz*v2 + 8*d33*pxz*qw*v2 - 16*d33*pxz*qz*v1 + 8*d33*pyw*qw*v2 - 16*d33*pyw*qz*v1 - 8*d33*pyz*qz*v2 - 4*d33*qz*r33*v3
        F_dxdx[:,0,5] = -16*d11*pxy*qx*v1 - 16*d11*pxy*qy*v2 - 8*d11*pxy*qz*v3 - 8*d11*pxz*qy*v3 + 8*d11*pyw*qy*v3 - 16*d11*pzw*qx*v1 - 16*d11*pzw*qy*v2 - 8*d11*pzw*qz*v3 - 4*d11*qy*r11*v1 - 4*d22*qw*r22*v3 - 4*d22*qy*r22*v1 - 16*d33*pxw*qw*v2 + 8*d33*pxw*qz*v1 + 8*d33*pxz*qw*v1 + 8*d33*pyw*qw*v1 + 16*d33*pyz*qw*v2 - 8*d33*pyz*qz*v1 + 4*d33*qw*r33*v3
        F_dxdx[:,0,6] = -8*d11*pxy*qz*v2 - 16*d11*pxz*qx*v1 - 8*d11*pxz*qy*v2 - 16*d11*pxz*qz*v3 + 16*d11*pyw*qx*v1 + 8*d11*pyw*qy*v2 + 16*d11*pyw*qz*v3 - 8*d11*pzw*qz*v2 - 4*d11*qz*r11*v1 - 16*d22*pxw*qw*v3 - 8*d22*pxw*qy*v1 - 8*d22*pxy*qw*v1 - 16*d22*pyz*qw*v3 - 8*d22*pyz*qy*v1 + 8*d22*pzw*qw*v1 - 4*d22*qw*r22*v2 + 4*d33*qw*r33*v2 - 4*d33*qz*r33*v1
        F_dxdx[:,1,1] = 8*v1*(d22*qx*(qx*v1 + 2*qy*v2 + qz*v3) + d22*v2*(pxy - pzw) + d33*qw*(qw*v1 + qz*v2)) + 4*v2*(-2*d11*qx*(qw*v3 - qx*v2) + 4*d22*qy*(qx*v1 + 2*qy*v2 + qz*v3) + d22*r22*v2 + d22*(r22*v2 + 2*v1*(pxy - pzw) + 2*v3*(pxw + pyz)) + 2*d33*qz*(qw*v1 + qz*v2)) + 8*v3*(d11*qw*(qw*v3 - qx*v2) + d22*qz*(qx*v1 + 2*qy*v2 + qz*v3) + d22*v2*(pxw + pyz))
        F_dxdx[:,1,2] = -8*d11*qw**2*v2*v3 + 8*d11*qw*qx*v2**2 - 8*d11*qw*qx*v3**2 + 8*d11*qx**2*v2*v3 + 8*d22*pxw*v3**2 + 8*d22*pxy*v1*v3 + 8*d22*pyz*v3**2 - 8*d22*pzw*v1*v3 - 8*d22*qw*qx*v1**2 - 16*d22*qw*qy*v1*v2 - 8*d22*qw*qz*v1*v3 + 8*d22*qx*qy*v1*v3 + 16*d22*qy**2*v2*v3 + 8*d22*qy*qz*v3**2 + 4*d22*r22*v2*v3 - 8*d33*pxw*v2**2 + 8*d33*pxz*v1*v2 + 8*d33*pyw*v1*v2 + 8*d33*pyz*v2**2 + 8*d33*qw*qx*v1**2 + 8*d33*qw*qy*v1*v2 + 16*d33*qw*qz*v1*v3 + 8*d33*qx*qz*v1*v2 + 8*d33*qy*qz*v2**2 + 16*d33*qz**2*v2*v3 + 4*d33*r33*v2*v3
        F_dxdx[:,1,3] = -8*d11*pxy*v2*v3 - 8*d11*pxz*v3**2 + 8*d11*pyw*v3**2 - 8*d11*pzw*v2*v3 - 16*d11*qw**2*v1*v3 + 16*d11*qw*qx*v1*v2 + 8*d11*qw*qy*v3**2 - 8*d11*qw*qz*v2*v3 - 8*d11*qx*qy*v2*v3 + 8*d11*qx*qz*v2**2 - 4*d11*r11*v1*v3 + 16*d22*qw*qx*v1*v2 + 32*d22*qw*qy*v2**2 + 16*d22*qw*qz*v2*v3 + 8*d22*qx**2*v1*v3 + 16*d22*qx*qy*v2*v3 - 8*d22*qx*qz*v1**2 + 8*d22*qx*qz*v3**2 - 16*d22*qy*qz*v1*v2 - 8*d22*qz**2*v1*v3 - 8*d33*pxw*v1*v2 + 8*d33*pxz*v1**2 + 8*d33*pyw*v1**2 + 8*d33*pyz*v1*v2 + 16*d33*qw**2*v1*v3 - 8*d33*qw*qx*v1*v2 + 8*d33*qw*qy*v1**2 + 16*d33*qw*qz*v2*v3 - 8*d33*qx*qz*v2**2 + 8*d33*qy*qz*v1*v2 + 4*d33*r33*v1*v3
        F_dxdx[:,1,4] = 4*d11*qw*r11*v3 - 4*d11*qx*r11*v2 - 8*d22*pxw*qx*v3 - 16*d22*pxy*qx*v1 - 16*d22*pxy*qy*v2 - 8*d22*pxy*qz*v3 - 8*d22*pyz*qx*v3 + 16*d22*pzw*qx*v1 + 16*d22*pzw*qy*v2 + 8*d22*pzw*qz*v3 - 4*d22*qx*r22*v2 + 8*d33*pxw*qw*v2 - 16*d33*pxz*qw*v1 - 8*d33*pxz*qz*v2 - 16*d33*pyw*qw*v1 - 8*d33*pyw*qz*v2 - 8*d33*pyz*qw*v2 - 4*d33*qw*r33*v3
        F_dxdx[:,1,5] = 8*d11*pxy*qw*v3 - 16*d11*pxy*qx*v2 - 8*d11*pxz*qx*v3 + 8*d11*pyw*qx*v3 + 8*d11*pzw*qw*v3 - 16*d11*pzw*qx*v2 - 4*d11*qx*r11*v1 - 16*d22*pxw*qy*v3 - 16*d22*pxy*qy*v1 - 16*d22*pyz*qy*v3 + 16*d22*pzw*qy*v1 - 4*d22*qx*r22*v1 - 16*d22*qy*r22*v2 - 4*d22*qz*r22*v3 + 8*d33*pxw*qw*v1 + 16*d33*pxw*qz*v2 - 8*d33*pxz*qz*v1 - 8*d33*pyw*qz*v1 - 8*d33*pyz*qw*v1 - 16*d33*pyz*qz*v2 - 4*d33*qz*r33*v3
        F_dxdx[:,1,6] = 8*d11*pxy*qw*v2 + 16*d11*pxz*qw*v3 - 8*d11*pxz*qx*v2 - 16*d11*pyw*qw*v3 + 8*d11*pyw*qx*v2 + 8*d11*pzw*qw*v2 + 4*d11*qw*r11*v1 - 8*d22*pxw*qx*v1 - 16*d22*pxw*qy*v2 - 16*d22*pxw*qz*v3 - 8*d22*pxy*qz*v1 - 8*d22*pyz*qx*v1 - 16*d22*pyz*qy*v2 - 16*d22*pyz*qz*v3 + 8*d22*pzw*qz*v1 - 4*d22*qz*r22*v2 - 4*d33*qw*r33*v1 - 4*d33*qz*r33*v2
        F_dxdx[:,2,2] = 8*v1*(d22*qw*(qw*v1 - qy*v3) + d33*qx*(qx*v1 + qy*v2 + 2*qz*v3) + d33*v3*(pxz + pyw)) + 8*v2*(d11*qw*(qw*v2 + qx*v3) + d33*qy*(qx*v1 + qy*v2 + 2*qz*v3) - d33*v3*(pxw - pyz)) + 4*v3*(2*d11*qx*(qw*v2 + qx*v3) - 2*d22*qy*(qw*v1 - qy*v3) + 4*d33*qz*(qx*v1 + qy*v2 + 2*qz*v3) + d33*r33*v3 + d33*(r33*v3 + 2*v1*(pxz + pyw) - 2*v2*(pxw - pyz)))
        F_dxdx[:,2,3] = 8*d11*pxy*v2**2 + 8*d11*pxz*v2*v3 - 8*d11*pyw*v2*v3 + 8*d11*pzw*v2**2 + 16*d11*qw**2*v1*v2 + 16*d11*qw*qx*v1*v3 - 8*d11*qw*qy*v2*v3 + 8*d11*qw*qz*v2**2 - 8*d11*qx*qy*v3**2 + 8*d11*qx*qz*v2*v3 + 4*d11*r11*v1*v2 - 8*d22*pxw*v1*v3 - 8*d22*pxy*v1**2 - 8*d22*pyz*v1*v3 + 8*d22*pzw*v1**2 - 16*d22*qw**2*v1*v2 - 8*d22*qw*qx*v1*v3 + 16*d22*qw*qy*v2*v3 + 8*d22*qw*qz*v1**2 + 8*d22*qx*qy*v3**2 - 8*d22*qy*qz*v1*v3 - 4*d22*r22*v1*v2 + 16*d33*qw*qx*v1*v3 + 16*d33*qw*qy*v2*v3 + 32*d33*qw*qz*v3**2 - 8*d33*qx**2*v1*v2 + 8*d33*qx*qy*v1**2 - 8*d33*qx*qy*v2**2 - 16*d33*qx*qz*v2*v3 + 8*d33*qy**2*v1*v2 + 16*d33*qy*qz*v1*v3
        F_dxdx[:,2,4] = -4*d11*qw*r11*v2 - 4*d11*qx*r11*v3 + 8*d22*pxw*qw*v3 + 16*d22*pxy*qw*v1 - 8*d22*pxy*qy*v3 + 8*d22*pyz*qw*v3 - 16*d22*pzw*qw*v1 + 8*d22*pzw*qy*v3 + 4*d22*qw*r22*v2 + 8*d33*pxw*qx*v2 - 16*d33*pxz*qx*v1 - 8*d33*pxz*qy*v2 - 16*d33*pxz*qz*v3 - 16*d33*pyw*qx*v1 - 8*d33*pyw*qy*v2 - 16*d33*pyw*qz*v3 - 8*d33*pyz*qx*v2 - 4*d33*qx*r33*v3
        F_dxdx[:,2,5] = -16*d11*pxy*qw*v2 - 8*d11*pxy*qx*v3 - 8*d11*pxz*qw*v3 + 8*d11*pyw*qw*v3 - 16*d11*pzw*qw*v2 - 8*d11*pzw*qx*v3 - 4*d11*qw*r11*v1 + 4*d22*qw*r22*v1 - 4*d22*qy*r22*v3 + 8*d33*pxw*qx*v1 + 16*d33*pxw*qy*v2 + 16*d33*pxw*qz*v3 - 8*d33*pxz*qy*v1 - 8*d33*pyw*qy*v1 - 8*d33*pyz*qx*v1 - 16*d33*pyz*qy*v2 - 16*d33*pyz*qz*v3 - 4*d33*qy*r33*v3
        F_dxdx[:,2,6] = -8*d11*pxy*qx*v2 - 8*d11*pxz*qw*v2 - 16*d11*pxz*qx*v3 + 8*d11*pyw*qw*v2 + 16*d11*pyw*qx*v3 - 8*d11*pzw*qx*v2 - 4*d11*qx*r11*v1 + 8*d22*pxw*qw*v1 - 16*d22*pxw*qy*v3 - 8*d22*pxy*qy*v1 + 8*d22*pyz*qw*v1 - 16*d22*pyz*qy*v3 + 8*d22*pzw*qy*v1 - 4*d22*qy*r22*v2 + 16*d33*pxw*qz*v2 - 16*d33*pxz*qz*v1 - 16*d33*pyw*qz*v1 - 16*d33*pyz*qz*v2 - 4*d33*qx*r33*v1 - 4*d33*qy*r33*v2 - 16*d33*qz*r33*v3
        F_dxdx[:,3,3] = 4*v1*(4*d11*qw*(2*qw*v1 - qy*v3 + qz*v2) + d11*r11*v1 + d11*(r11*v1 + 2*v2*(pxy + pzw) + 2*v3*(pxz - pyw)) - 2*d22*qz*(2*qw*v2 + qx*v3 - qz*v1) + 2*d22*v2*(pxy - pzw) + 2*d33*qy*(2*qw*v3 - qx*v2 + qy*v1) + 2*d33*v3*(pxz + pyw)) + 4*v2*(2*d11*qz*(2*qw*v1 - qy*v3 + qz*v2) + 2*d11*v1*(pxy + pzw) + 4*d22*qw*(2*qw*v2 + qx*v3 - qz*v1) + d22*r22*v2 + d22*(r22*v2 + 2*v1*(pxy - pzw) + 2*v3*(pxw + pyz)) - 2*d33*qx*(2*qw*v3 - qx*v2 + qy*v1) - 2*d33*v3*(pxw - pyz)) + 4*v3*(-2*d11*qy*(2*qw*v1 - qy*v3 + qz*v2) + 2*d11*v1*(pxz - pyw) + 2*d22*qx*(2*qw*v2 + qx*v3 - qz*v1) + 2*d22*v2*(pxw + pyz) + 4*d33*qw*(2*qw*v3 - qx*v2 + qy*v1) + d33*r33*v3 + d33*(r33*v3 + 2*v1*(pxz + pyw) - 2*v2*(pxw - pyz)))
        F_dxdx[:,3,4] = -16*d11*pxy*qw*v2 - 16*d11*pxz*qw*v3 + 16*d11*pyw*qw*v3 - 16*d11*pzw*qw*v2 - 16*d11*qw*r11*v1 + 4*d11*qy*r11*v3 - 4*d11*qz*r11*v2 + 8*d22*pxw*qz*v3 - 16*d22*pxy*qw*v2 - 8*d22*pxy*qx*v3 + 16*d22*pxy*qz*v1 + 8*d22*pyz*qz*v3 + 16*d22*pzw*qw*v2 + 8*d22*pzw*qx*v3 - 16*d22*pzw*qz*v1 + 4*d22*qz*r22*v2 + 8*d33*pxw*qy*v2 - 16*d33*pxz*qw*v3 + 8*d33*pxz*qx*v2 - 16*d33*pxz*qy*v1 - 16*d33*pyw*qw*v3 + 8*d33*pyw*qx*v2 - 16*d33*pyw*qy*v1 - 8*d33*pyz*qy*v2 - 4*d33*qy*r33*v3
        F_dxdx[:,3,5] = -16*d11*pxy*qw*v1 + 8*d11*pxy*qy*v3 - 16*d11*pxy*qz*v2 - 8*d11*pxz*qz*v3 + 8*d11*pyw*qz*v3 - 16*d11*pzw*qw*v1 + 8*d11*pzw*qy*v3 - 16*d11*pzw*qz*v2 - 4*d11*qz*r11*v1 - 16*d22*pxw*qw*v3 - 16*d22*pxy*qw*v1 - 16*d22*pyz*qw*v3 + 16*d22*pzw*qw*v1 - 16*d22*qw*r22*v2 - 4*d22*qx*r22*v3 + 4*d22*qz*r22*v1 + 16*d33*pxw*qw*v3 - 16*d33*pxw*qx*v2 + 8*d33*pxw*qy*v1 + 8*d33*pxz*qx*v1 + 8*d33*pyw*qx*v1 - 16*d33*pyz*qw*v3 + 16*d33*pyz*qx*v2 - 8*d33*pyz*qy*v1 + 4*d33*qx*r33*v3
        F_dxdx[:,3,6] = 8*d11*pxy*qy*v2 - 16*d11*pxz*qw*v1 + 16*d11*pxz*qy*v3 - 8*d11*pxz*qz*v2 + 16*d11*pyw*qw*v1 - 16*d11*pyw*qy*v3 + 8*d11*pyw*qz*v2 + 8*d11*pzw*qy*v2 + 4*d11*qy*r11*v1 - 16*d22*pxw*qw*v2 - 16*d22*pxw*qx*v3 + 8*d22*pxw*qz*v1 - 8*d22*pxy*qx*v1 - 16*d22*pyz*qw*v2 - 16*d22*pyz*qx*v3 + 8*d22*pyz*qz*v1 + 8*d22*pzw*qx*v1 - 4*d22*qx*r22*v2 + 16*d33*pxw*qw*v2 - 16*d33*pxz*qw*v1 - 16*d33*pyw*qw*v1 - 16*d33*pyz*qw*v2 - 16*d33*qw*r33*v3 + 4*d33*qx*r33*v2 - 4*d33*qy*r33*v1
        F_dxdx[:,4,4] = 2*d11*r11**2 + 8*d22*(pxy - pzw)**2 + 8*d33*(pxz + pyw)**2
        F_dxdx[:,4,5] = 4*d11*r11*(pxy + pzw) + 4*d22*r22*(pxy - pzw) - 8*d33*(pxw - pyz)*(pxz + pyw)
        F_dxdx[:,4,6] = 4*d11*r11*(pxz - pyw) + 8*d22*(pxw + pyz)*(pxy - pzw) + 4*d33*r33*(pxz + pyw)
        F_dxdx[:,5,5] = 8*d11*(pxy + pzw)**2 + 2*d22*r22**2 + 8*d33*(pxw - pyz)**2
        F_dxdx[:,5,6] = 8*d11*(pxy + pzw)*(pxz - pyw) + 4*d22*r22*(pxw + pyz) - 4*d33*r33*(pxw - pyz)
        F_dxdx[:,6,6] = 8*d11*(pxz - pyw)**2 + 8*d22*(pxw + pyz)**2 + 2*d33*r33**2
        for i in range(7):
            for j in range(i+1,7):
                F_dxdx[:,j,i] = F_dxdx[:,i,j]
    
        F_dpdpdp = torch.zeros((batch_size, dim_p, dim_p, dim_p), dtype=p.dtype, device=p.device)

        F_dpdpdx = torch.zeros((batch_size, dim_p, dim_p, dim_x), dtype=p.dtype, device=p.device)
        F_dpdpdx[:,0,0,0] = 16*d11*qx*r11 + 16*d22*qy*(pxy - pzw) + 16*d33*qz*(pxz + pyw)
        F_dpdpdx[:,0,0,1] = 16*d22*qx*(pxy - pzw) + 16*d33*qw*(pxz + pyw)
        F_dpdpdx[:,0,0,2] = -16*d22*qw*(pxy - pzw) + 16*d33*qx*(pxz + pyw)
        F_dpdpdx[:,0,0,3] = 16*d11*qw*r11 - 16*d22*qz*(pxy - pzw) + 16*d33*qy*(pxz + pyw)
        F_dpdpdx[:,0,1,0] = 16*d11*qx*(pxy + pzw) + 4*d11*qy*r11 + 4*d22*qy*r22 - 8*d33*qw*(pxz + pyw) - 8*d33*qz*(pxw - pyz)
        F_dpdpdx[:,0,1,1] = 4*d11*qx*r11 + 4*d22*qx*r22 + 16*d22*qy*(pxy - pzw) - 8*d33*qw*(pxw - pyz) + 8*d33*qz*(pxz + pyw)
        F_dpdpdx[:,0,1,2] = 4*d11*qw*r11 - 4*d22*qw*r22 - 8*d33*qx*(pxw - pyz) + 8*d33*qy*(pxz + pyw)
        F_dpdpdx[:,0,1,3] = 16*d11*qw*(pxy + pzw) + 4*d11*qz*r11 + 16*d22*qw*(pxy - pzw) - 4*d22*qz*r22 - 8*d33*qx*(pxz + pyw) - 8*d33*qy*(pxw - pyz)
        F_dpdpdx[:,0,2,0] = 16*d11*qx*(pxz - pyw) + 4*d11*qz*r11 + 8*d22*qw*(pxy - pzw) + 8*d22*qy*(pxw + pyz) + 4*d33*qz*r33
        F_dpdpdx[:,0,2,1] = -4*d11*qw*r11 + 8*d22*qx*(pxw + pyz) + 8*d22*qz*(pxy - pzw) + 4*d33*qw*r33
        F_dpdpdx[:,0,2,2] = 4*d11*qx*r11 - 8*d22*qw*(pxw + pyz) + 8*d22*qy*(pxy - pzw) + 4*d33*qx*r33 + 16*d33*qz*(pxz + pyw)
        F_dpdpdx[:,0,2,3] = 16*d11*qw*(pxz - pyw) - 4*d11*qy*r11 + 8*d22*qx*(pxy - pzw) - 8*d22*qz*(pxw + pyz) + 16*d33*qw*(pxz + pyw) + 4*d33*qy*r33
        F_dpdpdx[:,1,1,0] = 16*d11*qy*(pxy + pzw) + 16*d33*qw*(pxw - pyz)
        F_dpdpdx[:,1,1,1] = 16*d11*qx*(pxy + pzw) + 16*d22*qy*r22 - 16*d33*qz*(pxw - pyz)
        F_dpdpdx[:,1,1,2] = 16*d11*qw*(pxy + pzw) - 16*d33*qy*(pxw - pyz)
        F_dpdpdx[:,1,1,3] = 16*d11*qz*(pxy + pzw) + 16*d22*qw*r22 + 16*d33*qx*(pxw - pyz)
        F_dpdpdx[:,1,2,0] = 8*d11*qy*(pxz - pyw) + 8*d11*qz*(pxy + pzw) + 4*d22*qw*r22 - 4*d33*qw*r33
        F_dpdpdx[:,1,2,1] = -8*d11*qw*(pxy + pzw) + 8*d11*qx*(pxz - pyw) + 16*d22*qy*(pxw + pyz) + 4*d22*qz*r22 + 4*d33*qz*r33
        F_dpdpdx[:,1,2,2] = 8*d11*qw*(pxz - pyw) + 8*d11*qx*(pxy + pzw) + 4*d22*qy*r22 + 4*d33*qy*r33 - 16*d33*qz*(pxw - pyz)
        F_dpdpdx[:,1,2,3] = -8*d11*qy*(pxy + pzw) + 8*d11*qz*(pxz - pyw) + 16*d22*qw*(pxw + pyz) + 4*d22*qx*r22 - 16*d33*qw*(pxw - pyz) - 4*d33*qx*r33
        F_dpdpdx[:,2,2,0] = 16*d11*qz*(pxz - pyw) + 16*d22*qw*(pxw + pyz)
        F_dpdpdx[:,2,2,1] = -16*d11*qw*(pxz - pyw) + 16*d22*qz*(pxw + pyz)
        F_dpdpdx[:,2,2,2] = 16*d11*qx*(pxz - pyw) + 16*d22*qy*(pxw + pyz) + 16*d33*qz*r33
        F_dpdpdx[:,2,2,3] = -16*d11*qy*(pxz - pyw) + 16*d22*qx*(pxw + pyz) + 16*d33*qw*r33
        for i in range(F_dpdpdx.shape[1]):
            for j in range(i+1, F_dpdpdx.shape[2]):
                F_dpdpdx[:,j,i,:] = F_dpdpdx[:,i,j,:]

        F_dpdxdx = torch.zeros((batch_size, dim_p, dim_x, dim_x), dtype=p.dtype, device=p.device)
    
        F_dpdxdx[:,0,0,0] = 32*d11*pxx*v1 + 32*d11*pxy*v2 + 32*d11*pxz*v3 - 16*d11*pyw*v3 + 16*d11*pzw*v2 + 32*d11*qx**2*v1 + 16*d11*qx*qy*v2 + 16*d11*qx*qz*v3 + 16*d11*r11*v1 + 8*d22*pyw*v3 + 8*d22*pyy*v1 + 8*d22*qw*qy*v3 + 8*d22*qy**2*v1 - 8*d33*pzw*v2 + 8*d33*pzz*v1 - 8*d33*qw*qz*v2 + 8*d33*qz**2*v1
        F_dpdxdx[:,0,0,1] = -8*d11*pxw*v3 + 8*d11*pxx*v2 - 8*d11*qw*qx*v3 + 8*d11*qx**2*v2 + 4*d11*r11*v2 + 12*d22*pxw*v3 + 24*d22*pxy*v1 + 8*d22*pyy*v2 + 12*d22*pyz*v3 - 16*d22*pzw*v1 + 4*d22*qw*qx*v3 + 8*d22*qx*qy*v1 + 8*d22*qy**2*v2 + 4*d22*qy*qz*v3 + 4*d22*r22*v2 - 4*d33*pww*v2 + 8*d33*pzw*v1 + 4*d33*pzz*v2 - 4*d33*qw**2*v2 + 8*d33*qw*qz*v1 + 4*d33*qz**2*v2
        F_dpdxdx[:,0,0,2] = 8*d11*pxw*v2 + 8*d11*pxx*v3 + 8*d11*qw*qx*v2 + 8*d11*qx**2*v3 + 4*d11*r11*v3 - 4*d22*pww*v3 - 8*d22*pyw*v1 + 4*d22*pyy*v3 - 4*d22*qw**2*v3 - 8*d22*qw*qy*v1 + 4*d22*qy**2*v3 - 12*d33*pxw*v2 + 24*d33*pxz*v1 + 16*d33*pyw*v1 + 12*d33*pyz*v2 + 8*d33*pzz*v3 - 4*d33*qw*qx*v2 + 8*d33*qx*qz*v1 + 4*d33*qy*qz*v2 + 8*d33*qz**2*v3 + 4*d33*r33*v3
        F_dpdxdx[:,0,0,3] = 8*d11*qw*(2*qx*v1 + qy*v2 + qz*v3) + 8*d11*qx*(2*qw*v1 - qy*v3 + qz*v2) + 4*d22*qy*(2*qw*v2 + qx*v3 - qz*v1) - 4*d22*qz*(qw*v3 + qy*v1) + 4*d22*v3*(pxy - pzw) - 4*d33*qy*(qw*v2 - qz*v1) + 4*d33*qz*(2*qw*v3 - qx*v2 + qy*v1) - 4*d33*v2*(pxz + pyw) + 8*v1*(4*d11*pxw - d22*pyz + d33*pyz) + 4*v2*(2*d11*pxz + 2*d11*pyw + 2*d22*pyw - d33*pxz - d33*pyw - d33*(pxz + pyw)) + 4*v3*(-2*d11*pxy + 2*d11*pzw + d22*pxy - d22*pzw + d22*(pxy - pzw) + 2*d33*pzw)
        F_dpdxdx[:,0,0,4] = -16*d11*qx*r11 - 16*d22*qy*(pxy - pzw) - 16*d33*qz*(pxz + pyw)
        F_dpdxdx[:,0,0,5] = -16*d11*qx*(pxy + pzw) - 4*d11*qy*r11 - 4*d22*qy*r22 + 8*d33*qw*(pxz + pyw) + 8*d33*qz*(pxw - pyz)
        F_dpdxdx[:,0,0,6] = -16*d11*qx*(pxz - pyw) - 4*d11*qz*r11 - 8*d22*qw*(pxy - pzw) - 8*d22*qy*(pxw + pyz) - 4*d33*qz*r33
        F_dpdxdx[:,0,1,1] = 8*d22*pxz*v3 + 8*d22*qx*(qx*v1 + 2*qy*v2 + qz*v3) + 8*d22*v2*(pxy - pzw) + 8*d33*qw*(qw*v1 + qz*v2) + 8*v1*(d22*pxx + d33*pww) + 8*v2*(2*d22*pxy + d22*(pxy - pzw) + d33*pzw)
        F_dpdxdx[:,0,1,2] = -4*d22*qw*(qx*v1 + 2*qy*v2 + qz*v3) - 4*d22*qx*(qw*v1 - qy*v3) + 4*d22*v3*(pxy - pzw) + 4*d33*qw*(qx*v1 + qy*v2 + 2*qz*v3) + 4*d33*qx*(qw*v1 + qz*v2) + 4*d33*v2*(pxz + pyw) - 8*pxw*v1*(d22 - d33) + 4*v2*(-2*d22*pyw + d33*pxz + d33*pyw + d33*(pxz + pyw)) + 4*v3*(d22*pxy - d22*pzw + d22*(pxy - pzw) + 2*d33*pzw)
        F_dpdxdx[:,0,1,3] = -8*d11*pww*v3 + 8*d11*pxw*v2 - 8*d11*qw**2*v3 + 8*d11*qw*qx*v2 - 4*d11*r11*v3 + 8*d22*pxw*v2 + 4*d22*pxx*v3 - 8*d22*pxz*v1 - 8*d22*pyz*v2 - 4*d22*pzz*v3 + 8*d22*qw*qx*v2 + 4*d22*qx**2*v3 - 8*d22*qx*qz*v1 - 8*d22*qy*qz*v2 - 4*d22*qz**2*v3 + 8*d33*pww*v3 - 12*d33*pxw*v2 + 16*d33*pxz*v1 + 24*d33*pyw*v1 + 12*d33*pyz*v2 + 8*d33*qw**2*v3 - 4*d33*qw*qx*v2 + 8*d33*qw*qy*v1 + 4*d33*qy*qz*v2 + 4*d33*r33*v3
        F_dpdxdx[:,0,1,4] = -16*d22*qx*(pxy - pzw) - 16*d33*qw*(pxz + pyw)
        F_dpdxdx[:,0,1,5] = -4*d11*qx*r11 - 4*d22*qx*r22 - 16*d22*qy*(pxy - pzw) + 8*d33*qw*(pxw - pyz) - 8*d33*qz*(pxz + pyw)
        F_dpdxdx[:,0,1,6] = 4*d11*qw*r11 - 8*d22*qx*(pxw + pyz) - 8*d22*qz*(pxy - pzw) - 4*d33*qw*r33
        F_dpdxdx[:,0,2,2] = 8*d22*qw*(qw*v1 - qy*v3) + 8*d33*pxy*v2 + 8*d33*qx*(qx*v1 + qy*v2 + 2*qz*v3) + 8*d33*v3*(pxz + pyw) + 8*v1*(d22*pww + d33*pxx) + 8*v3*(-d22*pyw + 2*d33*pxz + d33*(pxz + pyw))
        F_dpdxdx[:,0,2,3] = 8*d11*pww*v2 + 8*d11*pxw*v3 + 8*d11*qw**2*v2 + 8*d11*qw*qx*v3 + 4*d11*r11*v2 - 8*d22*pww*v2 - 12*d22*pxw*v3 - 16*d22*pxy*v1 - 12*d22*pyz*v3 + 24*d22*pzw*v1 - 8*d22*qw**2*v2 - 4*d22*qw*qx*v3 + 8*d22*qw*qz*v1 - 4*d22*qy*qz*v3 - 4*d22*r22*v2 + 8*d33*pxw*v3 - 4*d33*pxx*v2 + 8*d33*pxy*v1 + 4*d33*pyy*v2 + 8*d33*pyz*v3 + 8*d33*qw*qx*v3 - 4*d33*qx**2*v2 + 8*d33*qx*qy*v1 + 4*d33*qy**2*v2 + 8*d33*qy*qz*v3
        F_dpdxdx[:,0,2,4] = 16*d22*qw*(pxy - pzw) - 16*d33*qx*(pxz + pyw)
        F_dpdxdx[:,0,2,5] = -4*d11*qw*r11 + 4*d22*qw*r22 + 8*d33*qx*(pxw - pyz) - 8*d33*qy*(pxz + pyw)
        F_dpdxdx[:,0,2,6] = -4*d11*qx*r11 + 8*d22*qw*(pxw + pyz) - 8*d22*qy*(pxy - pzw) - 4*d33*qx*r33 - 16*d33*qz*(pxz + pyw)
        F_dpdxdx[:,0,3,3] = 32*d11*pww*v1 + 16*d11*pxy*v2 + 16*d11*pxz*v3 - 32*d11*pyw*v3 + 32*d11*pzw*v2 + 32*d11*qw**2*v1 - 16*d11*qw*qy*v3 + 16*d11*qw*qz*v2 + 16*d11*r11*v1 + 16*d22*pxy*v2 - 8*d22*pxz*v3 - 32*d22*pzw*v2 + 8*d22*pzz*v1 - 16*d22*qw*qz*v2 - 8*d22*qx*qz*v3 + 8*d22*qz**2*v1 - 8*d33*pxy*v2 + 16*d33*pxz*v3 + 32*d33*pyw*v3 + 8*d33*pyy*v1 + 16*d33*qw*qy*v3 - 8*d33*qx*qy*v2 + 8*d33*qy**2*v1
        F_dpdxdx[:,0,3,4] = -16*d11*qw*r11 + 16*d22*qz*(pxy - pzw) - 16*d33*qy*(pxz + pyw)
        F_dpdxdx[:,0,3,5] = -16*d11*qw*(pxy + pzw) - 4*d11*qz*r11 - 16*d22*qw*(pxy - pzw) + 4*d22*qz*r22 + 8*d33*qx*(pxz + pyw) + 8*d33*qy*(pxw - pyz)
        F_dpdxdx[:,0,3,6] = -16*d11*qw*(pxz - pyw) + 4*d11*qy*r11 - 8*d22*qx*(pxy - pzw) + 8*d22*qz*(pxw + pyz) - 16*d33*qw*(pxz + pyw) - 4*d33*qy*r33
        F_dpdxdx[:,1,0,0] = 8*d11*pyz*v3 + 8*d11*qy*(2*qx*v1 + qy*v2 + qz*v3) + 8*d11*v1*(pxy + pzw) + 8*d33*qw*(qw*v2 - qz*v1) + 8*v1*(2*d11*pxy + d11*(pxy + pzw) - d33*pzw) + 8*v2*(d11*pyy + d33*pww)
        F_dpdxdx[:,1,0,1] = 8*d11*pxx*v1 + 24*d11*pxy*v2 + 12*d11*pxz*v3 - 12*d11*pyw*v3 + 16*d11*pzw*v2 - 4*d11*qw*qy*v3 + 8*d11*qx**2*v1 + 8*d11*qx*qy*v2 + 4*d11*qx*qz*v3 + 4*d11*r11*v1 + 8*d22*pyw*v3 + 8*d22*pyy*v1 + 8*d22*qw*qy*v3 + 8*d22*qy**2*v1 + 4*d22*r22*v1 - 4*d33*pww*v1 - 8*d33*pzw*v2 + 4*d33*pzz*v1 - 4*d33*qw**2*v1 - 8*d33*qw*qz*v2 + 4*d33*qz**2*v1
        F_dpdxdx[:,1,0,2] = 4*d11*qw*(2*qx*v1 + qy*v2 + qz*v3) + 4*d11*qy*(qw*v2 + qx*v3) + 4*d11*v3*(pxy + pzw) - 4*d33*qw*(qx*v1 + qy*v2 + 2*qz*v3) - 4*d33*qy*(qw*v2 - qz*v1) - 4*d33*v1*(pxw - pyz) + 8*pyw*v2*(d11 - d33) + 4*v1*(2*d11*pxw - d33*pxw + d33*pyz - d33*(pxw - pyz)) + 4*v3*(d11*pxy + d11*pzw + d11*(pxy + pzw) - 2*d33*pzw)
        F_dpdxdx[:,1,0,3] = 8*d11*pxz*v1 + 8*d11*pyw*v1 - 4*d11*pyy*v3 + 8*d11*pyz*v2 + 4*d11*pzz*v3 + 8*d11*qw*qy*v1 + 8*d11*qx*qz*v1 - 4*d11*qy**2*v3 + 8*d11*qy*qz*v2 + 4*d11*qz**2*v3 + 8*d22*pww*v3 + 8*d22*pyw*v1 + 8*d22*qw**2*v3 + 8*d22*qw*qy*v1 + 4*d22*r22*v3 - 8*d33*pww*v3 + 24*d33*pxw*v2 - 12*d33*pxz*v1 - 12*d33*pyw*v1 - 16*d33*pyz*v2 - 8*d33*qw**2*v3 + 8*d33*qw*qx*v2 - 4*d33*qw*qy*v1 - 4*d33*qx*qz*v1 - 4*d33*r33*v3
        F_dpdxdx[:,1,0,4] = -16*d11*qx*(pxy + pzw) - 4*d11*qy*r11 - 4*d22*qy*r22 + 8*d33*qw*(pxz + pyw) + 8*d33*qz*(pxw - pyz)
        F_dpdxdx[:,1,0,5] = -16*d11*qy*(pxy + pzw) - 16*d33*qw*(pxw - pyz)
        F_dpdxdx[:,1,0,6] = -8*d11*qy*(pxz - pyw) - 8*d11*qz*(pxy + pzw) - 4*d22*qw*r22 + 4*d33*qw*r33
        F_dpdxdx[:,1,1,1] = -8*d11*pxw*v3 + 8*d11*pxx*v2 - 8*d11*qw*qx*v3 + 8*d11*qx**2*v2 + 16*d22*pxw*v3 + 32*d22*pxy*v1 + 32*d22*pyy*v2 + 32*d22*pyz*v3 - 16*d22*pzw*v1 + 16*d22*qx*qy*v1 + 32*d22*qy**2*v2 + 16*d22*qy*qz*v3 + 16*d22*r22*v2 + 8*d33*pzw*v1 + 8*d33*pzz*v2 + 8*d33*qw*qz*v1 + 8*d33*qz**2*v2
        F_dpdxdx[:,1,1,2] = -4*d11*pww*v3 + 8*d11*pxw*v2 + 4*d11*pxx*v3 - 4*d11*qw**2*v3 + 8*d11*qw*qx*v2 + 4*d11*qx**2*v3 - 8*d22*pyw*v1 + 8*d22*pyy*v3 - 8*d22*qw*qy*v1 + 8*d22*qy**2*v3 + 4*d22*r22*v3 - 16*d33*pxw*v2 + 12*d33*pxz*v1 + 12*d33*pyw*v1 + 24*d33*pyz*v2 + 8*d33*pzz*v3 + 4*d33*qw*qy*v1 + 4*d33*qx*qz*v1 + 8*d33*qy*qz*v2 + 8*d33*qz**2*v3 + 4*d33*r33*v3
        F_dpdxdx[:,1,1,3] = 4*d11*qx*(2*qw*v1 - qy*v3 + qz*v2) - 4*d11*qz*(qw*v3 - qx*v2) - 4*d11*v3*(pxy + pzw) + 8*d22*qw*(qx*v1 + 2*qy*v2 + qz*v3) + 8*d22*qy*(2*qw*v2 + qx*v3 - qz*v1) - 4*d33*qx*(qw*v1 + qz*v2) + 4*d33*qz*(2*qw*v3 - qx*v2 + qy*v1) - 4*d33*v1*(pxw - pyz) + 4*v1*(2*d11*pxw + 2*d22*pxw - 2*d22*pyz - d33*pxw + d33*pyz - d33*(pxw - pyz)) + 8*v2*(d11*pxz + 4*d22*pyw - d33*pxz) - 4*v3*(d11*pxy + d11*pzw + d11*(pxy + pzw) - 2*d22*pxy - 2*d22*pzw - 2*d33*pzw)
        F_dpdxdx[:,1,1,4] = -4*d11*qx*r11 - 4*d22*qx*r22 - 16*d22*qy*(pxy - pzw) + 8*d33*qw*(pxw - pyz) - 8*d33*qz*(pxz + pyw)
        F_dpdxdx[:,1,1,5] = -16*d11*qx*(pxy + pzw) - 16*d22*qy*r22 + 16*d33*qz*(pxw - pyz)
        F_dpdxdx[:,1,1,6] = 8*d11*qw*(pxy + pzw) - 8*d11*qx*(pxz - pyw) - 16*d22*qy*(pxw + pyz) - 4*d22*qz*r22 - 4*d33*qz*r33
        F_dpdxdx[:,1,2,2] = 8*d11*qw*(qw*v2 + qx*v3) + 8*d33*pxy*v1 + 8*d33*qy*(qx*v1 + qy*v2 + 2*qz*v3) - 8*d33*v3*(pxw - pyz) + 8*v2*(d11*pww + d33*pyy) + 8*v3*(d11*pxw + 2*d33*pyz - d33*(pxw - pyz))
        F_dpdxdx[:,1,2,3] = 8*d11*pww*v1 + 16*d11*pxy*v2 + 12*d11*pxz*v3 - 12*d11*pyw*v3 + 24*d11*pzw*v2 + 8*d11*qw**2*v1 - 4*d11*qw*qy*v3 + 8*d11*qw*qz*v2 + 4*d11*qx*qz*v3 + 4*d11*r11*v1 - 8*d22*pww*v1 + 8*d22*pyw*v3 - 8*d22*qw**2*v1 + 8*d22*qw*qy*v3 - 4*d22*r22*v1 - 4*d33*pxx*v1 - 8*d33*pxy*v2 - 8*d33*pxz*v3 + 8*d33*pyw*v3 + 4*d33*pyy*v1 + 8*d33*qw*qy*v3 - 4*d33*qx**2*v1 - 8*d33*qx*qy*v2 - 8*d33*qx*qz*v3 + 4*d33*qy**2*v1
        F_dpdxdx[:,1,2,4] = -4*d11*qw*r11 + 4*d22*qw*r22 + 8*d33*qx*(pxw - pyz) - 8*d33*qy*(pxz + pyw)
        F_dpdxdx[:,1,2,5] = -16*d11*qw*(pxy + pzw) + 16*d33*qy*(pxw - pyz)
        F_dpdxdx[:,1,2,6] = -8*d11*qw*(pxz - pyw) - 8*d11*qx*(pxy + pzw) - 4*d22*qy*r22 - 4*d33*qy*r33 + 16*d33*qz*(pxw - pyz)
        F_dpdxdx[:,1,3,3] = 16*d11*pxy*v1 - 8*d11*pyz*v3 + 32*d11*pzw*v1 + 8*d11*pzz*v2 + 16*d11*qw*qz*v1 - 8*d11*qy*qz*v3 + 8*d11*qz**2*v2 + 32*d22*pww*v2 + 32*d22*pxw*v3 + 16*d22*pxy*v1 + 16*d22*pyz*v3 - 32*d22*pzw*v1 + 32*d22*qw**2*v2 + 16*d22*qw*qx*v3 - 16*d22*qw*qz*v1 + 16*d22*r22*v2 - 32*d33*pxw*v3 + 8*d33*pxx*v2 - 8*d33*pxy*v1 + 16*d33*pyz*v3 - 16*d33*qw*qx*v3 + 8*d33*qx**2*v2 - 8*d33*qx*qy*v1
        F_dpdxdx[:,1,3,4] = -16*d11*qw*(pxy + pzw) - 4*d11*qz*r11 - 16*d22*qw*(pxy - pzw) + 4*d22*qz*r22 + 8*d33*qx*(pxz + pyw) + 8*d33*qy*(pxw - pyz)
        F_dpdxdx[:,1,3,5] = -16*d11*qz*(pxy + pzw) - 16*d22*qw*r22 - 16*d33*qx*(pxw - pyz)
        F_dpdxdx[:,1,3,6] = 8*d11*qy*(pxy + pzw) - 8*d11*qz*(pxz - pyw) - 16*d22*qw*(pxw + pyz) - 4*d22*qx*r22 + 16*d33*qw*(pxw - pyz) + 4*d33*qx*r33
        F_dpdxdx[:,2,0,0] = 8*d11*pyz*v2 + 8*d11*qz*(2*qx*v1 + qy*v2 + qz*v3) + 8*d11*v1*(pxz - pyw) + 8*d22*qw*(qw*v3 + qy*v1) + 8*v1*(2*d11*pxz + d11*(pxz - pyw) + d22*pyw) + 8*v3*(d11*pzz + d22*pww)
        F_dpdxdx[:,2,0,1] = -4*d11*qw*(2*qx*v1 + qy*v2 + qz*v3) - 4*d11*qz*(qw*v3 - qx*v2) + 4*d11*v2*(pxz - pyw) + 4*d22*qw*(qx*v1 + 2*qy*v2 + qz*v3) + 4*d22*qz*(qw*v3 + qy*v1) + 4*d22*v1*(pxw + pyz) - 8*pzw*v3*(d11 - d22) + 4*v1*(-2*d11*pxw + d22*pxw + d22*pyz + d22*(pxw + pyz)) + 4*v2*(d11*pxz - d11*pyw + d11*(pxz - pyw) + 2*d22*pyw)
        F_dpdxdx[:,2,0,2] = 8*d11*pxx*v1 + 12*d11*pxy*v2 + 24*d11*pxz*v3 - 16*d11*pyw*v3 + 12*d11*pzw*v2 + 4*d11*qw*qz*v2 + 8*d11*qx**2*v1 + 4*d11*qx*qy*v2 + 8*d11*qx*qz*v3 + 4*d11*r11*v1 - 4*d22*pww*v1 + 8*d22*pyw*v3 + 4*d22*pyy*v1 - 4*d22*qw**2*v1 + 8*d22*qw*qy*v3 + 4*d22*qy**2*v1 - 8*d33*pzw*v2 + 8*d33*pzz*v1 - 8*d33*qw*qz*v2 + 8*d33*qz**2*v1 + 4*d33*r33*v1
        F_dpdxdx[:,2,0,3] = -8*d11*pxy*v1 - 4*d11*pyy*v2 - 8*d11*pyz*v3 + 8*d11*pzw*v1 + 4*d11*pzz*v2 + 8*d11*qw*qz*v1 - 8*d11*qx*qy*v1 - 4*d11*qy**2*v2 - 8*d11*qy*qz*v3 + 4*d11*qz**2*v2 + 8*d22*pww*v2 + 24*d22*pxw*v3 + 12*d22*pxy*v1 + 16*d22*pyz*v3 - 12*d22*pzw*v1 + 8*d22*qw**2*v2 + 8*d22*qw*qx*v3 - 4*d22*qw*qz*v1 + 4*d22*qx*qy*v1 + 4*d22*r22*v2 - 8*d33*pww*v2 + 8*d33*pzw*v1 - 8*d33*qw**2*v2 + 8*d33*qw*qz*v1 - 4*d33*r33*v2
        F_dpdxdx[:,2,0,4] = -16*d11*qx*(pxz - pyw) - 4*d11*qz*r11 - 8*d22*qw*(pxy - pzw) - 8*d22*qy*(pxw + pyz) - 4*d33*qz*r33
        F_dpdxdx[:,2,0,5] = -8*d11*qy*(pxz - pyw) - 8*d11*qz*(pxy + pzw) - 4*d22*qw*r22 + 4*d33*qw*r33
        F_dpdxdx[:,2,0,6] = -16*d11*qz*(pxz - pyw) - 16*d22*qw*(pxw + pyz)
        F_dpdxdx[:,2,1,1] = 8*d11*qw*(qw*v3 - qx*v2) + 8*d22*pxz*v1 + 8*d22*qz*(qx*v1 + 2*qy*v2 + qz*v3) + 8*d22*v2*(pxw + pyz) + 8*v2*(-d11*pxw + 2*d22*pyz + d22*(pxw + pyz)) + 8*v3*(d11*pww + d22*pzz)
        F_dpdxdx[:,2,1,2] = -4*d11*pww*v2 - 8*d11*pxw*v3 + 4*d11*pxx*v2 - 4*d11*qw**2*v2 - 8*d11*qw*qx*v3 + 4*d11*qx**2*v2 + 16*d22*pxw*v3 + 12*d22*pxy*v1 + 8*d22*pyy*v2 + 24*d22*pyz*v3 - 12*d22*pzw*v1 - 4*d22*qw*qz*v1 + 4*d22*qx*qy*v1 + 8*d22*qy**2*v2 + 8*d22*qy*qz*v3 + 4*d22*r22*v2 + 8*d33*pzw*v1 + 8*d33*pzz*v2 + 8*d33*qw*qz*v1 + 8*d33*qz**2*v2 + 4*d33*r33*v2
        F_dpdxdx[:,2,1,3] = -8*d11*pww*v1 - 12*d11*pxy*v2 - 16*d11*pxz*v3 + 24*d11*pyw*v3 - 12*d11*pzw*v2 - 8*d11*qw**2*v1 + 8*d11*qw*qy*v3 - 4*d11*qw*qz*v2 - 4*d11*qx*qy*v2 - 4*d11*r11*v1 + 4*d22*pxx*v1 + 8*d22*pxy*v2 + 8*d22*pxz*v3 + 8*d22*pzw*v2 - 4*d22*pzz*v1 + 8*d22*qw*qz*v2 + 4*d22*qx**2*v1 + 8*d22*qx*qy*v2 + 8*d22*qx*qz*v3 - 4*d22*qz**2*v1 + 8*d33*pww*v1 + 8*d33*pzw*v2 + 8*d33*qw**2*v1 + 8*d33*qw*qz*v2 + 4*d33*r33*v1
        F_dpdxdx[:,2,1,4] = 4*d11*qw*r11 - 8*d22*qx*(pxw + pyz) - 8*d22*qz*(pxy - pzw) - 4*d33*qw*r33
        F_dpdxdx[:,2,1,5] = 8*d11*qw*(pxy + pzw) - 8*d11*qx*(pxz - pyw) - 16*d22*qy*(pxw + pyz) - 4*d22*qz*r22 - 4*d33*qz*r33
        F_dpdxdx[:,2,1,6] = 16*d11*qw*(pxz - pyw) - 16*d22*qz*(pxw + pyz)
        F_dpdxdx[:,2,2,2] = 8*d11*pxw*v2 + 8*d11*pxx*v3 + 8*d11*qw*qx*v2 + 8*d11*qx**2*v3 - 8*d22*pyw*v1 + 8*d22*pyy*v3 - 8*d22*qw*qy*v1 + 8*d22*qy**2*v3 - 16*d33*pxw*v2 + 32*d33*pxz*v1 + 16*d33*pyw*v1 + 32*d33*pyz*v2 + 32*d33*pzz*v3 + 16*d33*qx*qz*v1 + 16*d33*qy*qz*v2 + 32*d33*qz**2*v3 + 16*d33*r33*v3
        F_dpdxdx[:,2,2,3] = 4*d11*qx*(2*qw*v1 - qy*v3 + qz*v2) - 4*d11*qy*(qw*v2 + qx*v3) + 4*d11*v2*(pxz - pyw) - 4*d22*qx*(qw*v1 - qy*v3) + 4*d22*qy*(2*qw*v2 + qx*v3 - qz*v1) - 4*d22*v1*(pxw + pyz) + 8*d33*qw*(qx*v1 + qy*v2 + 2*qz*v3) + 8*d33*qz*(2*qw*v3 - qx*v2 + qy*v1) + 4*v1*(2*d11*pxw - d22*pxw - d22*pyz - d22*(pxw + pyz) + 2*d33*pxw + 2*d33*pyz) + 4*v2*(d11*pxz - d11*pyw + d11*(pxz - pyw) + 2*d22*pyw - 2*d33*pxz + 2*d33*pyw) + 8*v3*(-d11*pxy + d22*pxy + 4*d33*pzw)
        F_dpdxdx[:,2,2,4] = -4*d11*qx*r11 + 8*d22*qw*(pxw + pyz) - 8*d22*qy*(pxy - pzw) - 4*d33*qx*r33 - 16*d33*qz*(pxz + pyw)
        F_dpdxdx[:,2,2,5] = -8*d11*qw*(pxz - pyw) - 8*d11*qx*(pxy + pzw) - 4*d22*qy*r22 - 4*d33*qy*r33 + 16*d33*qz*(pxw - pyz)
        F_dpdxdx[:,2,2,6] = -16*d11*qx*(pxz - pyw) - 16*d22*qy*(pxw + pyz) - 16*d33*qz*r33
        F_dpdxdx[:,2,3,3] = 16*d11*pxz*v1 - 32*d11*pyw*v1 + 8*d11*pyy*v3 - 8*d11*pyz*v2 - 16*d11*qw*qy*v1 + 8*d11*qy**2*v3 - 8*d11*qy*qz*v2 + 32*d22*pxw*v2 + 8*d22*pxx*v3 - 8*d22*pxz*v1 + 16*d22*pyz*v2 + 16*d22*qw*qx*v2 + 8*d22*qx**2*v3 - 8*d22*qx*qz*v1 + 32*d33*pww*v3 - 32*d33*pxw*v2 + 16*d33*pxz*v1 + 32*d33*pyw*v1 + 16*d33*pyz*v2 + 32*d33*qw**2*v3 - 16*d33*qw*qx*v2 + 16*d33*qw*qy*v1 + 16*d33*r33*v3
        F_dpdxdx[:,2,3,4] = -16*d11*qw*(pxz - pyw) + 4*d11*qy*r11 - 8*d22*qx*(pxy - pzw) + 8*d22*qz*(pxw + pyz) - 16*d33*qw*(pxz + pyw) - 4*d33*qy*r33
        F_dpdxdx[:,2,3,5] = 8*d11*qy*(pxy + pzw) - 8*d11*qz*(pxz - pyw) - 16*d22*qw*(pxw + pyz) - 4*d22*qx*r22 + 16*d33*qw*(pxw - pyz) + 4*d33*qx*r33
        F_dpdxdx[:,2,3,6] = 16*d11*qy*(pxz - pyw) - 16*d22*qx*(pxw + pyz) - 16*d33*qw*r33
        for i in range(F_dpdxdx.shape[2]):
            for j in range(i+1, F_dpdxdx.shape[3]):
                F_dpdxdx[:,:,j,i] = F_dpdxdx[:,:,i,j]
    
        return F, F_dp, F_dx, F_dpdp, F_dpdx, F_dxdx, F_dpdpdp, F_dpdpdx, F_dpdxdx