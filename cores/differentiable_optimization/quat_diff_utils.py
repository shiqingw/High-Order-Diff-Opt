import torch

class Quaternion_RDRT():
    """
    Let q = [qx, qy, qz, qw]. Define the rotation matrix R as
    R = [[2*(qw**2+qx**2)-1, 2*(qx*qy-qw*qz), 2*(qx*qz+qw*qy)],
        [2*(qx*qy+qw*qz), 2*(qw**2+qy**2)-1, 2*(qy*qz-qw*qx)],
        [2*(qx*qz-qw*qy), 2*(qy*qz+qw*qx), 2*(qw**2+qz**2)-1]].
    Let the diagonal matrix D be D = [[a, 0, 0], [0, b, 0], [0, 0, c]]. Then,
    the product M = R*D*R^T is a symmetric matrix, and we can use this class 
    to calculate the Jacobian and Hessian of each element M w.r.t. q.
    """
    @staticmethod
    def RDRT_dq(q, D):
        """
        Calculate the derivative of M.flatten() w.r.t. q.
        q (torch.Tensor, shape (batch_size, 4)): input
        D (torch.Tensor, shape (batch_size, 3, 3)): diagonal matrix

        Returns:
        (torch.Tensor, shape (batch_size, 9, 4)): dM/dq
        """
        qx, qy, qz, qw = q.unbind(-1)
        a, b, c = D[:,0,0], D[:,1,1], D[:,2,2]
        r11 = 2*(qw**2+qx**2)-1
        r12 = 2*(qx*qy-qw*qz)
        r13 = 2*(qx*qz+qw*qy)
        r21 = 2*(qx*qy+qw*qz)
        r22 = 2*(qw**2+qy**2)-1
        r23 = 2*(qy*qz-qw*qx)
        r31 = 2*(qx*qz-qw*qy)
        r32 = 2*(qy*qz+qw*qx)
        r33 = 2*(qw**2+qz**2)-1
        M11_dq = torch.stack([8*a*qx*r11 + 4*b*qy*r12 + 4*c*qz*r13,
                              4*b*qx*r12 + 4*c*qw*r13,
                              -4*b*qw*r12 + 4*c*qx*r13,
                              8*a*qw*r11 - 4*b*qz*r12 + 4*c*qy*r13], dim=1)
        
        M12_dq = torch.stack([4*a*qx*r21 + 2*a*qy*r11 + 2*b*qy*r22 - 2*c*qw*r13 + 2*c*qz*r23,
                              2*a*qx*r11 + 2*b*qx*r22 + 4*b*qy*r12 + 2*c*qw*r23 + 2*c*qz*r13,
                              2*a*qw*r11 - 2*b*qw*r22 + 2*c*qx*r23 + 2*c*qy*r13,
                              4*a*qw*r21 + 2*a*qz*r11 + 4*b*qw*r12 - 2*b*qz*r22 - 2*c*qx*r13 + 2*c*qy*r23], dim=1)
        
        M13_dq = torch.stack([4*a*qx*r31 + 2*a*qz*r11 + 2*b*qw*r12 + 2*b*qy*r32 + 2*c*qz*r33,
                              -2*a*qw*r11 + 2*b*qx*r32 + 2*b*qz*r12 + 2*c*qw*r33,
                              2*a*qx*r11 - 2*b*qw*r32 + 2*b*qy*r12 + 2*c*qx*r33 + 4*c*qz*r13,
                              4*a*qw*r31 - 2*a*qy*r11 + 2*b*qx*r12 - 2*b*qz*r32 + 4*c*qw*r13 + 2*c*qy*r33], dim=1)
        
        M22_dq = torch.stack([4*a*qy*r21 - 4*c*qw*r23,
                              4*a*qx*r21 + 8*b*qy*r22 + 4*c*qz*r23,
                              4*a*qw*r21 + 4*c*qy*r23,
                              4*a*qz*r21 + 8*b*qw*r22 - 4*c*qx*r23], dim=1)
        
        M23_dq = torch.stack([2*a*qy*r31 + 2*a*qz*r21 + 2*b*qw*r22 - 2*c*qw*r33,
                              -2*a*qw*r21 + 2*a*qx*r31 + 4*b*qy*r32 + 2*b*qz*r22 + 2*c*qz*r33,
                              2*a*qw*r31 + 2*a*qx*r21 + 2*b*qy*r22 + 2*c*qy*r33 + 4*c*qz*r23,
                              -2*a*qy*r21 + 2*a*qz*r31 + 4*b*qw*r32 + 2*b*qx*r22 + 4*c*qw*r23 - 2*c*qx*r33], dim=1)
        
        M33_dq = torch.stack([4*a*qz*r31 + 4*b*qw*r32,
                              -4*a*qw*r31 + 4*b*qz*r32,
                              4*a*qx*r31 + 4*b*qy*r32 + 8*c*qz*r33,
                              -4*a*qy*r31 + 4*b*qx*r32 + 8*c*qw*r33], dim=1)
        
        return torch.stack([M11_dq, M12_dq, M13_dq, M12_dq, M22_dq, M23_dq, M13_dq, M23_dq, M33_dq], dim=1)
    
    @staticmethod
    def RDRT_dqdq(q, D):
        """
        Calculate the hessian of M.flatten() w.r.t. q.
        q (torch.Tensor, shape (batch_size, 4)): input
        D (torch.Tensor, shape (batch_size, 3, 3)): diagonal matrix

        Returns:
        (torch.Tensor, shape (batch_size, 9, 4, 4)): d^2M/dqdq
        """
        qx, qy, qz, qw = q.unbind(-1)
        a, b, c = D[:,0,0], D[:,1,1], D[:,2,2]
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

        M11_dqdq_11 = 16*a*pww + 48*a*pxx - 8*a + 8*b*pyy + 8*c*pzz
        M11_dqdq_12 = 16*b*pxy - 8*b*pzw + 8*c*pzw
        M11_dqdq_13 = -8*b*pyw + 16*c*pxz + 8*c*pyw
        M11_dqdq_14 = 32*a*pxw - 8*b*pyz + 8*c*pyz
        M11_dqdq_22 = 8*b*pxx + 8*c*pww
        M11_dqdq_23 = 8*pxw*(-b + c)
        M11_dqdq_24 = -8*b*pxz + 8*c*pxz + 16*c*pyw
        M11_dqdq_33 = 8*b*pww + 8*c*pxx
        M11_dqdq_34 = -8*b*pxy + 16*b*pzw + 8*c*pxy
        M11_dqdq_44 = 48*a*pww + 16*a*pxx - 8*a + 8*b*pzz + 8*c*pyy
        M11_dqdq = torch.stack([torch.stack([M11_dqdq_11, M11_dqdq_12, M11_dqdq_13, M11_dqdq_14], dim=1),
                               torch.stack([M11_dqdq_12, M11_dqdq_22, M11_dqdq_23, M11_dqdq_24], dim=1),
                               torch.stack([M11_dqdq_13, M11_dqdq_23, M11_dqdq_33, M11_dqdq_34], dim=1),
                               torch.stack([M11_dqdq_14, M11_dqdq_24, M11_dqdq_34, M11_dqdq_44], dim=1)], dim=1)

        M12_dqdq_11 = 24*a*pxy + 8*a*pzw - 8*c*pzw
        M12_dqdq_12 = 4*a*pww + 12*a*pxx - 2*a + 4*b*pww + 12*b*pyy - 2*b - 4*c*pww + 4*c*pzz
        M12_dqdq_13 = 8*a*pxw - 8*c*pxw + 8*c*pyz
        M12_dqdq_14 = 8*a*pxz + 8*a*pyw + 8*b*pyw - 8*c*pxz - 8*c*pyw
        M12_dqdq_22 = 24*b*pxy - 8*b*pzw + 8*c*pzw
        M12_dqdq_23 = -8*b*pyw + 8*c*pxz + 8*c*pyw
        M12_dqdq_24 = 8*a*pxw + 8*b*pxw - 8*b*pyz - 8*c*pxw + 8*c*pyz
        M12_dqdq_33 = 8*c*pxy
        M12_dqdq_34 = 12*a*pww + 4*a*pxx - 2*a - 12*b*pww - 4*b*pyy + 2*b - 4*c*pxx + 4*c*pyy
        M12_dqdq_44 = 8*a*pxy + 24*a*pzw + 8*b*pxy - 24*b*pzw - 8*c*pxy
        M12_dqdq = torch.stack([torch.stack([M12_dqdq_11, M12_dqdq_12, M12_dqdq_13, M12_dqdq_14], dim=1),
                               torch.stack([M12_dqdq_12, M12_dqdq_22, M12_dqdq_23, M12_dqdq_24], dim=1),
                               torch.stack([M12_dqdq_13, M12_dqdq_23, M12_dqdq_33, M12_dqdq_34], dim=1),
                               torch.stack([M12_dqdq_14, M12_dqdq_24, M12_dqdq_34, M12_dqdq_44], dim=1)], dim=1)
        
        M13_dqdq_11 = 24*a*pxz - 8*a*pyw + 8*b*pyw
        M13_dqdq_12 = -8*a*pxw + 8*b*pxw + 8*b*pyz
        M13_dqdq_13 = 4*a*pww + 12*a*pxx - 2*a - 4*b*pww + 4*b*pyy + 4*c*pww + 12*c*pzz - 2*c
        M13_dqdq_14 = -8*a*pxy + 8*a*pzw + 8*b*pxy - 8*b*pzw + 8*c*pzw
        M13_dqdq_22 = 8*b*pxz
        M13_dqdq_23 = 8*b*pxy - 8*b*pzw + 8*c*pzw
        M13_dqdq_24 = -12*a*pww - 4*a*pxx + 2*a + 4*b*pxx - 4*b*pzz + 12*c*pww + 4*c*pzz - 2*c
        M13_dqdq_33 = -8*b*pyw + 24*c*pxz + 8*c*pyw
        M13_dqdq_34 = 8*a*pxw - 8*b*pxw - 8*b*pyz + 8*c*pxw + 8*c*pyz
        M13_dqdq_44 = 8*a*pxz - 24*a*pyw - 8*b*pxz + 8*c*pxz + 24*c*pyw
        M13_dqdq = torch.stack([torch.stack([M13_dqdq_11, M13_dqdq_12, M13_dqdq_13, M13_dqdq_14], dim=1),
                               torch.stack([M13_dqdq_12, M13_dqdq_22, M13_dqdq_23, M13_dqdq_24], dim=1),
                               torch.stack([M13_dqdq_13, M13_dqdq_23, M13_dqdq_33, M13_dqdq_34], dim=1),
                               torch.stack([M13_dqdq_14, M13_dqdq_24, M13_dqdq_34, M13_dqdq_44], dim=1)], dim=1)

        M22_dqdq_11 = 8*a*pyy + 8*c*pww
        M22_dqdq_12 = 16*a*pxy + 8*a*pzw - 8*c*pzw
        M22_dqdq_13 = 8*pyw*(a - c)
        M22_dqdq_14 = 8*a*pyz + 16*c*pxw - 8*c*pyz
        M22_dqdq_22 = 8*a*pxx + 16*b*pww + 48*b*pyy - 8*b + 8*c*pzz
        M22_dqdq_23 = 8*a*pxw - 8*c*pxw + 16*c*pyz
        M22_dqdq_24 = 8*a*pxz + 32*b*pyw - 8*c*pxz
        M22_dqdq_33 = 8*a*pww + 8*c*pyy
        M22_dqdq_34 = 8*a*pxy + 16*a*pzw - 8*c*pxy
        M22_dqdq_44 = 8*a*pzz + 48*b*pww + 16*b*pyy - 8*b + 8*c*pxx
        M22_dqdq = torch.stack([torch.stack([M22_dqdq_11, M22_dqdq_12, M22_dqdq_13, M22_dqdq_14], dim=1),
                               torch.stack([M22_dqdq_12, M22_dqdq_22, M22_dqdq_23, M22_dqdq_24], dim=1),
                               torch.stack([M22_dqdq_13, M22_dqdq_23, M22_dqdq_33, M22_dqdq_34], dim=1),
                               torch.stack([M22_dqdq_14, M22_dqdq_24, M22_dqdq_34, M22_dqdq_44], dim=1)], dim=1)  

        M23_dqdq_11 = 8*a*pyz
        M23_dqdq_12 = 8*a*pxz - 8*a*pyw + 8*b*pyw
        M23_dqdq_13 = 8*a*pxy + 8*a*pzw - 8*c*pzw
        M23_dqdq_14 = -4*a*pyy + 4*a*pzz + 12*b*pww + 4*b*pyy - 2*b - 12*c*pww - 4*c*pzz + 2*c
        M23_dqdq_22 = -8*a*pxw + 8*b*pxw + 24*b*pyz
        M23_dqdq_23 = -4*a*pww + 4*a*pxx + 4*b*pww + 12*b*pyy - 2*b + 4*c*pww + 12*c*pzz - 2*c
        M23_dqdq_24 = -8*a*pxy - 8*a*pzw + 8*b*pxy + 8*b*pzw + 8*c*pzw
        M23_dqdq_33 = 8*a*pxw - 8*c*pxw + 24*c*pyz
        M23_dqdq_34 = 8*a*pxz - 8*a*pyw + 8*b*pyw - 8*c*pxz + 8*c*pyw
        M23_dqdq_44 = -8*a*pyz + 24*b*pxw + 8*b*pyz - 24*c*pxw + 8*c*pyz
        M23_dqdq = torch.stack([torch.stack([M23_dqdq_11, M23_dqdq_12, M23_dqdq_13, M23_dqdq_14], dim=1),
                               torch.stack([M23_dqdq_12, M23_dqdq_22, M23_dqdq_23, M23_dqdq_24], dim=1),
                               torch.stack([M23_dqdq_13, M23_dqdq_23, M23_dqdq_33, M23_dqdq_34], dim=1),
                               torch.stack([M23_dqdq_14, M23_dqdq_24, M23_dqdq_34, M23_dqdq_44], dim=1)], dim=1)

        M33_dqdq_11 = 8*a*pzz + 8*b*pww
        M33_dqdq_12 = 8*pzw*(-a + b)
        M33_dqdq_13 = 16*a*pxz - 8*a*pyw + 8*b*pyw
        M33_dqdq_14 = -8*a*pyz + 16*b*pxw + 8*b*pyz
        M33_dqdq_22 = 8*a*pww + 8*b*pzz
        M33_dqdq_23 = -8*a*pxw + 8*b*pxw + 16*b*pyz
        M33_dqdq_24 = -8*a*pxz + 16*a*pyw + 8*b*pxz
        M33_dqdq_33 = 8*a*pxx + 8*b*pyy + 16*c*pww + 48*c*pzz - 8*c
        M33_dqdq_34 = -8*a*pxy + 8*b*pxy + 32*c*pzw
        M33_dqdq_44 = 8*a*pyy + 8*b*pxx + 48*c*pww + 16*c*pzz - 8*c
        M33_dqdq = torch.stack([torch.stack([M33_dqdq_11, M33_dqdq_12, M33_dqdq_13, M33_dqdq_14], dim=1),
                               torch.stack([M33_dqdq_12, M33_dqdq_22, M33_dqdq_23, M33_dqdq_24], dim=1),
                               torch.stack([M33_dqdq_13, M33_dqdq_23, M33_dqdq_33, M33_dqdq_34], dim=1),
                               torch.stack([M33_dqdq_14, M33_dqdq_24, M33_dqdq_34, M33_dqdq_44], dim=1)], dim=1)
        
        return torch.stack([M11_dqdq, M12_dqdq, M13_dqdq, M12_dqdq, M22_dqdq, M23_dqdq, M13_dqdq, M23_dqdq, M33_dqdq], dim=1)