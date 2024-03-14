import torch

class Quaternion_RDRT:
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
        
        M21_dq = torch.stack([4*a*qx*r21 + 2*a*qy*r11 + 2*b*qy*r22 - 2*c*qw*r13 + 2*c*qz*r23,
                              2*a*qx*r11 + 2*b*qx*r22 + 4*b*qy*r12 + 2*c*qw*r23 + 2*c*qz*r13,
                              2*a*qw*r11 - 2*b*qw*r22 + 2*c*qx*r23 + 2*c*qy*r13,
                              4*a*qw*r21 + 2*a*qz*r11 + 4*b*qw*r12 - 2*b*qz*r22 - 2*c*qx*r13 + 2*c*qy*r23], dim=1)
        
        M22_dq = torch.stack([4*a*qy*r21 - 4*c*qw*r23,
                              4*a*qx*r21 + 8*b*qy*r22 + 4*c*qz*r23,
                              4*a*qw*r21 + 4*c*qy*r23,
                              4*a*qz*r21 + 8*b*qw*r22 - 4*c*qx*r23], dim=1)
        
        M23_dq = torch.stack([2*a*qy*r31 + 2*a*qz*r21 + 2*b*qw*r22 - 2*c*qw*r33,
                              -2*a*qw*r21 + 2*a*qx*r31 + 4*b*qy*r32 + 2*b*qz*r22 + 2*c*qz*r33,
                              2*a*qw*r31 + 2*a*qx*r21 + 2*b*qy*r22 + 2*c*qy*r33 + 4*c*qz*r23,
                              -2*a*qy*r21 + 2*a*qz*r31 + 4*b*qw*r32 + 2*b*qx*r22 + 4*c*qw*r23 - 2*c*qx*r33], dim=1)
        
        M31_dq = torch.stack([4*a*qx*r31 + 2*a*qz*r11 + 2*b*qw*r12 + 2*b*qy*r32 + 2*c*qz*r33,
                              -2*a*qw*r11 + 2*b*qx*r32 + 2*b*qz*r12 + 2*c*qw*r33,
                              2*a*qx*r11 - 2*b*qw*r32 + 2*b*qy*r12 + 2*c*qx*r33 + 4*c*qz*r13,
                              4*a*qw*r31 - 2*a*qy*r11 + 2*b*qx*r12 - 2*b*qz*r32 + 4*c*qw*r13 + 2*c*qy*r33], dim=1)
        
        M32_dq = torch.stack([2*a*qy*r31 + 2*a*qz*r21 + 2*b*qw*r22 - 2*c*qw*r33,
                              -2*a*qw*r21 + 2*a*qx*r31 + 4*b*qy*r32 + 2*b*qz*r22 + 2*c*qz*r33,
                              2*a*qw*r31 + 2*a*qx*r21 + 2*b*qy*r22 + 2*c*qy*r33 + 4*c*qz*r23,
                              -2*a*qy*r21 + 2*a*qz*r31 + 4*b*qw*r32 + 2*b*qx*r22 + 4*c*qw*r23 - 2*c*qx*r33], dim=1)
        
        M33_dq = torch.stack([4*a*qz*r31 + 4*b*qw*r32,
                              -4*a*qw*r31 + 4*b*qz*r32,
                              4*a*qx*r31 + 4*b*qy*r32 + 8*c*qz*r33,
                              -4*a*qy*r31 + 4*b*qx*r32 + 8*c*qw*r33], dim=1)
        
        return torch.stack([M11_dq, M12_dq, M13_dq, M21_dq, M22_dq, M23_dq, M31_dq, M32_dq, M33_dq], dim=1)
    
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
        r11 = 2*(qw**2+qx**2)-1
        r12 = 2*(qx*qy-qw*qz)
        r13 = 2*(qx*qz+qw*qy)
        r21 = 2*(qx*qy+qw*qz)
        r22 = 2*(qw**2+qy**2)-1
        r23 = 2*(qy*qz-qw*qx)
        r31 = 2*(qx*qz-qw*qy)
        r32 = 2*(qy*qz+qw*qx)
        r33 = 2*(qw**2+qz**2)-1

        M11_dqdq_11 = 16*a*qw**2 + 48*a*qx**2 - 8*a + 8*b*qy**2 + 8*c*qz**2
        M11_dqdq_12 = -8*b*qw*qz + 16*b*qx*qy + 8*c*qw*qz
        M11_dqdq_13 = -8*b*qw*qy + 8*c*qw*qy + 16*c*qx*qz
        M11_dqdq_14 = 32*a*qw*qx - 8*b*qy*qz + 8*c*qy*qz
        M11_dqdq_22 = 8*b*qx**2 + 8*c*qw**2
        M11_dqdq_23 = 8*qw*qx*(-b + c)
        M11_dqdq_24 = -8*b*qx*qz + 16*c*qw*qy + 8*c*qx*qz
        M11_dqdq_33 = 8*b*qw**2 + 8*c*qx**2
        M11_dqdq_34 = 16*b*qw*qz - 8*b*qx*qy + 8*c*qx*qy
        M11_dqdq_44 = 48*a*qw**2 + 16*a*qx**2 - 8*a + 8*b*qz**2 + 8*c*qy**2
        M11_dqdq = torch.stack(torch.stack([M11_dqdq_11, M11_dqdq_12, M11_dqdq_13, M11_dqdq_14], dim=1),
                               torch.stack([M11_dqdq_12, M11_dqdq_22, M11_dqdq_23, M11_dqdq_24], dim=1),
                               torch.stack([M11_dqdq_13, M11_dqdq_23, M11_dqdq_33, M11_dqdq_34], dim=1),
                               torch.stack([M11_dqdq_14, M11_dqdq_24, M11_dqdq_34, M11_dqdq_44], dim=1), dim=1)

        M12_dqdq_11 = 8*a*qw*qz + 24*a*qx*qy - 8*c*qw*qz
        M12_dqdq_12 = 4*a*qw**2 + 12*a*qx**2 - 2*a + 4*b*qw**2 + 12*b*qy**2 - 2*b - 4*c*qw**2 + 4*c*qz**2
        M12_dqdq_13 = 8*a*qw*qx - 8*c*qw*qx + 8*c*qy*qz
        M12_dqdq_14 = 8*a*qw*qy + 8*a*qx*qz + 8*b*qw*qy - 8*c*qw*qy - 8*c*qx*qz
        M12_dqdq_22 = -8*b*qw*qz + 24*b*qx*qy + 8*c*qw*qz
        M12_dqdq_23 = -8*b*qw*qy + 8*c*qw*qy + 8*c*qx*qz
        M12_dqdq_24 = 8*a*qw*qx + 8*b*qw*qx - 8*b*qy*qz - 8*c*qw*qx + 8*c*qy*qz
        M12_dqdq_33 = 8*c*qx*qy
        M12_dqdq_34 = 12*a*qw**2 + 4*a*qx**2 - 2*a - 12*b*qw**2 - 4*b*qy**2 + 2*b - 4*c*qx**2 + 4*c*qy**2
        M12_dqdq_44 = 24*a*qw*qz + 8*a*qx*qy - 24*b*qw*qz + 8*b*qx*qy - 8*c*qx*qy
        M12_dqdq = torch.stack(torch.stack([M12_dqdq_11, M12_dqdq_12, M12_dqdq_13, M12_dqdq_14], dim=1),
                               torch.stack([M12_dqdq_12, M12_dqdq_22, M12_dqdq_23, M12_dqdq_24], dim=1),
                               torch.stack([M12_dqdq_13, M12_dqdq_23, M12_dqdq_33, M12_dqdq_34], dim=1),
                               torch.stack([M12_dqdq_14, M12_dqdq_24, M12_dqdq_34, M12_dqdq_44], dim=1), dim=1)
        
        M13_dqdq_11 = -8*a*qw*qy + 24*a*qx*qz + 8*b*qw*qy
        M13_dqdq_12 = -8*a*qw*qx + 8*b*qw*qx + 8*b*qy*qz
        M13_dqdq_13 = 4*a*qw**2 + 12*a*qx**2 - 2*a - 4*b*qw**2 + 4*b*qy**2 + 4*c*qw**2 + 12*c*qz**2 - 2*c
        M13_dqdq_14 = 8*a*qw*qz - 8*a*qx*qy - 8*b*qw*qz + 8*b*qx*qy + 8*c*qw*qz
        M13_dqdq_22 = 8*b*qx*qz
        M13_dqdq_23 = -8*b*qw*qz + 8*b*qx*qy + 8*c*qw*qz
        M13_dqdq_24 = -12*a*qw**2 - 4*a*qx**2 + 2*a + 4*b*qx**2 - 4*b*qz**2 + 12*c*qw**2 + 4*c*qz**2 - 2*c
        M13_dqdq_33 = -8*b*qw*qy + 8*c*qw*qy + 24*c*qx*qz
        M13_dqdq_34 = 8*a*qw*qx - 8*b*qw*qx - 8*b*qy*qz + 8*c*qw*qx + 8*c*qy*qz
        M13_dqdq_44 = -24*a*qw*qy + 8*a*qx*qz - 8*b*qx*qz + 24*c*qw*qy + 8*c*qx*qz
        M13_dqdq = torch.stack(torch.stack([M13_dqdq_11, M13_dqdq_12, M13_dqdq_13, M13_dqdq_14], dim=1),
                               torch.stack([M13_dqdq_12, M13_dqdq_22, M13_dqdq_23, M13_dqdq_24], dim=1),
                               torch.stack([M13_dqdq_13, M13_dqdq_23, M13_dqdq_33, M13_dqdq_34], dim=1),
                               torch.stack([M13_dqdq_14, M13_dqdq_24, M13_dqdq_34, M13_dqdq_44], dim=1), dim=1)

        M21_dqdq_11 = 8*a*qw*qz + 24*a*qx*qy - 8*c*qw*qz
        M21_dqdq_12 = 4*a*qw**2 + 12*a*qx**2 - 2*a + 4*b*qw**2 + 12*b*qy**2 - 2*b - 4*c*qw**2 + 4*c*qz**2
        M21_dqdq_13 = 8*a*qw*qx - 8*c*qw*qx + 8*c*qy*qz
        M21_dqdq_14 = 8*a*qw*qy + 8*a*qx*qz + 8*b*qw*qy - 8*c*qw*qy - 8*c*qx*qz
        M21_dqdq_22 = -8*b*qw*qz + 24*b*qx*qy + 8*c*qw*qz
        M21_dqdq_23 = -8*b*qw*qy + 8*c*qw*qy + 8*c*qx*qz
        M21_dqdq_24 = 8*a*qw*qx + 8*b*qw*qx - 8*b*qy*qz - 8*c*qw*qx + 8*c*qy*qz
        M21_dqdq_33 = 8*c*qx*qy
        M21_dqdq_34 = 12*a*qw**2 + 4*a*qx**2 - 2*a - 12*b*qw**2 - 4*b*qy**2 + 2*b - 4*c*qx**2 + 4*c*qy**2
        M21_dqdq_44 = 24*a*qw*qz + 8*a*qx*qy - 24*b*qw*qz + 8*b*qx*qy - 8*c*qx*qy
        M21_dqdq = torch.stack(torch.stack([M21_dqdq_11, M21_dqdq_12, M21_dqdq_13, M21_dqdq_14], dim=1),
                               torch.stack([M21_dqdq_12, M21_dqdq_22, M21_dqdq_23, M21_dqdq_24], dim=1),
                               torch.stack([M21_dqdq_13, M21_dqdq_23, M21_dqdq_33, M21_dqdq_34], dim=1),
                               torch.stack([M21_dqdq_14, M21_dqdq_24, M21_dqdq_34, M21_dqdq_44], dim=1), dim=1)

        M22_dqdq_11 = 8*a*qy**2 + 8*c*qw**2
        M22_dqdq_12 = 8*a*qw*qz + 16*a*qx*qy - 8*c*qw*qz
        M22_dqdq_13 = 8*qw*qy*(a - c)
        M22_dqdq_14 = 8*a*qy*qz + 16*c*qw*qx - 8*c*qy*qz
        M22_dqdq_22 = 8*a*qx**2 + 16*b*qw**2 + 48*b*qy**2 - 8*b + 8*c*qz**2
        M22_dqdq_23 = 8*a*qw*qx - 8*c*qw*qx + 16*c*qy*qz
        M22_dqdq_24 = 8*a*qx*qz + 32*b*qw*qy - 8*c*qx*qz
        M22_dqdq_33 = 8*a*qw**2 + 8*c*qy**2
        M22_dqdq_34 = 16*a*qw*qz + 8*a*qx*qy - 8*c*qx*qy
        M22_dqdq_44 = 8*a*qz**2 + 48*b*qw**2 + 16*b*qy**2 - 8*b + 8*c*qx**2
        M22_dqdq = torch.stack(torch.stack([M22_dqdq_11, M22_dqdq_12, M22_dqdq_13, M22_dqdq_14], dim=1),
                               torch.stack([M22_dqdq_12, M22_dqdq_22, M22_dqdq_23, M22_dqdq_24], dim=1),
                               torch.stack([M22_dqdq_13, M22_dqdq_23, M22_dqdq_33, M22_dqdq_34], dim=1),
                               torch.stack([M22_dqdq_14, M22_dqdq_24, M22_dqdq_34, M22_dqdq_44], dim=1), dim=1)  

        M23_dqdq_11 = 8*a*qy*qz
        M23_dqdq_12 = -8*a*qw*qy + 8*a*qx*qz + 8*b*qw*qy
        M23_dqdq_13 = 8*a*qw*qz + 8*a*qx*qy - 8*c*qw*qz
        M23_dqdq_14 = -4*a*qy**2 + 4*a*qz**2 + 12*b*qw**2 + 4*b*qy**2 - 2*b - 12*c*qw**2 - 4*c*qz**2 + 2*c
        M23_dqdq_22 = -8*a*qw*qx + 8*b*qw*qx + 24*b*qy*qz
        M23_dqdq_23 = -4*a*qw**2 + 4*a*qx**2 + 4*b*qw**2 + 12*b*qy**2 - 2*b + 4*c*qw**2 + 12*c*qz**2 - 2*c
        M23_dqdq_24 = -8*a*qw*qz - 8*a*qx*qy + 8*b*qw*qz + 8*b*qx*qy + 8*c*qw*qz
        M23_dqdq_33 = 8*a*qw*qx - 8*c*qw*qx + 24*c*qy*qz
        M23_dqdq_34 = -8*a*qw*qy + 8*a*qx*qz + 8*b*qw*qy + 8*c*qw*qy - 8*c*qx*qz
        M23_dqdq_44 = -8*a*qy*qz + 24*b*qw*qx + 8*b*qy*qz - 24*c*qw*qx + 8*c*qy*qz
        M23_dqdq = torch.stack(torch.stack([M23_dqdq_11, M23_dqdq_12, M23_dqdq_13, M23_dqdq_14], dim=1),
                               torch.stack([M23_dqdq_12, M23_dqdq_22, M23_dqdq_23, M23_dqdq_24], dim=1),
                               torch.stack([M23_dqdq_13, M23_dqdq_23, M23_dqdq_33, M23_dqdq_34], dim=1),
                               torch.stack([M23_dqdq_14, M23_dqdq_24, M23_dqdq_34, M23_dqdq_44], dim=1), dim=1)

        M31_dqdq_11 = -8*a*qw*qy + 24*a*qx*qz + 8*b*qw*qy
        M31_dqdq_12 = -8*a*qw*qx + 8*b*qw*qx + 8*b*qy*qz
        M31_dqdq_13 = 4*a*qw**2 + 12*a*qx**2 - 2*a - 4*b*qw**2 + 4*b*qy**2 + 4*c*qw**2 + 12*c*qz**2 - 2*c
        M31_dqdq_14 = 8*a*qw*qz - 8*a*qx*qy - 8*b*qw*qz + 8*b*qx*qy + 8*c*qw*qz
        M31_dqdq_22 = 8*b*qx*qz
        M31_dqdq_23 = -8*b*qw*qz + 8*b*qx*qy + 8*c*qw*qz
        M31_dqdq_24 = -12*a*qw**2 - 4*a*qx**2 + 2*a + 4*b*qx**2 - 4*b*qz**2 + 12*c*qw**2 + 4*c*qz**2 - 2*c
        M31_dqdq_33 = -8*b*qw*qy + 8*c*qw*qy + 24*c*qx*qz
        M31_dqdq_34 = 8*a*qw*qx - 8*b*qw*qx - 8*b*qy*qz + 8*c*qw*qx + 8*c*qy*qz
        M31_dqdq_44 = -24*a*qw*qy + 8*a*qx*qz - 8*b*qx*qz + 24*c*qw*qy + 8*c*qx*qz
        M31_dqdq = torch.stack(torch.stack([M31_dqdq_11, M31_dqdq_12, M31_dqdq_13, M31_dqdq_14], dim=1),
                               torch.stack([M31_dqdq_12, M31_dqdq_22, M31_dqdq_23, M31_dqdq_24], dim=1),
                               torch.stack([M31_dqdq_13, M31_dqdq_23, M31_dqdq_33, M31_dqdq_34], dim=1),
                               torch.stack([M31_dqdq_14, M31_dqdq_24, M31_dqdq_34, M31_dqdq_44], dim=1), dim=1)

        M32_dqdq_11 = 8*a*qy*qz
        M32_dqdq_12 = -8*a*qw*qy + 8*a*qx*qz + 8*b*qw*qy
        M32_dqdq_13 = 8*a*qw*qz + 8*a*qx*qy - 8*c*qw*qz
        M32_dqdq_14 = -4*a*qy**2 + 4*a*qz**2 + 12*b*qw**2 + 4*b*qy**2 - 2*b - 12*c*qw**2 - 4*c*qz**2 + 2*c
        M32_dqdq_22 = -8*a*qw*qx + 8*b*qw*qx + 24*b*qy*qz
        M32_dqdq_23 = -4*a*qw**2 + 4*a*qx**2 + 4*b*qw**2 + 12*b*qy**2 - 2*b + 4*c*qw**2 + 12*c*qz**2 - 2*c
        M32_dqdq_24 = -8*a*qw*qz - 8*a*qx*qy + 8*b*qw*qz + 8*b*qx*qy + 8*c*qw*qz
        M32_dqdq_33 = 8*a*qw*qx - 8*c*qw*qx + 24*c*qy*qz
        M32_dqdq_34 = -8*a*qw*qy + 8*a*qx*qz + 8*b*qw*qy + 8*c*qw*qy - 8*c*qx*qz
        M32_dqdq_44 = -8*a*qy*qz + 24*b*qw*qx + 8*b*qy*qz - 24*c*qw*qx + 8*c*qy*qz
        M32_dqdq = torch.stack(torch.stack([M32_dqdq_11, M32_dqdq_12, M32_dqdq_13, M32_dqdq_14], dim=1),
                               torch.stack([M32_dqdq_12, M32_dqdq_22, M32_dqdq_23, M32_dqdq_24], dim=1),
                               torch.stack([M32_dqdq_13, M32_dqdq_23, M32_dqdq_33, M32_dqdq_34], dim=1),
                               torch.stack([M32_dqdq_14, M32_dqdq_24, M32_dqdq_34, M32_dqdq_44], dim=1), dim=1)

        M33_dqdq_11 = 8*a*qz**2 + 8*b*qw**2
        M33_dqdq_12 = 8*qw*qz*(-a + b)
        M33_dqdq_13 = -8*a*qw*qy + 16*a*qx*qz + 8*b*qw*qy
        M33_dqdq_14 = -8*a*qy*qz + 16*b*qw*qx + 8*b*qy*qz
        M33_dqdq_22 = 8*a*qw**2 + 8*b*qz**2
        M33_dqdq_23 = -8*a*qw*qx + 8*b*qw*qx + 16*b*qy*qz
        M33_dqdq_24 = 16*a*qw*qy - 8*a*qx*qz + 8*b*qx*qz
        M33_dqdq_33 = 8*a*qx**2 + 8*b*qy**2 + 16*c*qw**2 + 48*c*qz**2 - 8*c
        M33_dqdq_34 = -8*a*qx*qy + 8*b*qx*qy + 32*c*qw*qz
        M33_dqdq_44 = 8*a*qy**2 + 8*b*qx**2 + 48*c*qw**2 + 16*c*qz**2 - 8*c
        M33_dqdq = torch.stack(torch.stack([M33_dqdq_11, M33_dqdq_12, M33_dqdq_13, M33_dqdq_14], dim=1),
                               torch.stack([M33_dqdq_12, M33_dqdq_22, M33_dqdq_23, M33_dqdq_24], dim=1),
                               torch.stack([M33_dqdq_13, M33_dqdq_23, M33_dqdq_33, M33_dqdq_34], dim=1),
                               torch.stack([M33_dqdq_14, M33_dqdq_24, M33_dqdq_34, M33_dqdq_44], dim=1), dim=1)
        
        return torch.stack([M11_dqdq, M12_dqdq, M13_dqdq, M21_dqdq, M22_dqdq, M23_dqdq, M31_dqdq, M32_dqdq, M33_dqdq], dim=1)