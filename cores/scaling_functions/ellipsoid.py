import torch

class Ellipsoid():
    """
    We consider F(p) = (p - a)^T A (p - a) where A is a symmetric matrix and a is a vector.
    The parameters of F are A and a. In compact form, x = [A.flatten(), a]. This class calculates
    various derivatives of F w.r.t. x and p.
    """
    @staticmethod
    def F(p, A, a):
        """
        Compute the value of F.
        p (torch.Tensor, shape (batch_size, dim(p))): input
        A (torch.Tensor, shape (batch_size, dim(p), dim(p))): real symmetric quadratic coefficient matrix
        a (torch.Tensor, shape (batch_size, dim(p))): center of the ellipsoid

        Returns:
        (torch.Tensor, shape (batch_size,)): ellipsoid function value
        """
        tmp = (p - a).unsqueeze(-1) # shape (batch_size, dim(p), 1)

        return torch.matmul(torch.matmul(tmp.transpose(-1,-2), A), tmp).flatten()# (batch_size,) 

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

        return torch.matmul(A + A.transpose(-1,-2), tmp).squeeze(-1)

    @staticmethod
    def F_dx(p, A, a):
        """
        Compute dF/dx.
        p (torch.Tensor, shape (batch_size, dim(p))): input
        A (torch.Tensor, shape (batch_size, dim(p), dim(p))): real symmetric quadratic coefficient matrix
        a (torch.Tensor, shape (batch_size, dim(p))): center of the ellipsoid
        
        Returns:
        (torch.Tensor, shape (batch_size, dim(p)): dF/dx
        """
        vector = (a - p).unsqueeze(-1) # shape (batch_size, dim(p), 1)
        tmp1 = torch.matmul(vector, vector.transpose(-1,-2)).flatten(1) # shape (batch_size, dim(p)**2)
        tmp2 = torch.matmul(A + A.transpose(-1,-2), vector).squeeze(-1) # shape (batch_size, dim(p))
        
        return torch.cat((tmp1, tmp2), dim=1) # shape (batch_size, dim(p)**2 + dim(p))

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

        return A + A.transpose(-1,-2)

    @staticmethod
    def F_dpdx(p, A, a):
        """
        Compute d^2F/dpdx.
        p (torch.Tensor, shape (batch_size, dim(p))): input
        A (torch.Tensor, shape (batch_size, dim(p), dim(p))): real symmetric quadratic coefficient matrix
        a (torch.Tensor, shape (batch_size, dim(p))): center of the ellipsoid

        Returns:
        (torch.Tensor, shape (batch_size, dim(p), dim(p)**2 + dim(p)): d^2F/dpdx
        """
        batch_size = A.shape[0]
        dim_p = A.shape[-1]
        vector = p - a
        tmp1 = torch.zeros((batch_size, dim_p, dim_p**2), dtype=A.dtype, device=A.device)
        for i in range(dim_p):
            tmp1[:,i,i*dim_p:(i+1)*dim_p] = vector
        for i in range(dim_p):
            tmp1[:,i,i:dim_p**2:dim_p] += vector
        tmp2 = - A - A.transpose(-1,-2) # shape (batch_size, dim(p), dim(p))

        return torch.cat((tmp1, tmp2), dim=2) # shape (batch_size, dim(p), dim(p)**2 + dim(p))
    
    @staticmethod
    def F_dxdx(p, A, a):
        """
        Compute d^2F/dxdx.
        p (torch.Tensor, shape (batch_size, dim(p))): input
        A (torch.Tensor, shape (batch_size, dim(p), dim(p))): real symmetric quadratic coefficient matrix
        a (torch.Tensor, shape (batch_size, dim(p))): center of the ellipsoid

        Returns:
        (torch.Tensor, shape (batch_size, dim(p)**2 + dim(p), dim(p)**2 + dim(p)): d^2F/dxdx
        """
        batch_size = A.shape[0]
        dim_p = A.shape[-1]
        dim_x = dim_p**2 + dim_p
        total = torch.zeros((batch_size, dim_x, dim_x), dtype=A.dtype, device=A.device)
        vector = a - p
        tmp = torch.zeros((batch_size, dim_p, dim_p**2), dtype=A.dtype, device=A.device)
        for i in range(dim_p):
            tmp[:,i,i*dim_p:(i+1)*dim_p] = vector
        for i in range(dim_p):
            tmp[:,i,i:dim_p**2:dim_p] += vector
        total[:,dim_p**2:dim_x,0:dim_p**2] = tmp
        total[:,0:dim_p**2,dim_p**2:dim_x] = tmp.transpose(-1,-2)
        total[:,dim_p**2:dim_x,dim_p**2:dim_x] = A + A.transpose(-1,-2)

        return total

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
    def F_dpdpdx(p, A, a):
        """
        Compute d^3F/dpdpdx.
        p (torch.Tensor, shape (batch_size, dim(p))): input
        A (torch.Tensor, shape (batch_size, dim(p), dim(p))): real symmetric quadratic coefficient matrix
        a (torch.Tensor, shape (batch_size, dim(p))): center of the ellipsoid

        Returns:
        (torch.Tensor, shape (batch_size, dim(p), dim(p), dim(p)**2 + dim(p)): d^3F/dpdpdx
        """
        batch_size = A.shape[0]
        dim_p = A.shape[-1]
        dim_x = dim_p**2 + dim_p
        total = torch.zeros((batch_size, dim_p, dim_p, dim_x), dtype=A.dtype, device=A.device)
        for i in range(dim_p**2):
            total[:,i//dim_p,i%dim_p,i] += 1.0
            total[:,i%dim_p,i//dim_p,i] += 1.0

        return total
    
    @staticmethod
    def F_dpdxdx(p, A, a):
        """
        Compute d^3F/dpdxdx.
        p (torch.Tensor, shape (batch_size, dim(p))): input
        A (torch.Tensor, shape (batch_size, dim(p), dim(p))): real symmetric quadratic coefficient matrix
        a (torch.Tensor, shape (batch_size, dim(p))): center of the ellipsoid

        Returns:
        (torch.Tensor, shape (batch_size, dim(p), dim(p)**2 + dim(p), dim(p)**2 + dim(p)): d^3F/dpdxdx
        """
        batch_size = A.shape[0]
        dim_p = A.shape[-1]
        dim_x = dim_p**2 + dim_p
        total = torch.zeros((batch_size, dim_p, dim_x, dim_x), dtype=A.dtype, device=A.device)

        tmp = torch.zeros((batch_size, dim_p, dim_p, dim_p**2), dtype=A.dtype, device=A.device)
        for i in range(dim_p**2):
            tmp[:,i//dim_p,i%dim_p,i] -= 1.0
            tmp[:,i%dim_p,i//dim_p,i] -= 1.0
        
        total[:,:,dim_p**2:dim_x,0:dim_p**2] = tmp
        total[:,:,0:dim_p**2,dim_p**2:dim_x] = tmp.transpose(-1,-2)
        
        return total