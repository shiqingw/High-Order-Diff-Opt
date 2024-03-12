import torch
import timeit

# create random tensor of shape (2, 3)
N = 100
x = torch.rand(N, 3)
A = torch.rand(N, 3, 3)
a = torch.rand(N, 3)

def einsum(x,a,A):
    tmp = x - a
    return torch.einsum('bi,bij,bj->b', tmp, A, tmp)

def matmul(x,a,A):
    tmp = (x - a).unsqueeze(-2)
    return torch.matmul(torch.matmul(tmp, A), tmp.transpose(-1,-2)).squeeze()

num = 10000
print('einsum:', timeit.timeit(lambda: einsum(x,a,A), number=num)/num)
print('matmul:', timeit.timeit(lambda: matmul(x,a,A), number=num)/num)
print('einsum:', timeit.timeit(lambda: einsum(x,a,A), number=num)/num)
print('matmul:', timeit.timeit(lambda: matmul(x,a,A), number=num)/num)