import torch
import timeit

N = 100
lambda_min = torch.rand(N, 3)

def view(lambda_min):
    return lambda_min.view(-1, 1, 3)

def unsqueeze(lambda_min):
    return lambda_min.unsqueeze(-2)

num = 10000
print('view:', timeit.timeit(lambda: view(lambda_min), number=num)/num)
print('unsqueeze:', timeit.timeit(lambda: unsqueeze(lambda_min), number=num)/num)
print('view:', timeit.timeit(lambda: view(lambda_min), number=num)/num)
print('unsqueeze:', timeit.timeit(lambda: unsqueeze(lambda_min), number=num)/num)