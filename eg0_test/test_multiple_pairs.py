import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
from cores.rimon_method_python.rimon_method import rimon_method_numpy, rimon_method_pytorch
import time
import torch
import matplotlib.pyplot as plt

Ns = np.arange(1, 1000, 20)
repeat = 10
np_time = np.zeros((len(Ns), repeat))
torch_total_time = np.zeros((len(Ns), repeat))
torch_computation_time = np.zeros((len(Ns), repeat))
for q in range(len(Ns)):
    for r in range(repeat):
        # Define the ellipsoid
        N = Ns[q]
        nv = 3
        A = np.zeros((N, nv, nv))
        a = np.zeros((N, nv))
        B = np.zeros((N, nv, nv))
        b = np.zeros((N, nv))

        for i in range(N):
            A[i] = np.random.rand(nv, nv)
            A[i] = A[i] @ A[i].T + 1*np.eye(nv)
            a[i] = np.random.rand(nv)
            B[i] = np.random.rand(nv, nv)
            B[i] = B[i] @ B[i].T + 1*np.eye(nv)
            b[i] = np.random.rand(nv) + 10

        # Compute the results using pytorch
        start = time.time()
        A_torch = torch.from_numpy(A)
        a_torch = torch.from_numpy(a)
        B_torch = torch.from_numpy(B)
        b_torch = torch.from_numpy(b)
        middle = time.time()
        x_rimon_torch = rimon_method_pytorch(A_torch, a_torch, B_torch, b_torch)
        end = time.time()
        # print(f"Elapsed time using pytorch: {end - start} seconds")
        # print(f"Elapsed time only for computing: {end - middle} seconds")
        torch_total_time[q, r] = 1000*(end - start)
        torch_computation_time[q, r] = 1000*(end - middle)

        # Compute the results using numpy
        ans = np.zeros((N, nv))
        start = time.time()
        for i in range(N):
            x_rimon_numpy = rimon_method_numpy(A[i], a[i], B[i], b[i])
            ans[i] = x_rimon_numpy
        end = time.time()
        # print(f"Elapsed time using numpy: {end - start} seconds")
        np_time[q, r] = 1000*(end - start)


# plt.plot(Ns, np.mean(np_time, axis=1), label="Numpy")
plt.plot(Ns, np.mean(torch_total_time, axis=1), label="PyTorch")
plt.plot(Ns, np.mean(torch_computation_time, axis=1), label="PyTorch Computation")
plt.xlabel("Number of ellipsoids")
plt.ylabel("Time (ms)")
plt.legend()
plt.show()
