import numpy as np
import HOCBFHelperPy as hh
import scalingFunctionsHelperPy as sfh
import time
from scipy.spatial.transform import Rotation

Q_d = np.diag([1,2,3])
mu_d = np.array([4,1,1])
A_d = np.array([[1,0,0],
              [-1,0,0],
              [0,1,0],
              [0,-1,0],
              [0,0,1],
              [0,0,-1]])
b_d = -np.ones(6)
kappa_d = 10.0
vertices = np.array([[1,1,1], [1,1,-1], [1,-1,1], [1,-1,-1], [-1,1,1], [-1,1,-1], [-1,-1,1], [-1,-1,-1]])

SF_rob = sfh.Ellipsoid3d(True, Q_d, np.zeros(3))
SF_obs = sfh.LogSumExp3d(False, A_d, b_d, kappa_d)
prob = hh.ElliposoidAndLogSumExp3dPrb(SF_rob, SF_obs, vertices)

q = np.array([0,0,0,1])

time_start = time.time()
alpha, alpha_dx, alpha_dxdx = prob.solve(mu_d, q)
time_end = time.time()
print("Time elapsed: ", time_end - time_start)
# print(alpha)
# print(alpha_dx)
# print(alpha_dxdx)

n_threads = 9
N = 18
collection = hh.Problem3dCollection(n_threads)
for i in range(N):
    prob = hh.ElliposoidAndLogSumExp3dPrb(SF_rob, SF_obs, vertices)
    collection.addProblem(prob, i)
all_d = np.zeros((N, 3))
all_q = np.zeros((N, 4))
for i in range(N):
    all_q[i] = q
    all_d[i] = mu_d

import timeit
print(timeit.timeit(lambda: collection.solveGradientAndHessian(all_d, all_q), number=1000))

time_start = time.time()
for i in range(1000):
    all_d = all_d + 0.01

    all_q = all_q + np.random.rand(N, 4) * 0.01
    all_q = all_q / np.linalg.norm(all_q, axis=1).reshape(-1, 1)

    all_alpha, all_alpha_dx, all_alpha_dxdx = collection.solveGradientAndHessian(all_d, all_q)
    print(all_alpha)

time_end = time.time()
print("Time elapsed: ", time_end - time_start)
