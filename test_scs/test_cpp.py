import numpy as np
import HOCBFHelperPy as hh
import scalingFunctionsHelperPy as sfh
import time

Q_d = np.eye(3)
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
print(prob.p_sol)
print(alpha)
print(alpha_dx)
print(alpha_dxdx)

import timeit

# print(timeit.timeit(lambda: prob.solve(mu_d, q), number=1000))

# print(timeit.timeit(lambda: prob.solve_scs_prb(mu_d, q), number=1000))
# print(timeit.timeit(lambda: sfh.getGradientAndHessian3d(p_sol, SF_rob, mu_d, q, SF_obs, d2, q2), number=1000))

d2 = np.zeros(3)
q2 = np.array([0,0,0,1])
p_sol = np.array([1.1700977, 0.8654747, 0.8654747])
alpha, alpha_dx, alpha_dxdx = sfh.getGradientAndHessian3d(p_sol, SF_rob, mu_d, q, SF_obs, d2, q2)

alpha2, alpha_dx2, alpha_dxdx2 = sfh.getGradientAndHessian3dOld(p_sol, SF_rob, mu_d, q, SF_obs, d2, q2)

print(alpha-alpha2)
print(alpha_dx-alpha_dx2)
print(alpha_dxdx-alpha_dxdx2)