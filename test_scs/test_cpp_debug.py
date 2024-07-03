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

N = 10
for i in range(N):
    SF_rob = sfh.Ellipsoid3d(True, Q_d, np.zeros(3))
    SF_obs = sfh.LogSumExp3d(False, A_d, b_d, kappa_d)
    prob = hh.ElliposoidAndLogSumExp3dPrb(SF_rob, SF_obs, vertices)
    q = np.array([0,0,0,1])
    alpha, alpha_dx, alpha_dxdx = prob.solve(mu_d, q)

