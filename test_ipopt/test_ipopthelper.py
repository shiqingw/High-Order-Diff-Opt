import IpoptHelper as ih
import numpy as np

Q = np.eye(3).astype(np.float64)
mu = np.array([4,0,0]).astype(np.float64)
A = np.array([[1,0,0],
              [-1,0,0],
              [0,1,0],
              [0,-1,0],
              [0,0,1],
              [0,0,-1]]).astype(np.float64)
b = -np.ones(6).astype(np.float64)
kappa = 10.0
x0 = np.array([0,0,0]).astype(np.float64)

problem = ih.EllipsoidAndLogSumExpNLP(Q, mu, A, b, kappa, x0, 3)
problem.solve()
print(problem.optimal_solution)
