import IpoptHelper as ih
import numpy as np
import timeit

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

n_worker = 100
all_n = np.array([3]*n_worker)
all_Q = np.array([Q]*n_worker)
all_mu = np.array([mu]*n_worker)
all_A = np.array([A]*n_worker)
all_b = np.array([b]*n_worker)
all_kappa = np.array([kappa]*n_worker)

problems = ih.EllipsoidAndLogSumExpNLPAndSolverMultiple(n_worker, all_n, all_Q, all_mu, all_A, all_b, all_kappa)
problems.solve()
time = timeit.timeit(problems.solve, number=1000)
print("Time to solve: ", time)
# print(problems.get_optimal_solution())

# all_Q = np.array([Q + i *np.eye(3) for i in range(n_worker)])
# all_mu = np.array([mu + i * np.ones(3) for i in range(n_worker)])
# problems.update_problem_data(all_Q, all_mu, all_A, all_b, all_kappa)
# problems.solve()
# print(problems.get_optimal_solution())

