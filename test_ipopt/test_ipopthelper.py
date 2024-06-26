import IpoptHelper as ih
import numpy as np
import timeit

Q = np.eye(3).astype(np.float64)
mu = np.array([4,1,1]).astype(np.float64)
A = np.array([[1,0,0],
              [-1,0,0],
              [0,1,0],
              [0,-1,0],
              [0,0,1],
              [0,0,-1]]).astype(np.float64)
b = -np.ones(6).astype(np.float64)
kappa = 10.0

problem = ih.EllipsoidAndLogSumExpNLPAndSolver(3, Q, mu, A, b, kappa)
# print(problem.get_initial_guess())
# print(problem.get_optimal_solution())
problem.solve()
print(problem.get_optimal_solution_x())
# print(problem.get_initial_guess())
# print(problem.get_optimal_solution())
time = timeit.timeit(problem.solve, number=1000)
print("Time to solve: ", time)
# print(problem.get_optimal_solution_x())

# import cvxpy as cp
# x = cp.Variable(3)
# objective = cp.Minimize(cp.quad_form(x-mu, Q))
# constraints = [cp.log_sum_exp(kappa*(A @ x + b)) - np.log(A.shape[0])<= 0]
# prob = cp.Problem(objective, constraints)
# prob.solve(warm_start=True, solver=cp.SCS)
# # time the solving time for 1000 iterations
# time = timeit.timeit(prob.solve, number=1000)
# print("Time to solve: ", time)
