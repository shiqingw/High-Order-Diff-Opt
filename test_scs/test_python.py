import scs
import numpy as np
from scipy import sparse
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
c_d = np.log(A_d.shape[0])
kappa_d = 10.0

A_d = kappa_d * A_d
b_d = kappa_d * b_d

time_start = time.time()
n = A_d.shape[0]
n_p = 3
A_exp = sparse.lil_matrix((3 * n, n_p + n))
b_exp = np.zeros(3 * n)
for i in range(n):
    A_exp[i * 3, 0] = - A_d[i, 0]
    A_exp[i * 3, 1] = - A_d[i, 1]
    A_exp[i * 3, 2] = - A_d[i, 2]
    A_exp[i * 3 + 2, i + n_p] = -1

    b_exp[i * 3] = b_d[i] - c_d
    b_exp[i * 3 + 1] = 1

A = sparse.vstack(
    [
        # positive cone
        sparse.hstack([sparse.csc_matrix((1, n_p)), np.ones((1, n))]),
        # exponential cones
        A_exp,
    ],
    format="csc",
)

P = sparse.block_diag([Q_d, sparse.csc_matrix((n,n))], format="csc")

b = np.hstack([1, b_exp])
c = np.hstack([-Q_d @ mu_d, np.zeros(n)])

# SCS data
data = dict(P=P, A=A, b=b, c=c)
# ep is exponential cone (primal), with n triples
cone = dict(l=1, ep=n)

# Setup workspace
solver = scs.SCS(
    data,
    cone,
    eps_abs=1e-5,
    eps_rel=1e-5,
    verbose=False
)
sol = solver.solve()
time_end = time.time()
x_scs = sol["x"][:n_p]
print(x_scs)
print("Time elapsed: ", time_end - time_start)
print(sol["x"].shape)
print(sol["y"].shape)
print(sol["s"].shape)

# Compare with cvxpy
# import cvxpy as cp
# time_start = time.time()
# x = cp.Variable(n_p)
# objective = cp.Minimize(0.5 * cp.quad_form(x, Q_d) - mu_d.T @ Q_d @ x)
# constraint = [cp.log_sum_exp(A_d @ x + b_d) <= c_d]
# prob = cp.Problem(objective, constraint)
# prob.solve(solver=cp.SCS, verbose=False)
# time_end = time.time()
# print("Time elapsed: ", time_end - time_start)
# print(x.value)