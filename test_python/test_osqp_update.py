import osqp
import numpy as np
import scipy.sparse as sparse

# Define problem data
nv = 2
P = sparse.eye(nv, format='csc')
A = sparse.csc_matrix([[1.0, 0.0], [0.0, 1.0]])
q = -np.array([1.1, 1.1])
l = np.array([1.2, 1.2])
u = np.array([2.0, 2.0])

# Create an OSQP object
prob = osqp.OSQP()
prob.setup(P=P, q=q, A=A, l=l, u=u, verbose=False)
results = prob.solve()
print(results.x)

# Update problem data
A_new = sparse.csc_matrix([[1.0, 0.1], [0.1, 1.0]])
q_new = -np.array([1.1, 1.1])
l_new = np.array([-1.1, -1.1])
u_new = np.array([1.1, 1.1])
prob.update(q=q_new, l=l_new, u=u_new, Ax=A_new.data)
results = prob.solve()
print(results.x)

