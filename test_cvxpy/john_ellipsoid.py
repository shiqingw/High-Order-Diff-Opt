import cvxpy as cp
import numpy as np

# Assume some dimensions for A and b, and m
n = 2  # This should be the dimension of A and b
m = 4  # Number of constraints

# Define the variable for A which should be symmetric positive definite
A = cp.Variable((n, n), PSD=True)

# Define the variable for b
b = cp.Variable(n)

# Assuming xi are given constant vectors, we should have them in an array. 
# We will use dummy data for xi as an example. Replace it with your actual data.
xis = np.random.randn(m, n)  # Replace this with your actual xi data
xis = np.array([[1,1],
                [1,-1],
                [-1,1],
                [-1,-1]])

# The constraints are that the 2-norm of (A*xi + b) is less than or equal to 1
constraints = [cp.norm(A @ xis[i] + b, 2) <= 1 for i in range(m)]

# The objective is to minimize the negative log determinant of A
objective = cp.Minimize(-cp.log_det(A))

# Define the problem and solve it
problem = cp.Problem(objective, constraints)
problem.solve()

# After solving the problem, A.value and b.value will contain the optimized values
print("The optimized matrix A is:")
print(A.value)

print("\nThe optimized vector b is:")
print(b.value)

A_np = A.value
print(np.linalg.eigvals(A_np))