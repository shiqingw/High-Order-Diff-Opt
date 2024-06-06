import cvxpy as cp

# Define the variables
x = cp.Variable()
y = cp.Variable()
z = cp.Variable()

# Define the parameters
alpha1 = 1.0  # You can set these to any positive values
alpha2 = 1.0
alpha3 = 1.0
epsilon1 = 1.5  # Make sure these are in the range (0, 2)
epsilon2 = 1.5

# Define the function f
term1 = cp.power((cp.power(x / alpha1, 2 / epsilon2) + cp.power(y / alpha2, 2 / epsilon2)), epsilon2 / epsilon1)
term2 = cp.power(z / alpha3, 2 / epsilon1)
f = term1 + term2

# Define the objective
objective = cp.Minimize(f)

# Define the constraints (if any)
constraints = [cp.power((cp.power(x-10 / alpha1, 2 / epsilon2) + cp.power(y-10 / alpha2, 2 / epsilon2)), epsilon2 / epsilon1)+cp.power(z-10 / alpha3, 2 / epsilon1)<=1]

# Form and solve the problem
problem = cp.Problem(objective, constraints)
problem.solve(solver=cp.ECOS)
assert problem.is_dcp()
assert problem.is_dpp()

# Output the results
print("Optimal value of f:", f.value)
print("Optimal values of x, y, z:", x.value, y.value, z.value)
