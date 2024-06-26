import numpy as np
from cyipopt import minimize_ipopt
import multiprocessing

def objective_function(x):
    return (x[0] - 1)**2 + (x[1] - 2)**2

def constraint_function(x):
    return [x[0] + x[1] - 3]

def optimization_task(starting_point):
    x0 = np.array(starting_point)
    bounds = [(0, None), (0, None)]
    constraints = [{'type': 'eq', 'fun': constraint_function}]
    result = minimize_ipopt(objective_function, x0, bounds=bounds, constraints=constraints)
    return starting_point, result

if __name__ == "__main__":
    starting_points = [[0, 0], [1, 1], [2, 2], [3, 3]]

    with multiprocessing.Pool() as pool:
        results = pool.map(optimization_task, starting_points)
    
    for starting_point, result in results:
        print(f"Starting point: {starting_point} -> Result: {result['x']}, Objective: {result['fun']}")
