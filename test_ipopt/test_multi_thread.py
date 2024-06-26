import numpy as np
from cyipopt import minimize_ipopt
import concurrent.futures

def objective_function(x):
    return (x[0] - 1)**2 + (x[1] - 2)**2

def constraint_function(x):
    return [x[0] + x[1] - 3]

def optimization_task(starting_point):
    x0 = np.array(starting_point)
    bounds = [(0, None), (0, None)]
    constraints = [{'type': 'eq', 'fun': constraint_function}]
    result = minimize_ipopt(objective_function, x0, bounds=bounds, constraints=constraints)
    return result

if __name__ == "__main__":
    starting_points = [[0, 0], [1, 1], [2, 2], [3, 3]]

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(optimization_task, sp) for sp in starting_points]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            print(f"Starting point: {result['x0']} -> Result: {result['x']}, Objective: {result['fun']}")
