import diffOptEllipsoidCpp
import numpy as np
import scipy as sp
import timeit
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from cores.configuration.configuration import Configuration
config = Configuration()
from cores.rimon_method_python.rimon_method import rimon_method_numpy

a = np.array([0.1, 0.2, 0.3], dtype=config.np_dtype)
A = np.array([[2, 0.1, 0.1],
              [0.1, 2, 0.1],
              [0.1, 0.1, 3]], dtype=config.np_dtype)
# print(sp.linalg.eigvals(A))

b = np.array([3.3, 3.2, 3.1], dtype=config.np_dtype)
B = np.array([[1, 0.1, 0.1],
              [0.1, 3, 0.1],
              [0.1, 0.1, 1]], dtype=config.np_dtype)
# print(sp.linalg.eigvals(B))

number = 1000
print("C++ Eigen time:", timeit.timeit(lambda: diffOptEllipsoidCpp.rimonMethod(A, a, B, b), number=number)/1000)
x_rimon = diffOptEllipsoidCpp.rimonMethod(A, a, B, b)
print(x_rimon)

print()
print("C++ xtensor time:", timeit.timeit(lambda: diffOptEllipsoidCpp.rimonMethodXtensor(A, a, B, b), number=number)/1000)
x_rimon = diffOptEllipsoidCpp.rimonMethodXtensor(A, a, B, b)
print(x_rimon)

print()
print("Numpy time:", timeit.timeit(lambda: rimon_method_numpy(A, a, B, b), number=number)/1000)
x_rimon = rimon_method_numpy(A, a, B, b)
print(x_rimon)

