import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import numpy as np
import time
from cores.utils.osqp_utils import init_osqp
from cores.configuration.configuration import Configuration
config = Configuration()

n_v = 2
n_in = 2
qp = init_osqp(n_v, n_in)

A = np.array([[1,-1],[1,1]], dtype=config.np_dtype)
u = np.array([-1,1], dtype=config.np_dtype)
start = time.time()
qp.update(Ax=A, u=u)
results = qp.solve()
end = time.time()
print(end-start)
print(results.x)
