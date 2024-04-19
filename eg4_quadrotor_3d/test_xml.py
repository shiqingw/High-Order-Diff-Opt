import mujoco
import time
import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

# Load the model
model = mujoco.MjModel.from_xml_path("skydio_x2/scene_cylinder.xml")

data = mujoco.MjData(model)

viewer = mujoco.viewer.launch_passive(model, data)

start = time.time()

while viewer.is_running() and time.time() - start < 30:
    hover_control = np.array([3.2495625, 3.2495625, 3.2495625, 3.2495625])
    data.ctrl[:] = hover_control
    Minv_mj = np.zeros((model.nv, model.nv))
    mujoco.mj_solveM(model, data, Minv_mj, np.eye(Minv_mj.shape[0]))
    print(Minv_mj)
    mujoco.mj_step(model, data)
    viewer.sync()

viewer.close()
