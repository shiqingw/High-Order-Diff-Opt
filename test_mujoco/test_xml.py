import mujoco
import time
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

# Load the model
model = mujoco.MjModel.from_xml_path("skydio_x2/scene.xml")

data = mujoco.MjData(model)

viewer = mujoco.viewer.launch_passive(model, data)

start = time.time()

while viewer.is_running() and time.time() - start < 30:
    pass

viewer.close()
