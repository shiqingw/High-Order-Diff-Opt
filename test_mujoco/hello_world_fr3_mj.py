import time
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import numpy as np

from fr3_envs.fr3_mj_env_collision import FR3MuJocoEnv


def main():
    env = FR3MuJocoEnv(xml_name="fr3_on_table_with_bounding_boxes_wiping")
    info = env.reset([0.0, -0.785, 0.0, -2.356, 0.0, 2, 0.785, 0.001, 0.001])

    target = np.array(
        [0, -0.785398163, 0.0, -2.35619449, 0.0, 1.57079632679, 0.785398163397]
    )
    Kd = 10.0

    # Close the viewer automatically after 30 wall-seconds.
    start = time.time()

    while env.viewer.is_running() and time.time() - start < 30:
        step_start = time.time()

        error = target - info["q"][:7]
        error_norm = np.linalg.norm(error)

        if error_norm >= 0.3:
            Kp = 10.0
            finger_pos = 0.0
        else:
            Kp = np.clip(1.0 / error_norm, 0, 100)
            finger_pos = 0.03

        tau = Kp * error + Kd * (0 - info["dq"][:7]) + info["G"][:7]

        info = env.step(tau, finger_pos)
        env.sleep(step_start)

    env.close()


if __name__ == "__main__":
    main()
