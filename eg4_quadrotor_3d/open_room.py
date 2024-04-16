import json
import sys
import os
import argparse
import shutil
import time
import numpy as np
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from skydio_envs.skydio_mj_env import SkydioMuJocoEnv
from cores.utils.utils import seed_everything, save_dict, load_dict
from cores.utils.control_utils import solve_LQR_tracking
from cores.utils.proxsuite_utils import init_proxsuite_qp
from cores.dynamical_systems.create_system import get_system
import cores_cpp.diffOptCpp as DOC
from cores.configuration.configuration import Configuration

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_num', default=1, type=int, help='test case number')
    args = parser.parse_args()

    # Create result directory
    exp_num = args.exp_num
    results_dir = "{}/eg4_results/{:03d}".format(str(Path(__file__).parent.parent), exp_num)
    test_settings_path = "{}/test_settings/test_settings_{:03d}.json".format(str(Path(__file__).parent), exp_num)
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)
    shutil.copy(test_settings_path, results_dir)

    # Load test settings
    with open(test_settings_path, "r", encoding="utf8") as f:
        test_settings = json.load(f)

    # Seed everything
    seed_everything(test_settings["seed"])

    # Load configuration
    config = Configuration()

    # Build dynamical system
    system_name = test_settings["system_name"]
    system = get_system(system_name)
    
    # Various configs
    simulator_config = test_settings["simulator_config"]
    controller_config = test_settings["controller_config"]
    CBF_config = test_settings["CBF_config"]
    trajectory_config = test_settings["trajectory_config"]

    # Input torque limits
    input_limits_config = test_settings["input_limits_config"]
    input_lb = np.array(input_limits_config["lb"], dtype=config.np_dtype)
    input_ub = np.array(input_limits_config["ub"], dtype=config.np_dtype)

    # Create and reset simulation
    cam_distance = simulator_config["cam_distance"]
    cam_azimuth = simulator_config["cam_azimuth"]
    cam_elevation = simulator_config["cam_elevation"]
    cam_lookat = simulator_config["cam_lookat"]
    initial_pos = test_settings["initial_pos"]
    initial_quat = test_settings["initial_quat"]
    dt = 0.01

    env = SkydioMuJocoEnv(xml_name="scene", cam_distance=cam_distance, cam_azimuth=cam_azimuth,
                          cam_elevation=cam_elevation, cam_lookat=cam_lookat, dt=dt)
    
    env.reset(initial_pos, initial_quat, np.zeros(3), np.zeros(3))

    horizon = test_settings["horizon"]
    for i in range(horizon):
        time_control_loop_start = time.time()
        if i == 0:
            env.step([0,0,0.1], initial_quat, [0,0,0.1], np.zeros(3))
        else:
            env.step([0,0,0.1], initial_quat, np.zeros(3), np.zeros(3))
        time_control_loop_end = time.time()
        time.sleep(max(0,dt-time_control_loop_end+time_control_loop_start))
