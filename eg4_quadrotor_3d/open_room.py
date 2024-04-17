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
from cores.utils.rotation_utils import np_get_quat_qw_first
from cores.utils.control_utils import solve_LQR_tracking, solve_infinite_LQR
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
    dt = system.delta_t
    env = SkydioMuJocoEnv(xml_name="scene", cam_distance=cam_distance, cam_azimuth=cam_azimuth,
                          cam_elevation=cam_elevation, cam_lookat=cam_lookat, dt=dt)

    # Tracking control via LQR
    t_final = 100
    horizon = int(t_final/dt)

    ####### state vector = [x,y,z,qx,qy,qz,qw,vx,vy,vz,wx,wy,wz] #######
    x_traj = np.zeros([horizon+1, system.n_states]).astype(config.np_dtype)
    t_traj = np.linspace(0, t_final, horizon+1)
    
    ####### Ellipse trajectory #######
    angular_vel_traj = 1
    semi_major_axis = 2
    semi_minor_axis = 1
    x_traj[:,1] = semi_major_axis * np.cos(angular_vel_traj*t_traj) # y
    x_traj[:,2] = semi_minor_axis * np.sin(angular_vel_traj*t_traj) + 2 # z
    x_traj[:,6] = np.ones(x_traj.shape[0])
    x_traj[:,8] = -semi_major_axis * angular_vel_traj * np.sin(angular_vel_traj*t_traj) # xdot
    x_traj[:,9] = semi_minor_axis * angular_vel_traj * np.cos(angular_vel_traj*t_traj) # ydot
    u_traj = np.zeros([horizon, system.n_controls])
    u_traj[:,0] = system.gravity * np.ones(u_traj.shape[0])

    ####### Striaght line trajectory #######
    # x_vel = 1
    # y_vel = 1
    # z_vel = 1
    # x_traj[:,0] = x_vel * t_traj
    # x_traj[:,1] = y_vel * t_traj
    # x_traj[:,2] = z_vel * t_traj + 0.2
    # x_traj[:,6] = np.ones(x_traj.shape[0])
    # x_traj[:,8] = x_vel * np.ones(x_traj.shape[0])
    # x_traj[:,9] = y_vel * np.ones(x_traj.shape[0])
    # x_traj[:,10] = z_vel * np.ones(x_traj.shape[0])
    # u_traj = np.zeros([horizon, system.n_controls])
    # u_traj[:,0] = system.gravity * np.ones(u_traj.shape[0])

    A_list = np.empty((horizon, system.n_states, system.n_states))
    B_list = np.empty((horizon, system.n_states, system.n_controls))
    for i in range(horizon):
        A, B = system.get_linearization(x_traj[i,:], u_traj[i,:]) 
        A_list[i] = A
        B_list[i] = B
    Q_pos = [10,10,10]
    Q_quat = [10,10,10,10]
    Q_v = [10,10,10]
    Q_omega = [0,0,0]
    Q_lqr = np.diag(Q_pos+Q_quat+Q_v+Q_omega)
    Q_list = np.array([Q_lqr]*(horizon +1))
    R_lqr = np.diag([1,1,1,1])
    R_list = np.array([R_lqr]*(horizon))
    print("==> Solve LQR gain")
    K_gains, k_feedforward = solve_LQR_tracking(A_list, B_list, Q_list, R_list, x_traj, horizon)

    def lqr_controller(state, i):
        K = K_gains[i]
        k = k_feedforward[i]
        return K @ state + k + u_traj[i,:]

    # Create records
    print("==> Create records")
    times = np.linspace(0, t_final, horizon+1).astype(config.np_dtype)
    states = np.zeros([horizon+1, system.n_states], dtype=config.np_dtype)
    states[0,:] = x_traj[0,:]
    controls = np.zeros([horizon, system.n_controls], dtype=config.np_dtype)
    desired_controls = np.zeros([horizon, system.n_controls], dtype=config.np_dtype)
    phi1s = np.zeros(horizon, dtype=config.np_dtype)
    phi2s = np.zeros(horizon, dtype=config.np_dtype)
    cbf_values = np.zeros(horizon, dtype=config.np_dtype)
    time_cvxpy = np.zeros(horizon, dtype=config.np_dtype)
    time_diff_helper = np.zeros(horizon, dtype=config.np_dtype)
    time_cbf_qp = np.zeros(horizon, dtype=config.np_dtype)
    time_control_loop = np.zeros(horizon, dtype=config.np_dtype)
    
    env.reset(states[0,0:3], np_get_quat_qw_first(states[0,3:7]), states[0,7:10], states[0,10:13])
    p_prev = states[0,0:3]
    for i in range(horizon):
        time_control_loop_start = time.time()
        state = states[i,:]
        u_lqr = lqr_controller(state, i)

        # u_nominal = np.array([9.81,0,0,0])
        # print(x_traj[i,:])
        # print(u_traj[i,:])
        # print(state)
        # print(u_nominal)
        # print(u_lqr)
        # print("############")
        # assert False


        new_state = system.get_next_state(state, u_lqr)
        env.step(new_state[0:3], np_get_quat_qw_first(new_state[3:7]), new_state[7:10], new_state[10:13])
        states[i+1,:] = new_state

        p_new = new_state[0:3]
        speed = np.linalg.norm((p_new-p_prev)/dt)
        rgba=np.array((np.clip(speed/10, 0, 1),
                     np.clip(1-speed/10, 0, 1),
                     .5, 1.))
        radius=.003*(1+speed)
        env.add_visual_capsule(p_prev, p_new, radius, rgba, 0, True)
        p_prev = p_new


        time_control_loop_end = time.time()
        time.sleep(max(0,dt-time_control_loop_end+time_control_loop_start))
