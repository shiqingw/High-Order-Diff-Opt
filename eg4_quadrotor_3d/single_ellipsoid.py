import json
import sys
import os
import argparse
import shutil
import time
import numpy as np
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Set the backend to 'Agg'

from skydio_envs.skydio_mj_env import SkydioMuJocoEnv
from cores.utils.utils import seed_everything, save_dict, load_dict
from cores.utils.rotation_utils import np_get_quat_qw_first
from cores.utils.control_utils import solve_LQR_tracking, solve_infinite_LQR
from cores.utils.proxsuite_utils import init_proxsuite_qp
from cores.dynamical_systems.create_system import get_system
import diffOptHelper as DOC
from cores.configuration.configuration import Configuration
from scipy.spatial.transform import Rotation

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
    initial_pos = np.array(test_settings["initial_pos"], dtype=config.np_dtype)
    initial_quat = np.array(test_settings["initial_quat"], dtype=config.np_dtype)
    dt = system.delta_t
    env = SkydioMuJocoEnv(xml_name="scene_ellipsoid", cam_distance=cam_distance, cam_azimuth=cam_azimuth,
                          cam_elevation=cam_elevation, cam_lookat=cam_lookat, dt=dt)
    env.reset(initial_pos, np_get_quat_qw_first(initial_quat), np.zeros(3), np.zeros(3))

    # Quadrotor bounding ellipsoid
    bounding_ellipsoid_offset = np.array(test_settings["bounding_ellipsoid_offset"], config.np_dtype)
    tmp = np.array(test_settings["bounding_ellipsoid_size"], config.np_dtype)
    D_BB_sqrt = np.diag(1.0/tmp)
    D_BB = D_BB_sqrt @ D_BB_sqrt.T

    # Obstacle
    obstacle_config = test_settings["obstacle_config"]
    obs_pos_np = np.array(obstacle_config["pos"], dtype=config.np_dtype)
    obs_quat_np = np.array(obstacle_config["quat"], dtype=config.np_dtype) # (x, y, z, w)
    obs_size_np = np.array(obstacle_config["size"])
    obs_coef_np = np.diag(1/obs_size_np**2)
    obs_R_np = Rotation.from_quat(obs_quat_np).as_matrix()
    obs_coef_np = obs_R_np @ obs_coef_np @ obs_R_np.T

    # Tracking control via LQR
    horizon = test_settings["horizon"]
    t_final = horizon * dt

    ####### state vector = [x, y, z, qx, qy, qz, qw, vx, vy, vz, wx, wy, wz] #######
    x_traj = np.zeros([horizon+1, system.n_states]).astype(config.np_dtype)
    t_traj = np.linspace(0, t_final, horizon+1)
    
    ####### Ellipse trajectory #######
    # angular_vel_traj = 0.4
    # semi_major_axis = 2
    # semi_minor_axis = 1
    # initial_phase = -np.pi/4
    # traj_center = np.array([0,0,2])
    # ######## Visualize the trajectory #######
    # N = 100
    # thetas = np.linspace(0, 2*np.pi, N)
    # traj_ellipse = np.zeros([N, 3], dtype=config.np_dtype)
    # traj_ellipse[:,0] = traj_center[0] + semi_major_axis * np.sin(thetas)
    # traj_ellipse[:,1] = traj_center[1] + semi_minor_axis * np.cos(thetas)
    # traj_ellipse[:,2] = traj_center[2]
    # for i in range(N-1):
    #     env.add_visual_capsule(traj_ellipse[i], traj_ellipse[i+1], 0.004, np.array([0,0,1,1]))
    #     env.viewer.sync()
    # id_geom_offset = env.viewer.user_scn.ngeom 
    ######## Compute the trajectory #######
    # x_traj[:,0] = semi_major_axis * np.cos(angular_vel_traj*t_traj + initial_phase) + traj_center[0] # x
    # x_traj[:,1] = semi_minor_axis * np.sin(angular_vel_traj*t_traj + initial_phase) + traj_center[1] # y
    # x_traj[:,2] = traj_center[2] # z
    # x_traj[:,3:7] = np.array([0,0,0,1], dtype=config.np_dtype)
    # x_traj[:,7] = -semi_major_axis * angular_vel_traj * np.sin(angular_vel_traj*t_traj) # xdot
    # x_traj[:,8] = semi_minor_axis * angular_vel_traj * np.cos(angular_vel_traj*t_traj) # ydot

    # u_traj = np.zeros([horizon, system.n_controls])
    # u_traj[:,0] = system.gravity * np.ones(u_traj.shape[0])

    ####### Striaght line trajectory #######
    start_point = np.array([0,0,0], dtype=config.np_dtype)
    end_point = np.array([0,0,5], dtype=config.np_dtype)
    ######## Visualize the trajectory #######
    N = 100
    thetas = np.linspace(0, 1, N)
    traj = start_point + thetas[:,None] * (end_point - start_point)
    for i in range(N-1):
        env.add_visual_capsule(traj[i], traj[i+1], 0.004, np.array([0,0,1,1]))
        env.viewer.sync()
    id_geom_offset = env.viewer.user_scn.ngeom 
    ####### Striaght line trajectory #######
    linear_vel = (end_point - start_point)/t_final
    x_traj[:,0] = start_point[0] + linear_vel[0] * t_traj
    x_traj[:,1] = start_point[1] + linear_vel[1] * t_traj
    x_traj[:,2] = start_point[2] + linear_vel[2] * t_traj
    x_traj[:,3:7] = np.array([0.0, 0.0, -np.sqrt(2), np.sqrt(2)], dtype=config.np_dtype)
    x_traj[:,7] = linear_vel[0] * np.ones(x_traj.shape[0])
    x_traj[:,8] = linear_vel[1] * np.ones(x_traj.shape[0])
    x_traj[:,9] = linear_vel[2] * np.ones(x_traj.shape[0])

    u_traj = np.zeros([horizon, system.n_controls])
    u_traj[:,0] = system.gravity * np.ones(u_traj.shape[0])

    A_list = np.empty((horizon, system.n_states, system.n_states))
    B_list = np.empty((horizon, system.n_states, system.n_controls))
    for i in range(horizon):
        A, B = system.get_linearization(x_traj[i,:], u_traj[i,:]) 
        A_list[i] = A
        B_list[i] = B
    Q_pos = [10,10,10]
    Q_quat = [10,10,10,10]
    Q_v = [10,10,10]
    Q_omega = [1,1,1]
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
    
    # CBF parameters
    CBF_config = test_settings["CBF_config"]
    alpha0 = CBF_config["alpha0"]
    gamma1 = CBF_config["gamma1"]
    gamma2 = CBF_config["gamma2"]
    compensation = CBF_config["compensation"] 

    # Define proxuite problem
    print("==> Define proxuite problem")
    n_CBF = 1
    n_controls = 4
    cbf_qp = init_proxsuite_qp(n_v=n_controls, n_eq=0, n_in=n_controls+n_CBF+1)

    # Create records
    print("==> Create records")
    times = np.linspace(0, t_final, horizon+1).astype(config.np_dtype)
    states = np.zeros([horizon+1, system.n_states], dtype=config.np_dtype)
    states[0,0:3] = initial_pos
    states[0,3:7] = initial_quat
    controls = np.zeros([horizon, system.n_controls], dtype=config.np_dtype)
    desired_controls = np.zeros([horizon, system.n_controls], dtype=config.np_dtype)
    phi1s = np.zeros([horizon, n_CBF], dtype=config.np_dtype)
    phi2s = np.zeros([horizon, n_CBF], dtype=config.np_dtype)
    cbf_values = np.zeros([horizon, n_CBF], dtype=config.np_dtype)
    time_cvxpy = np.zeros(horizon, dtype=config.np_dtype)
    time_diff_helper = np.zeros(horizon, dtype=config.np_dtype)
    time_cbf_qp = np.zeros(horizon, dtype=config.np_dtype)
    time_control_loop = np.zeros(horizon, dtype=config.np_dtype)
    
    p_prev = states[0,0:3]
    for i in range(horizon):
        time_control_loop_start = time.time()
        state = states[i,:]
        P_BB = state[0:3] + bounding_ellipsoid_offset
        quat_BB = state[3:7]
        R_BB = Rotation.from_quat(quat_BB).as_matrix()
        u_nominal = lqr_controller(state, i)

        time_diff_helper_tmp = 0
        if CBF_config["active"]:
            # Matrices for the CBF-QP constraints
            C = np.zeros([n_controls+n_CBF+1, n_controls], dtype=config.np_dtype)
            lb = np.zeros(n_controls+n_CBF+1, dtype=config.np_dtype)
            ub = np.zeros(n_controls+n_CBF+1, dtype=config.np_dtype)
            CBF_tmp = np.zeros(n_CBF, dtype=config.np_dtype)
            phi1_tmp = np.zeros(n_CBF, dtype=config.np_dtype)
            phi2_tmp = np.zeros(n_CBF, dtype=config.np_dtype)

            time_diff_helper_tmp -= time.time()
            alpha, _, alpha_dx_tmp, alpha_dxdx_tmp = DOC.getGradientAndHessianEllipsoids(P_BB, quat_BB, D_BB, 
                                R_BB, obs_coef_np, obs_pos_np)
            time_diff_helper_tmp += time.time()
            
            CBF = alpha - alpha0
            
            # Order of parameters in alpha_dx_tmp and alpha_dxdx_tmp: [qx, qy, qz, qw, x, y, z]
            # Convert to the order of [x, y, z, qx, qy, qz, qw, vx, vy, vz, wx, wy, wz]
            alpha_dx = np.zeros(system.n_states, dtype=config.np_dtype)
            alpha_dx[0:3] = alpha_dx_tmp[4:7]
            alpha_dx[3:7] = alpha_dx_tmp[0:4]

            alpha_dxdx = np.zeros((system.n_states, system.n_states), dtype=config.np_dtype)
            alpha_dxdx[0:3,0:3] = alpha_dxdx_tmp[4:7,4:7]
            alpha_dxdx[3:7,3:7] = alpha_dxdx_tmp[0:4,0:4]
            alpha_dxdx[0:3,3:7] = alpha_dxdx_tmp[4:7,0:4]
            alpha_dxdx[3:7,0:3] = alpha_dxdx_tmp[0:4,4:7]

            drift = np.squeeze(system.drift(state)) # f(x), shape = (6,)
            actuation = system.actuation(state) # g(x), shape = (6,2)
            drift_jac = system.drift_jac(state) # df/dx, shape = (6,6)

            phi1 = alpha_dx @ drift + gamma1 * CBF # scalar
            g = -u_nominal # shape = (2,)
            C[0,:] = alpha_dx @ drift_jac @ actuation
            lb[0] = -drift.T @ alpha_dxdx @ drift - alpha_dx @ drift_jac @ drift \
                - (gamma1+gamma2) * alpha_dx @ drift - gamma1 * gamma2 * CBF + compensation
            ub[0] = np.inf

            # tmp = np.array([C[0,2], 0, -C[0,0], 0]).astype(config.np_dtype)
            tmp = np.array([C[0,2], 0.01*C[0,3], -C[0,0], -0.01*C[0,1]]).astype(config.np_dtype)
            C[1,:] = tmp
            lb[1] = C[1,:] @ system.ueq + 0.1 - 10*CBF
            ub[1] = np.inf

            g = -u_nominal
            C[n_CBF+1:n_CBF+1+n_controls,:] = np.eye(n_controls, dtype=config.np_dtype)
            lb[n_CBF+1:] = input_lb
            ub[n_CBF+1:] = input_ub
            cbf_qp.update(g=g, C=C, l=lb, u=ub)
            time_cbf_qp_start = time.time()
            cbf_qp.solve()
            time_cbf_qp_end = time.time()
            u = cbf_qp.results.x

            CBF_tmp[0] = CBF
            phi1_tmp[0] = phi1
            phi2_tmp[0] = 0

            for kk in range(n_CBF):
                phi2_tmp[kk] = C[kk,:] @ u - lb[kk]
            time_control_loop_end = time.time()
        else:
            u = u_nominal
            time_cbf_qp_start = 0
            time_cbf_qp_end = 0
            time_control_loop_end = time.time()
            CBF_tmp = np.zeros(n_CBF, dtype=config.np_dtype)
            phi1_tmp = np.zeros(n_CBF, dtype=config.np_dtype)
            phi2_tmp = np.zeros(n_CBF, dtype=config.np_dtype)

        # Step the environment
        new_state = system.get_next_state(state, u)
        env.step(new_state[0:3], np_get_quat_qw_first(new_state[3:7]), new_state[7:10], new_state[10:13])

        # Draw trjaectory
        p_new = new_state[0:3]
        speed = np.linalg.norm((p_new-p_prev)/dt)
        rgba=np.array((np.clip(speed/10, 0, 1),
                     np.clip(1-speed/10, 0, 1),
                     .5, 1.))
        radius=.003*(1+speed)
        env.add_visual_capsule(p_prev, p_new, radius, rgba, id_geom_offset, True)
        p_prev = p_new

        # Record
        states[i+1,:] = new_state
        controls[i,:] = u
        desired_controls[i,:] = u_nominal
        cbf_values[i,:] = CBF_tmp
        phi1s[i,:] = phi1_tmp
        phi2s[i,:] = phi2_tmp
        time_diff_helper[i] = time_diff_helper_tmp
        time_cbf_qp[i] = time_cbf_qp_end - time_cbf_qp_start
        time_control_loop[i] = time_control_loop_end - time_control_loop_start

        time.sleep(max(0,dt-time_control_loop_end+time_control_loop_start))

    # Close the environment
    env.close()

    # Save summary
    print("==> Save results")
    summary = {"times": times,
               "states": states,
               "controls": controls,
               "desired_controls": desired_controls,
               "phi1s": phi1s,
               "phi2s": phi2s,
               "cbf_values": cbf_values,
               "time_cvxpy": time_cvxpy,
               "time_diff_helper": time_diff_helper,
               "time_cbf_qp": time_cbf_qp}
    save_dict(summary, os.path.join(results_dir, 'summary.pkl'))
    
    # Visualization
    print("==> Draw plots")
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams.update({"text.usetex": True,
                         "text.latex.preamble": r"\usepackage{amsmath}"})
    plt.rcParams.update({'pdf.fonttype': 42})

    fig, ax = plt.subplots(figsize=(10,8), dpi=config.dpi, frameon=True)
    plt.plot(times, states[:,0], linestyle="-", label=r"$x$")
    plt.plot(times, states[:,1], linestyle="-", label=r"$y$")
    plt.plot(times, states[:,2], linestyle="-", label=r"$z$")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'plot_state_pos.pdf'))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10,8), dpi=config.dpi, frameon=True)
    plt.plot(times, states[:,3], linestyle="-", label=r"$q_x$")
    plt.plot(times, states[:,4], linestyle="-", label=r"$q_y$")
    plt.plot(times, states[:,5], linestyle="-", label=r"$q_z$")
    plt.plot(times, states[:,6], linestyle="-", label=r"$q_w$")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'plot_state_quat.pdf'))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10,8), dpi=config.dpi, frameon=True)
    plt.plot(times, states[:,7], linestyle="-", label=r"$v_x$")
    plt.plot(times, states[:,8], linestyle="-", label=r"$v_y$")
    plt.plot(times, states[:,9], linestyle="-", label=r"$v_z$")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'plot_state_linear_vel.pdf'))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10,8), dpi=config.dpi, frameon=True)
    plt.plot(times, states[:,10], linestyle="-", label=r"$\omega_x$")
    plt.plot(times, states[:,11], linestyle="-", label=r"$\omega_y$")
    plt.plot(times, states[:,12], linestyle="-", label=r"$\omega_z$")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'plot_state_angular_vel.pdf'))
    plt.close(fig)


    for i in range(system.n_controls):
        fig, ax = plt.subplots(figsize=(10,8), dpi=config.dpi, frameon=True)
        plt.plot(times[0:horizon], desired_controls[:,i], color="tab:blue", linestyle=":", 
                label="u_{:d} nominal".format(i+1))
        plt.plot(times[0:horizon], controls[:,i], color="tab:blue", linestyle="-", label="u_{:d}".format(i+1))
        plt.axhline(y = input_lb[i], color = 'black', linestyle = 'dotted', linewidth = 2)
        plt.axhline(y = input_ub[i], color = 'black', linestyle = 'dotted', linewidth = 2)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'plot_controls_{:d}.pdf'.format(i+1)))
        plt.close(fig)

    for i in range(n_CBF):
        fig, ax = plt.subplots(figsize=(10,8), dpi=config.dpi, frameon=True)
        plt.plot(times[0:horizon], phi1s[:,i], label="phi1")
        plt.plot(times[0:horizon], phi2s[:,i], label="phi2")
        plt.axhline(y = 0.0, color = 'black', linestyle = 'dotted', linewidth = 2)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'plot_phi_{:d}.pdf'.format(i+1)))
        plt.close(fig)
    
    for i in range(n_CBF):
        fig, ax = plt.subplots(figsize=(10,8), dpi=config.dpi, frameon=True)
        plt.plot(times[0:horizon], cbf_values[:,i], label="CBF")
        plt.axhline(y = 0.0, color = 'black', linestyle = 'dotted', linewidth = 2)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'plot_cbf_{:d}.pdf'.format(i+1)))
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(10,8), dpi=config.dpi, frameon=True)
    plt.plot(times[0:horizon], time_cvxpy, label="cvxpy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'plot_time_cvxpy.pdf'))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10,8), dpi=config.dpi, frameon=True)
    plt.plot(times[0:horizon], time_diff_helper, label="diff helper")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'plot_time_diff_helper.pdf'))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10,8), dpi=config.dpi, frameon=True)
    plt.plot(times[0:horizon], time_cbf_qp, label="CBF-QP")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'plot_time_cbf_qp.pdf'))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10,8), dpi=config.dpi, frameon=True)
    plt.plot(times[0:horizon], time_control_loop, label="control loop")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'plot_time_control_loop.pdf'))
    plt.close(fig)

    # Print solving time
    print("==> Control loop solving time: {:.5f} s".format(np.mean(time_control_loop)))
    print("==> CVXPY solving time: {:.5f} s".format(np.mean(time_cvxpy)))
    print("==> Diff helper solving time: {:.5f} s".format(np.mean(time_diff_helper)))
    print("==> CBF-QP solving time: {:.5f} s".format(np.mean(time_cbf_qp)))

    print("==> Done!")
