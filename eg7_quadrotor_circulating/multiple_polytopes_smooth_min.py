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
import diffOptHelper2 as DOC
from cores.configuration.configuration import Configuration
from scipy.spatial.transform import Rotation
from cores.obstacle_collections.john_ellipsoid_collection import JohnEllipsoidCollection
from cores.obstacle_collections.polytope_collection import PolytopeCollection
import cvxpy as cp
import mediapy as media

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_num', default=2, type=int, help='test case number')
    args = parser.parse_args()

    # Create result directory
    exp_num = args.exp_num
    results_dir = "{}/eg7_results/{:03d}".format(str(Path(__file__).parent.parent), exp_num)
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
    image_height = simulator_config["image_height"]
    image_width = simulator_config["image_width"]
    initial_pos = np.array(test_settings["initial_pos"], dtype=config.np_dtype)
    initial_quat = np.array(test_settings["initial_quat"], dtype=config.np_dtype)
    dt = system.delta_t
    env = SkydioMuJocoEnv(xml_name="scene_multiple_polytopes", cam_distance=cam_distance, cam_azimuth=cam_azimuth,
                          cam_elevation=cam_elevation, cam_lookat=cam_lookat, dt=dt)
    env.reset(initial_pos, np_get_quat_qw_first(initial_quat), np.zeros(3), np.zeros(3))
    env.create_renderer(image_height, image_width)
    camera_name = "fixed"

    # Quadrotor bounding ellipsoid
    bounding_ellipsoid_offset = np.array(test_settings["bounding_ellipsoid_offset"], config.np_dtype)
    tmp = np.array(test_settings["bounding_ellipsoid_size"], config.np_dtype)
    D_BB_sqrt = np.diag(1.0/tmp)
    D_BB = D_BB_sqrt @ D_BB_sqrt.T
    robot_SF = DOC.Ellipsoid3d(True, D_BB, bounding_ellipsoid_offset)

    # Obstacle
    obstacle_config = test_settings["obstacle_config"]
    n_obstacle = len(obstacle_config)
    obs_col = PolytopeCollection(3, n_obstacle, obstacle_config)
    
    # Define cvxpy problem
    cvxpy_config = test_settings["cvxpy_config"]
    x_max_np = cvxpy_config["x_max"]
    obstacle_kappa = cvxpy_config["obstacle_kappa"]
    print("==> Define cvxpy problem")
    cp_problems = []
    obstacle_SFs = []
    _p = cp.Variable(3)
    _ellipse_Q_sqrt = cp.Parameter((3,3))
    _ellipse_b = cp.Parameter(3)
    _ellipse_c = cp.Parameter()
    id_geom_offset = 0
    for (i, key) in enumerate(obs_col.face_equations.keys()):
        A_obs_np = obs_col.face_equations[key]["A"]
        b_obs_np = obs_col.face_equations[key]["b"]
        n_cons_obs = len(b_obs_np)
        obj = cp.Minimize(cp.sum_squares(_ellipse_Q_sqrt @ _p) + _ellipse_b.T @ _p + _ellipse_c)
        cons = [cp.log_sum_exp(obstacle_kappa*(A_obs_np @ _p + b_obs_np)-x_max_np) - np.log(n_cons_obs) + x_max_np <= 0]
        problem = cp.Problem(obj, cons)
        assert problem.is_dcp()
        assert problem.is_dpp()

        # warm start with fake data
        R_b_to_w_np = np.eye(3)
        drone_pos_np = np.array([0,0,0], dtype=config.np_dtype)
        ellipse_Q_sqrt_np = D_BB_sqrt @ R_b_to_w_np.T
        ellipse_Q_np = ellipse_Q_sqrt_np.T @ ellipse_Q_sqrt_np
        _ellipse_Q_sqrt.value = ellipse_Q_sqrt_np
        _ellipse_b.value = -2 * ellipse_Q_np @ drone_pos_np
        _ellipse_c.value = drone_pos_np.T @ ellipse_Q_np @ drone_pos_np
        problem.solve(warm_start=True, solver=cp.SCS)
        cp_problems.append(problem)

        # create scaling functions
        SF =DOC.LogSumExp3d(False, A_obs_np, b_obs_np, obstacle_kappa)
        obstacle_SFs.append(SF)

        # add visual ellipsoids
        all_points = obs_col.face_equations[key]["vertices_in_world"]
        for j in range(len(all_points)):
            env.add_visual_ellipsoid(0.05*np.ones(3), all_points[j], np.eye(3), np.array([1,0,0,1]),id_geom_offset=id_geom_offset)
            id_geom_offset = env.viewer.user_scn.ngeom

    # Tracking control via LQR
    horizon = test_settings["horizon"]
    t_1 = 0.6 * horizon * dt
    t_final = horizon * dt

    ####### state vector = [x, y, z, qx, qy, qz, qw, vx, vy, vz, wx, wy, wz] #######
    x_traj = np.zeros([horizon+1, system.n_states]).astype(config.np_dtype)
    t_traj = np.linspace(0, t_final, horizon+1)
    
    ####### Striaght line trajectory #######
    start_point = np.array([0,0,0], dtype=config.np_dtype)
    end_point = np.array([0,0,5], dtype=config.np_dtype)
    ######## Visualize the trajectory #######
    N = 100
    thetas = np.linspace(0, 1, N)
    traj = start_point + thetas[:,None] * (end_point - start_point)
    for i in range(N-1):
        env.add_visual_capsule(traj[i], traj[i+1], 0.004, np.array([0,0,1,1]),id_geom_offset=id_geom_offset)
        env.viewer.sync()
    id_geom_offset = env.viewer.user_scn.ngeom 
    ####### Striaght line trajectory #######
    linear_vel = (end_point - start_point)/t_1
    for i in range(x_traj.shape[0]):
        if i*dt <= t_1:
            x_traj[i,0] = start_point[0] + linear_vel[0] * i*dt
            x_traj[i,1] = start_point[1] + linear_vel[1] * i*dt
            x_traj[i,2] = start_point[2] + linear_vel[2] * i*dt
            x_traj[i,7] = linear_vel[0]
            x_traj[i,8] = linear_vel[1]
            x_traj[i,9] = linear_vel[2]
        else:
            x_traj[i,0] = end_point[0]
            x_traj[i,1] = end_point[1]
            x_traj[i,2] = end_point[2]
            x_traj[i,7] = 0
            x_traj[i,8] = 0
            x_traj[i,9] = 0
        x_traj[i,3:7] = np.array([0.0, 0.0, -np.sqrt(2), np.sqrt(2)], dtype=config.np_dtype)

    u_traj = np.zeros([horizon, system.n_controls])
    u_traj[:,0] = system.gravity * np.ones(u_traj.shape[0])

    A_list = np.empty((horizon, system.n_states, system.n_states))
    B_list = np.empty((horizon, system.n_states, system.n_controls))
    for i in range(horizon):
        A, B = system.get_linearization(x_traj[i,:], u_traj[i,:]) 
        A_list[i] = A
        B_list[i] = B
    Q_pos = [10,15,15]
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
    distance_threshold = CBF_config["distance_threshold"]
    alpha0 = CBF_config["alpha0"]
    gamma1 = CBF_config["gamma1"]
    gamma2 = CBF_config["gamma2"]
    compensation = CBF_config["compensation"] 

    # Define proxuite problem
    print("==> Define proxuite problem")
    n_CBF = obs_col.n_obsctacles
    n_controls = 4
    cbf_qp = init_proxsuite_qp(n_v=n_controls, n_eq=0, n_in=n_controls+2)

    # Create records
    print("==> Create records")
    times = np.linspace(0, t_final, horizon+1).astype(config.np_dtype)
    states = np.zeros([horizon+1, system.n_states], dtype=config.np_dtype)
    states[0,0:3] = initial_pos
    states[0,3:7] = initial_quat
    controls = np.zeros([horizon, system.n_controls], dtype=config.np_dtype)
    desired_controls = np.zeros([horizon, system.n_controls], dtype=config.np_dtype)
    phi1s = np.zeros(horizon, dtype=config.np_dtype)
    phi2s = np.zeros(horizon, dtype=config.np_dtype)
    smooth_mins = np.zeros(horizon, dtype=config.np_dtype)
    cbf_values = np.zeros([horizon, n_CBF], dtype=config.np_dtype)
    time_cvxpy = np.zeros(horizon, dtype=config.np_dtype)
    time_diff_helper = np.zeros(horizon, dtype=config.np_dtype)
    time_cbf_qp = np.zeros(horizon, dtype=config.np_dtype)
    time_control_loop = np.zeros(horizon, dtype=config.np_dtype)

    video_frames = []
    
    p_prev = states[0,0:3]
    for i in range(horizon):
        if simulator_config["save_video"]:
            rgb_image = env.get_rgb_image(camera=camera_name)
            video_frames.append(rgb_image)

        time_control_loop_start = time.time()
        state = states[i,:]
        P_BB = state[0:3]
        quat_BB = state[3:7]
        R_BB = Rotation.from_quat(quat_BB).as_matrix()
        u_nominal = lqr_controller(state, i)

        time_diff_helper_tmp = 0
        time_cvxpy_tmp = 0
        if CBF_config["active"]:
            # Matrices for the CBF-QP constraints
            C = np.zeros([n_controls+2, n_controls], dtype=config.np_dtype)
            lb = np.zeros(n_controls+2, dtype=config.np_dtype)
            ub = np.zeros(n_controls+2, dtype=config.np_dtype)
            CBF_tmp = np.zeros(n_CBF, dtype=config.np_dtype)
            smooth_min_tmp = 0
            phi1_tmp = 0
            phi2_tmp = 0

            all_h = np.zeros(n_CBF, dtype=config.np_dtype)
            all_h_dx = np.zeros((n_CBF, system.n_states), dtype=config.np_dtype)
            all_h_dxdx = np.zeros((n_CBF, system.n_states, system.n_states), dtype=config.np_dtype)

            # Pass parameter values to cvxpy problem
            ellipse_Q_sqrt_np = D_BB_sqrt @ R_BB.T
            ellipse_Q_np = ellipse_Q_sqrt_np.T @ ellipse_Q_sqrt_np
            _ellipse_Q_sqrt.value = ellipse_Q_sqrt_np
            _ellipse_b.value = -2 * ellipse_Q_np @ P_BB
            _ellipse_c.value = P_BB.T @ ellipse_Q_np @ P_BB

            for (kk, key) in enumerate(obs_col.face_equations.keys()):
                all_vertices_in_world = obs_col.face_equations[key]["vertices_in_world"]
                min_distance = np.linalg.norm(all_vertices_in_world - P_BB, axis=1).min()
                A_obs_np = obs_col.face_equations[key]["A"]
                b_obs_np = obs_col.face_equations[key]["b"]
                if min_distance < distance_threshold:
                    # solve cvxpy problem
                    time_cvxpy_tmp -= time.time()
                    cp_problems[kk].solve(solver=cp.SCS, warm_start=True)
                    time_cvxpy_tmp += time.time()

                    # Solve the scaling function, gradient, and Hessian
                    p_sol_np = np.squeeze(_p.value)
                    time_diff_helper_tmp -= time.time()
                    SF = obstacle_SFs[kk]
                    alpha, alpha_dx_tmp, alpha_dxdx_tmp = DOC.getGradientAndHessian3d(p_sol_np, robot_SF, P_BB, quat_BB, 
                                                                                      SF, np.zeros(3), np.array([0,0,0,1]))
                    time_diff_helper_tmp += time.time()

                    alpha_dx = np.zeros(system.n_states, dtype=config.np_dtype)
                    alpha_dx[0:7] = alpha_dx_tmp

                    alpha_dxdx = np.zeros((system.n_states, system.n_states), dtype=config.np_dtype)
                    alpha_dxdx[0:7,0:7] = alpha_dxdx_tmp

                    h = alpha - alpha0
                    all_h[kk] = h
                    all_h_dx[kk,:] = alpha_dx
                    all_h_dxdx[kk,:,:] = alpha_dxdx
                else:
                    all_h[kk] = np.inf
                    CBF_tmp[kk] = 0

            rho = 2
            min_h = np.min(all_h)
            indices = np.where(rho*(all_h - min_h) < 708)[0]
            h_selected = all_h[indices]
            h_dx_selected = all_h_dx[indices,:]
            h_dxdx_selected = all_h_dxdx[indices,:,:]
            alpha, alpha_dx, alpha_dxdx = DOC.getSmoothMinimumAndTotalGradientAndHessian(rho, h_selected, h_dx_selected, h_dxdx_selected)
            CBF = alpha - 1

            drift = np.squeeze(system.drift(state)) # f(x), shape = (6,)
            actuation = system.actuation(state) # g(x), shape = (6,2)
            drift_jac = system.drift_jac(state) # df/dx, shape = (6,6)

            phi1 = alpha_dx @ drift + gamma1 * CBF # scalar
            C[0,:] = alpha_dx @ drift_jac @ actuation
            lb[0] = -drift.T @ alpha_dxdx @ drift - alpha_dx @ drift_jac @ drift \
                - (gamma1+gamma2) * alpha_dx @ drift - gamma1 * gamma2 * CBF + compensation
            ub[0] = np.inf
            
            tmp = np.array([-C[0,1], C[0,0], 0, 0]).astype(config.np_dtype)
            C[1,:] = tmp

            threshold = 1
            lb[1] = C[1,:] @ system.ueq - 0.4*(CBF-threshold)*(CBF+threshold)
            ub[1] = np.inf

            H = np.diag([1,1,1,1]).astype(config.np_dtype)
            g = -H @ u_nominal
            C[2:,:] = np.eye(n_controls, dtype=config.np_dtype)
            lb[2:] = input_lb
            ub[2:] = input_ub
            cbf_qp.update(H=H, g=g, C=C, l=lb, u=ub)
            time_cbf_qp_start = time.time()
            cbf_qp.solve()
            time_cbf_qp_end = time.time()
            u = cbf_qp.results.x

            time_control_loop_end = time.time()
            smooth_min_tmp = CBF
            phi1_tmp = phi1
            phi2_tmp = C[0,:] @ u - lb[0]
        else:
            u = u_nominal
            time_cbf_qp_start = 0
            time_cbf_qp_end = 0
            time_control_loop_end = time.time()
            smooth_min_tmp = 0
            CBF_tmp = np.zeros(n_CBF, dtype=config.np_dtype)
            phi1_tmp = 0
            phi2_tmp = 0

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
        phi1s[i] = phi1_tmp
        phi2s[i] = phi2_tmp
        smooth_mins[i] = smooth_min_tmp
        time_cvxpy[i] = time_cvxpy_tmp
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

    fig, ax = plt.subplots(figsize=(10,8), dpi=config.dpi, frameon=True)
    plt.plot(times[0:horizon], phi1s, label="phi1")
    plt.plot(times[0:horizon], phi2s, label="phi2")
    plt.axhline(y = 0.0, color = 'black', linestyle = 'dotted', linewidth = 2)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'plot_phi.pdf'))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10,8), dpi=config.dpi, frameon=True)
    plt.plot(times[0:horizon], smooth_mins, label="smooth_mins")
    plt.axhline(y = 0.0, color = 'black', linestyle = 'dotted', linewidth = 2)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'plot_smooth_mins.pdf'))
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

    # Save video
    if simulator_config["save_video"]:
        print("==> Save video")
        video_path = os.path.join(results_dir, "video.mp4")
        media.write_video(video_path, video_frames, fps=1/dt)

    print("==> Done!")
