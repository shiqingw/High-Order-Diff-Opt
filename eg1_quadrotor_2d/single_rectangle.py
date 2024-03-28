import json
import sys
import os
import argparse
import shutil
import numpy as np
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import matplotlib.pyplot as plt
import matplotlib.patches as mp
import time
import cvxpy as cp

from cores.utils.utils import seed_everything, solve_LQR_tracking, save_dict, load_dict, points2d_to_ineq
from cores.utils.proxsuite_utils import init_proxsuite_qp
from cores.dynamical_systems.create_system import get_system
import cores_cpp.diffOptCpp as DOC
from cores.configuration.configuration import Configuration

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_num', default=2, type=int, help='test case number')
    args = parser.parse_args()

    # Create result directory
    exp_num = args.exp_num
    results_dir = "{}/eg1_results/{:03d}".format(str(Path(__file__).parent.parent), exp_num)
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

    # Quadrotor bounding ellipse
    D_sqrt_np = np.diag([1.0/system.bounding_shape_config["semi_major_axis"], 
                        1.0/system.bounding_shape_config["semi_minor_axis"]]).astype(config.np_dtype)
    D_np = D_sqrt_np @ D_sqrt_np.T

    # Obstacles
    obstacle_config = test_settings["obstacle_config"]
    obstacle_artists = []
    obstacle_width = obstacle_config["rec_width"]
    obstacle_height = obstacle_config["rec_height"]
    obstacle_center = np.array(obstacle_config["rec_center"], dtype=config.np_dtype)
    obstacle_left_bottom = obstacle_center - np.array([obstacle_width/2, obstacle_height/2], dtype=config.np_dtype)
    obstacle_rot_angle = obstacle_config["obstacle_rot_angle"]
    obstacle_kappa = obstacle_config["kappa"]
    rectangle = mp.Rectangle(xy=obstacle_left_bottom, width=obstacle_width, height=obstacle_height, angle=obstacle_rot_angle/np.pi*180,
                          rotation_point=obstacle_config["rec_rotation_point"],
                          facecolor="tab:blue", alpha=1, edgecolor="black", linewidth=1, zorder=1.8)
    obstacle_artists.append(rectangle)
    obstacle_corners_in_b = np.array([[obstacle_width/2, obstacle_height/2],
                                [obstacle_width/2, -obstacle_height/2],
                                [-obstacle_width/2, -obstacle_height/2],
                                [-obstacle_width/2, obstacle_height/2]], dtype=config.np_dtype)
    rotation_point = obstacle_center
    obstacle_R_b_to_w = np.array([[np.cos(obstacle_rot_angle), -np.sin(obstacle_rot_angle)],
                                [np.sin(obstacle_rot_angle), np.cos(obstacle_rot_angle)]], dtype=config.np_dtype)
    obstacle_corners_in_w = obstacle_corners_in_b @ obstacle_R_b_to_w.T + rotation_point
    A_obs_np, b_obs_np = points2d_to_ineq(obstacle_corners_in_w)
    n_cons_obs = A_obs_np.shape[0]

    # Define cvxpy problem
    cvxpy_config = test_settings["cvxpy_config"]
    x_max_np = cvxpy_config["x_max"]
    print("==> Define cvxpy problem")
    _p = cp.Variable(2)
    _ellipse_Q_sqrt = cp.Parameter((2,2))
    _ellipse_b = cp.Parameter(2)
    _ellipse_c = cp.Parameter()
    obj = cp.Minimize(cp.sum_squares(_ellipse_Q_sqrt @ _p) + _ellipse_b.T @ _p + _ellipse_c)
    cons = [cp.log_sum_exp(obstacle_kappa*(A_obs_np @ _p + b_obs_np)-x_max_np) - np.log(n_cons_obs) + x_max_np <= 0]
    problem = cp.Problem(obj, cons)
    assert problem.is_dcp()
    assert problem.is_dpp()

    # Tracking control via LQR
    t_final = 10
    dt = system.delta_t
    horizon = int(t_final/dt)

    x_traj = np.zeros([horizon+1, system.n_states]).astype(config.np_dtype)
    t_traj = np.linspace(0, t_final, horizon+1)
    angular_vel_traj = 2*np.pi/t_final
    semi_major_axis = 2
    semi_minor_axis = 1
    x_traj[:,0] = semi_major_axis * np.cos(angular_vel_traj*t_traj) # x
    x_traj[:,1] = semi_minor_axis * np.sin(angular_vel_traj*t_traj) # y
    x_traj[:,3] = -semi_major_axis * angular_vel_traj * np.sin(angular_vel_traj*t_traj) # xdot
    x_traj[:,4] = semi_minor_axis * angular_vel_traj * np.cos(angular_vel_traj*t_traj) # ydot
    u_traj = 0.5 * system.mass * system.gravity * np.ones([horizon, system.n_controls])
    A_list = np.empty((horizon, system.n_states, system.n_states))
    B_list = np.empty((horizon, system.n_states, system.n_controls))
    for i in range(horizon):
        A, B = system.get_linearization(x_traj[i,:], u_traj[i,:]) 
        A_list[i] = A
        B_list[i] = B

    Q_lqr = np.diag([100,100,0,100,100,0])
    Q_list = np.array([Q_lqr]*(horizon +1))
    R_lqr = np.diag([1,1])
    R_list = np.array([R_lqr]*(horizon))
    print("==> Solve LQR gain")
    K_gains, k_feedforward = solve_LQR_tracking(A_list, B_list, Q_list, R_list, x_traj, horizon)

    def lqr_controller(state, i):
        K = K_gains[i]
        k = k_feedforward[i]
        return K @ state + k + u_traj[i,:]
    
    # Define proxuite problem
    print("==> Define proxuite problem")
    # n_in = collision avoidance + box constraints on the control inputs
    cbf_qp = init_proxsuite_qp(n_v=system.n_controls, n_eq=0, n_in=system.n_controls+1)

    # CBF parameters
    CBF_config = test_settings["CBF_config"]
    alpha0 = CBF_config["alpha0"]
    gamma1 = CBF_config["gamma1"]
    gamma2 = CBF_config["gamma2"]

    # Create records
    print("==> Create records")
    times = np.linspace(0, t_final, horizon+1).astype(config.np_dtype)
    states = np.zeros([horizon+1, system.n_states], dtype=config.np_dtype)
    states[0,:] = x_traj[0,:] + np.array([0,0.0,0,0,0,0])
    controls = np.zeros([horizon, system.n_controls], dtype=config.np_dtype)
    desired_controls = np.zeros([horizon, system.n_controls], dtype=config.np_dtype)
    phi1s = np.zeros(horizon, dtype=config.np_dtype)
    phi2s = np.zeros(horizon, dtype=config.np_dtype)
    cbf_values = np.zeros(horizon, dtype=config.np_dtype)
    time_cvxpy = np.zeros(horizon, dtype=config.np_dtype)
    time_diff_helper = np.zeros(horizon, dtype=config.np_dtype)
    time_cbf_qp = np.zeros(horizon, dtype=config.np_dtype)
    time_control_loop = np.zeros(horizon, dtype=config.np_dtype)
    if test_settings["debug_constraints"]:
        debug_constraint_robot = []
        debug_constraint_obstacle = []
        x = np.linspace(-4, 4, 400)
        y = np.linspace(-4, 4, 400)
        X, Y = np.meshgrid(x, y)
        X = np.reshape(X, (-1,))
        Y = np.reshape(Y, (-1,))
        def F_obs(X,Y):
            XY_vec = np.column_stack((X,Y))
            return np.log(np.sum(np.exp(obstacle_kappa*(XY_vec @ A_obs_np.T + b_obs_np)-x_max_np), axis=1))- np.log(n_cons_obs) + x_max_np + 1
        def F_robot(X,Y,a,R):
            XY_vec = np.column_stack((X,Y))
            return np.sum(np.square((XY_vec-a) @ (D_sqrt_np @ R.T).T ), axis=1)

    # Forward simulate the system
    print("==> Forward simulate the system")
    for i in range(horizon):
        time_control_loop_start = time.time()
        state = states[i,:]
        drone_pos_np = state[0:2]
        drone_ori_np = state[2]

        R_b_to_w_np = np.array([[np.cos(drone_ori_np), -np.sin(drone_ori_np)],
                            [np.sin(drone_ori_np), np.cos(drone_ori_np)]]).astype(config.np_dtype)

        if test_settings["debug_constraints"]:
            F_robot_values = F_robot(X,Y,drone_pos_np,R_b_to_w_np)
            F_obs_values = F_obs(X,Y)
            F_robot_satisfied = np.column_stack((X[F_robot_values <= 1], Y[F_robot_values <= 1]))
            F_obs_satisfied = np.column_stack((X[F_obs_values <= 1], Y[F_obs_values <= 1]))
            debug_constraint_robot.append(F_robot_satisfied)
            debug_constraint_obstacle.append(F_obs_satisfied)

        # Pass parameter values to cvxpy problem
        ellipse_Q_sqrt_np = D_sqrt_np @ R_b_to_w_np.T
        ellipse_Q_np = ellipse_Q_sqrt_np.T @ ellipse_Q_sqrt_np
        _ellipse_Q_sqrt.value = ellipse_Q_sqrt_np
        _ellipse_b.value = -2 * ellipse_Q_np @ drone_pos_np
        _ellipse_c.value = drone_pos_np.T @ ellipse_Q_np @ drone_pos_np

        try:
            time_cvxpy_start = time.time()
            problem.solve(warm_start=True, solver=cp.ECOS)
            time_cvxpy_end = time.time()
            # Solve the scaling function, gradient, and Hessian
            p_sol_np = np.squeeze(_p.value)
            time_diff_helper_start = time.time()
            alpha, alpha_dx_tmp, alpha_dxdx_tmp = DOC.getGradientAndHessianEllipseAndLogSumExp(p_sol_np, drone_pos_np,
                                            drone_ori_np, D_np, R_b_to_w_np, A_obs_np, b_obs_np, obstacle_kappa)
            time_diff_helper_end = time.time()
            
            # Evaluate the CBF
            CBF = alpha - alpha0
        except Exception as e:
            CBF = 0
            time_cvxpy_start = 0
            time_cvxpy_end = 0
            time_diff_helper_start = 0
            time_diff_helper_end = 0
        
        # Nominal control
        u_nominal = lqr_controller(state, i)

        if CBF_config["active"]:
            # Order of states = [x, y, theta, vx, vy omega]
            # Order of parameters in grad and hessian: [theta, x, y]
            alpha_dx = np.zeros(system.n_states, dtype=config.np_dtype)
            alpha_dx[0:2] = alpha_dx_tmp[1:3]
            alpha_dx[2] = alpha_dx_tmp[0]
            alpha_dxdx = np.zeros((system.n_states, system.n_states), dtype=config.np_dtype)
            alpha_dxdx[0:2,0:2] = alpha_dxdx_tmp[1:3,1:3]
            alpha_dxdx[2,2] = alpha_dxdx_tmp[0,0]
            alpha_dxdx[0:2,2] = alpha_dxdx_tmp[0,1:3]
            alpha_dxdx[2,0:2] = alpha_dxdx_tmp[1:3,0]

            drift = np.squeeze(system.drift(state)) # f(x), shape = (6,)
            actuation = system.actuation(state) # g(x), shape = (6,2)
            drift_jac = system.drift_jac(state) # df/dx, shape = (6,6)

            phi1 = alpha_dx @ drift + gamma1 * CBF # scalar
            g = -u_nominal # shape = (2,)
            C = np.zeros([system.n_controls+1,system.n_controls], dtype=config.np_dtype)
            C[0,:] = alpha_dx @ drift_jac @ actuation
            C[1:system.n_controls+1,:] = np.eye(system.n_controls)
            lb = np.zeros(system.n_controls+1, dtype=config.np_dtype)
            ub = np.zeros(system.n_controls+1, dtype=config.np_dtype) 
            lb[0] = -drift.T @ alpha_dxdx @ drift - alpha_dx @ drift_jac @ drift \
                - (gamma1+gamma2) * alpha_dx @ drift - gamma1 * gamma2 * CBF
            ub[0] = np.inf
            lb[1] = system.u1_constraint[0]
            ub[1] = system.u1_constraint[1]
            lb[2] = system.u2_constraint[0]
            ub[2] = system.u2_constraint[1]
            cbf_qp.update(g=g, C=C, l=lb, u=ub)
            time_cbf_qp_start = time.time()
            cbf_qp.solve()
            time_cbf_qp_end = time.time()
            u_safe = cbf_qp.results.x
            phi2 = -lb[0] + C[0,:] @ u_safe
            u = u_safe
        else:
            u = u_nominal
            phi1 = 0
            phi2 = 0
            time_cbf_qp_start = 0
            time_cbf_qp_end = 0

        time_control_loop_end = time.time()

        # Step the simulation
        new_state = system.get_next_state(state, u)
        states[i+1,:] = new_state

        # Record
        states[i+1,:] = new_state
        controls[i,:] = u
        desired_controls[i,:] = u_nominal
        cbf_values[i] = CBF
        phi1s[i] = phi1
        phi2s[i] = phi2
        time_cvxpy[i] = time_cvxpy_end - time_cvxpy_start
        time_diff_helper[i] = time_diff_helper_end - time_diff_helper_start
        time_cbf_qp[i] = time_cbf_qp_end - time_cbf_qp_start
        time_control_loop[i] = time_control_loop_end - time_control_loop_start

    # Create animation
    print("==> Create animation")
    save_video_path = "{}/video.mp4".format(results_dir)
    if test_settings["debug_constraints"]:
        system.animate_robot(states, controls, dt, save_video_path, plot_bounding_ellipse=True, plot_traj=True,
                         obstacles=obstacle_artists, robot_constraints=debug_constraint_robot,
                         obstacle_constraints=debug_constraint_obstacle)
    else:
        system.animate_robot(states, controls, dt, save_video_path, plot_bounding_ellipse=True, plot_traj=True,
                         obstacles=obstacle_artists)

    # Save summary
    print("==> Save dictionary")
    summary = {"times": times,
               "states": states,
               "controls": controls,
               "desired_controls": desired_controls,
               "x_traj": x_traj,
               "u_traj": u_traj,
               "K_gains": K_gains,
               "k_feedforward": k_feedforward,
               "phi1s": phi1s,
               "phi2s": phi2s,
               "cbf_values": cbf_values}
    save_dict(summary, os.path.join(results_dir, 'summary.pkl'))
    
    # Draw plots
    print("==> Draw plots")
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams.update({"text.usetex": True,
                         "text.latex.preamble": r"\usepackage{amsmath}"})
    plt.rcParams.update({'pdf.fonttype': 42})

    fig, ax = plt.subplots(figsize=(10,8), dpi=config.dpi, frameon=True)
    plt.plot(times, x_traj[:,0], color="tab:blue", linestyle=":", label="x desired")
    plt.plot(times, x_traj[:,1], color="tab:green", linestyle=":", label="y desired")
    plt.plot(times, states[:,0], color="tab:blue", linestyle="-", label="x")
    plt.plot(times, states[:,1], color="tab:green", linestyle="-", label="y")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'plot_traj_x_y.pdf'))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10,8), dpi=config.dpi, frameon=True)
    plt.plot(times, states[:,0], linestyle="-", label=r"$x$")
    plt.plot(times, states[:,1], linestyle="-", label=r"$y$")
    plt.plot(times, states[:,2], linestyle="-", label=r"$\theta$")
    plt.plot(times, states[:,3], linestyle="-", label=r"$v_x$")
    plt.plot(times, states[:,4], linestyle="-", label=r"$v_y$")
    plt.plot(times, states[:,5], linestyle="-", label=r"$\omega$")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'plot_traj.pdf'))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10,8), dpi=config.dpi, frameon=True)
    plt.plot(times[:len(controls)], desired_controls[:,0], color="tab:blue", linestyle=":", 
             label="u_1 nominal")
    plt.plot(times[:len(controls)], desired_controls[:,1], color="tab:green", linestyle=":",
              label="u_2 nominal")
    plt.plot(times[:len(controls)], controls[:,0], color="tab:blue", linestyle="-", label="u_1")
    plt.plot(times[:len(controls)], controls[:,1], color="tab:green", linestyle="-", label="u_2")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'plot_controls.pdf'))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10,8), dpi=config.dpi, frameon=True)
    plt.plot(times[:len(phi1s)], phi1s, label="phi1")
    plt.plot(times[:len(phi2s)], phi2s, label="phi2")
    plt.axhline(y = 0.0, color = 'black', linestyle = 'dotted', linewidth = 2)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'plot_phi.pdf'))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10,8), dpi=config.dpi, frameon=True)
    plt.plot(times[:len(cbf_values)], cbf_values, label="CBF value")
    plt.axhline(y = 0.0, color = 'black', linestyle = 'dotted', linewidth = 2)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'plot_cbf.pdf'))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10,8), dpi=config.dpi, frameon=True)
    plt.plot(times[:len(time_cvxpy)], time_cvxpy, label="cvxpy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'plot_time_cvxpy.pdf'))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10,8), dpi=config.dpi, frameon=True)
    plt.plot(times[:len(time_diff_helper)], time_diff_helper, label="diff helper")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'plot_time_diff_helper.pdf'))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10,8), dpi=config.dpi, frameon=True)
    plt.plot(times[:len(time_cbf_qp)], time_cbf_qp, label="CBF-QP")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'plot_time_cbf_qp.pdf'))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10,8), dpi=config.dpi, frameon=True)
    plt.plot(times[:len(time_control_loop)], time_control_loop, label="control loop")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'plot_time_control_loop.pdf'))
    plt.close(fig)

    # Print solving time
    print("==> Control loop solving time: {:.6f} s".format(np.mean(time_control_loop)))
    print("==> CVXPY solving time: {:.6f} s".format(np.mean(time_cvxpy)))
    print("==> Diff helper solving time: {:.6f} s".format(np.mean(time_diff_helper)))
    print("==> CBF-QP solving time: {:.6f} s".format(np.mean(time_cbf_qp)))



    
