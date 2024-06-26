import json
import sys
import os
import argparse
import shutil
import numpy as np
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Set the backend to 'Agg'
import time

from fr3_envs.fr3_mj_env_collision import FR3MuJocoEnv
from fr3_envs.bounding_shape_coef_mj import BoundingShapeCoef
from cores.utils.utils import seed_everything, save_dict
from cores.utils.proxsuite_utils import init_proxsuite_qp
import scalingFunctionsHelper as doh
from cores.utils.rotation_utils import get_quat_from_rot_matrix, get_Q_matrix_from_quat, get_dQ_matrix
from cores.configuration.configuration import Configuration
from scipy.spatial.transform import Rotation
from liegroups import SO3
from cores.utils.trajectory_utils import PositionTrapezoidalTrajectory, OrientationTrapezoidalTrajectory
from cores.obstacle_collections.polytope_collection import PolytopeCollection
import cvxpy as cp

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_num', default=5, type=int, help='test case number')
    args = parser.parse_args()

    # Create result directory
    exp_num = args.exp_num
    results_dir = "{}/eg8_results/{:03d}".format(str(Path(__file__).parent.parent), exp_num)
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

    # Various configs
    simulator_config = test_settings["simulator_config"]
    CBF_config = test_settings["CBF_config"]
    trajectory_config = test_settings["trajectory_config"]

    # Joint limits
    joint_limits_config = test_settings["joint_limits_config"]
    joint_lb = np.array(joint_limits_config["lb"], dtype=config.np_dtype)
    joint_ub = np.array(joint_limits_config["ub"], dtype=config.np_dtype)

    # Joint acceleration limits
    joint_acc_limits_config = test_settings["joint_acceleration_limits_config"]
    joint_acc_lb = np.array(joint_acc_limits_config["lb"], dtype=config.np_dtype)
    joint_acc_ub = np.array(joint_acc_limits_config["ub"], dtype=config.np_dtype)

    # Create and reset simulation
    cam_distance = simulator_config["cam_distance"]
    cam_azimuth = simulator_config["cam_azimuth"]
    cam_elevation = simulator_config["cam_elevation"]
    cam_lookat = simulator_config["cam_lookat"]
    base_pos = simulator_config["base_pos"]
    base_quat = simulator_config["base_quat"]
    initial_joint_angles = test_settings["initial_joint_angles"]
    dt = 1.0/240.0

    env = FR3MuJocoEnv(xml_name="fr3_on_table_with_bounding_boxes_circulation_polytope2", base_pos=base_pos, base_quat=base_quat,
                    cam_distance=cam_distance, cam_azimuth=cam_azimuth, cam_elevation=cam_elevation, cam_lookat=cam_lookat, dt=dt)
    info = env.reset(initial_joint_angles)

    # cvxpy config
    solvers = {'ECOS': cp.ECOS, 'SCS': cp.SCS, 'CLARABEL': cp.CLARABEL}
    cvxpy_config = test_settings["cvxpy_config"]
    obstacle_kappa = cvxpy_config["obstacle_kappa"]
    cp_solver = solvers[cvxpy_config["solver"]]
    cp_max_iters = cvxpy_config["max_iters"]

    # Load the bounding shape coefficients
    BB_coefs = BoundingShapeCoef()
    robot_SFs = {}
    for (i, bb_key) in enumerate(CBF_config["selected_bbs"]):
        quadratic_coef = BB_coefs.coefs[bb_key]
        SF = doh.Ellipsoid3d(True, quadratic_coef, np.zeros(3))
        robot_SFs[bb_key] = SF

    # Obstacle
    obstacle_config = test_settings["obstacle_config"]
    n_obstacle = len(obstacle_config)
    obstacle_SFs = {}
    obs_col = PolytopeCollection(3, n_obstacle, obstacle_config)
    id_geom_offset = 0
    for (i, obs_key) in enumerate(obs_col.face_equations.keys()):
        A_obs_np = obs_col.face_equations[obs_key]["A"]
        b_obs_np = obs_col.face_equations[obs_key]["b"]
        SF =doh.LogSumExp3d(False, A_obs_np, b_obs_np, obstacle_kappa)
        obstacle_SFs[obs_key] = SF

        # add visual ellipsoids
        all_points = obs_col.face_equations[obs_key]["vertices_in_world"]
        for j in range(len(all_points)):
            env.add_visual_ellipsoid(0.05*np.ones(3), all_points[j], np.eye(3), np.array([1,0,0,1]),id_geom_offset=id_geom_offset)
            id_geom_offset = env.viewer.user_scn.ngeom

    # Define cvxpy problem
    print("==> Define cvxpy problem")
    x_max_np = cvxpy_config["x_max"]
    cp_problems = {}
    _ellipse_Q_sqrt = cp.Parameter((3,3))
    _ellipse_b = cp.Parameter(3)
    _ellipse_c = cp.Parameter()
    _p = cp.Variable(3)

    for (i, bb_key) in enumerate(CBF_config["selected_bbs"]):
        cp_problem_bb = {}
        D_BB_sqrt = BB_coefs.coefs_sqrt[bb_key]

        for (j, obs_key) in enumerate(obs_col.face_equations.keys()):
            A_obs_np = obs_col.face_equations[obs_key]["A"]
            b_obs_np = obs_col.face_equations[obs_key]["b"]
            n_cons_obs = len(b_obs_np)

            obj = cp.Minimize(cp.sum_squares(_ellipse_Q_sqrt @ _p) + _ellipse_b.T @ _p + _ellipse_c)
            cons = [cp.log_sum_exp(obstacle_kappa*(A_obs_np @ _p + b_obs_np)-x_max_np) - np.log(n_cons_obs) + x_max_np <= 0]
            problem = cp.Problem(obj, cons)
            assert problem.is_dcp()
            assert problem.is_dpp()

            # warm start with fake data
            R_b_to_w_np = np.eye(3)
            bb_pos_np = np.array([0,0,0], dtype=config.np_dtype)
            ellipse_Q_sqrt_np = D_BB_sqrt @ R_b_to_w_np.T
            ellipse_Q_np = ellipse_Q_sqrt_np.T @ ellipse_Q_sqrt_np
            _ellipse_Q_sqrt.value = ellipse_Q_sqrt_np
            _ellipse_b.value = -2 * ellipse_Q_np @ bb_pos_np
            _ellipse_c.value = bb_pos_np.T @ ellipse_Q_np @ bb_pos_np
            problem.solve(warm_start=True, solver=cp_solver)
            cp_problem_bb[obs_key] = problem

        cp_problems[bb_key] = cp_problem_bb

    # Compute desired trajectory
    t_final = 15
    P_EE_0 = np.array([0.2, 0.2, 0.86])
    P_EE_1 = np.array([0.2, -0.3, 0.86])
    P_EE_2 = np.array([0.2, -0.3, 0.86])
    via_points = np.array([P_EE_0, P_EE_1, P_EE_2])
    target_time = np.array([0, 5, t_final])
    Ts = 0.01
    traj_line = PositionTrapezoidalTrajectory(via_points, target_time, T_antp=0.2, Ts=Ts)

    R_EE_0 = np.array([[1, 0, 0],
                        [0, -1, 0],
                        [0, 0, -1]], dtype=config.np_dtype)
    R_EE_1 = np.array([[1, 0, 0],
                        [0, -1, 0],
                        [0, 0, -1]], dtype=config.np_dtype)
    R_EE_2 = R_EE_1
    orientations = np.array([R_EE_0, R_EE_1, R_EE_2], dtype=config.np_dtype)
    target_time = np.array([0, 5, t_final])
    traj_orientation = OrientationTrapezoidalTrajectory(orientations, target_time, Ts=Ts)

    # Visualize the trajectory
    N = 100
    len_traj = len(traj_line.t)
    sampled = np.linspace(0, len_traj-1, N).astype(int)
    sampled = traj_line.pd[sampled]
    id_geom_offset = 0
    for i in range(N-1):
        env.add_visual_capsule(sampled[i], sampled[i+1], 0.004, np.array([0,0,1,1]), id_geom_offset)
        id_geom_offset = env.viewer.user_scn.ngeom 
    env.viewer.sync()

    # CBF parameters
    CBF_config = test_settings["CBF_config"]
    alpha0 = CBF_config["alpha0"]
    f0 = CBF_config["f0"]
    gamma1 = CBF_config["gamma1"]
    gamma2 = CBF_config["gamma2"]
    compensation = CBF_config["compensation"] 
    selected_BBs = CBF_config["selected_bbs"]
    n_controls = 9

    # Define proxuite problem
    print("==> Define proxuite problem")
    n_CBF = len(robot_SFs)*len(obstacle_SFs)
    cbf_qp = init_proxsuite_qp(n_v=n_controls, n_eq=0, n_in=n_controls+n_CBF)

    # Create records
    print("==> Create records")
    horizon = int(t_final/dt)
    times = np.linspace(0, (horizon-1)*dt, horizon)
    joint_angles = np.zeros([horizon, n_controls], dtype=config.np_dtype)
    controls = np.zeros([horizon, 7], dtype=config.np_dtype)
    desired_controls = np.zeros([horizon, n_controls], dtype=config.np_dtype)
    phi1s = np.zeros([horizon, n_CBF], dtype=config.np_dtype)
    phi2s = np.zeros([horizon, n_CBF], dtype=config.np_dtype)
    cbf_values = np.zeros([horizon, n_CBF], dtype=config.np_dtype)
    time_cvxpy = np.zeros(horizon, dtype=config.np_dtype)
    time_diff_helper = np.zeros(horizon, dtype=config.np_dtype)
    time_cbf_qp = np.zeros(horizon, dtype=config.np_dtype)
    time_control_loop = np.zeros(horizon, dtype=config.np_dtype)
    all_info = []

    # Forward simulate the system
    P_EE_prev = info["P_EE"]
    time_prev = time.time()
    print("==> Forward simulate the system")
    for i in range(horizon):
        all_info.append(info)

        time_control_loop_start = time.time()

        # Update info
        q = info["q"]
        dq = info["dq"]
        nle = info["nle"]
        Minv = info["Minv"]
        M = info["M"]
        G = info["G"]

        Minv_mj = info["Minv_mj"]
        M_mj = info["M_mj"]
        nle_mj = info["nle_mj"]

        P_EE = info["P_EE"]
        R_EE = info["R_EE"]
        J_EE = info["J_EE"]
        dJdq_EE = info["dJdq_EE"]
        v_EE = J_EE @ dq

        # Visualize the trajectory
        speed = np.linalg.norm((P_EE-P_EE_prev)/dt)
        rgba=np.array((np.clip(speed/10, 0, 1),
                     np.clip(1-speed/10, 0, 1),
                     .5, 1.))
        radius=.003*(1+speed)
        env.add_visual_capsule(P_EE_prev, P_EE, radius, rgba, id_geom_offset, True)

        # Primary obejctive: tracking control
        K_p_pos = np.diag([100,100,100]).astype(config.np_dtype)
        K_d_pos = np.diag([50,50,50]).astype(config.np_dtype)
        traj_pos, traj_pos_dt, traj_pos_dtdt = traj_line.get_traj_and_ders(i*dt)
        e_pos = P_EE - traj_pos # shape (3,)
        e_pos_dt = v_EE[:3] - traj_pos_dt # shape (3,)
        v_dt = traj_pos_dtdt - K_p_pos @ e_pos - K_d_pos @ e_pos_dt

        K_p_rot = np.diag([200,200,200]).astype(config.np_dtype)
        K_d_rot = np.diag([100,100,100]).astype(config.np_dtype)
        traj_ori, traj_ori_dt, traj_ori_dtdt = traj_orientation.get_traj_and_ders(i*dt)
        e_rot = SO3(R_EE @ traj_ori.T).log() # shape (3,)
        e_rot_dt = v_EE[3:]-traj_ori_dt # shape (3,)
        omega_dt = traj_ori_dtdt - K_p_rot @ e_rot - K_d_rot @ e_rot_dt

        v_EE_dt_desired = np.concatenate([v_dt, omega_dt])
        S = J_EE
        S_pinv = S.T @ np.linalg.pinv(S @ S.T + 0.01* np.eye(S.shape[0]))
        S_null = (np.eye(len(q)) - S_pinv @ S)
        q_dtdt_task = S_pinv @ (v_EE_dt_desired - dJdq_EE)

        # Secondary objective: encourage the joints to remain close to the initial configuration
        W = np.diag(1.0/(joint_ub-joint_lb))
        q_bar = 1/2*(joint_ub+joint_lb)
        e_joint = W @ (q - q_bar)
        e_joint_dot = W @ dq
        Kp_joint = 80*np.diag([1, 1, 1, 1, 1, 1, 1, 1, 1]).astype(config.np_dtype)
        Kd_joint = 40*np.diag([1, 1, 1, 1, 1, 1, 1, 1, 1]).astype(config.np_dtype)
        q_dtdt = q_dtdt_task + S_null @ (- Kp_joint @ e_joint - Kd_joint @ e_joint_dot)

        # Map to torques
        u_nominal = q_dtdt

        time_diff_helper_tmp = 0
        time_cvxpy_tmp = 0
        if CBF_config["active"]:
            # Matrices for the CBF-QP constraints
            C = np.zeros([n_controls+n_CBF, n_controls], dtype=config.np_dtype)
            lb = np.zeros(n_controls+n_CBF, dtype=config.np_dtype)
            ub = np.zeros(n_controls+n_CBF, dtype=config.np_dtype)
            CBF_tmp = np.zeros(n_CBF, dtype=config.np_dtype)
            phi1_tmp = np.zeros(n_CBF, dtype=config.np_dtype)
            phi2_tmp = np.zeros(n_CBF, dtype=config.np_dtype)

            all_h = np.zeros(n_CBF, dtype=config.np_dtype)
            first_order_all_average_scalar = np.zeros(n_CBF, dtype=config.np_dtype)
            second_order_all_average_scalar = np.zeros(n_CBF, dtype=config.np_dtype)
            second_order_all_average_vector = np.zeros((n_CBF, 9), dtype=config.np_dtype)

            n_obs = len(obstacle_SFs)

            for kk in range(len(selected_BBs)):
                bb_key = selected_BBs[kk]
                P_BB = info["P_"+bb_key]
                R_BB = info["R_"+bb_key]
                J_BB = info["J_"+bb_key]
                dJdq_BB = info["dJdq_"+bb_key]
                v_BB = J_BB @ dq
                D_BB = BB_coefs.coefs[bb_key]
                quat_BB = get_quat_from_rot_matrix(R_BB)

                A_BB = np.zeros((7,6), dtype=config.np_dtype)
                Q_BB = get_Q_matrix_from_quat(quat_BB) # shape (4,3)
                A_BB[0:3,0:3] = np.eye(3, dtype=config.np_dtype)
                A_BB[3:7,3:6] = Q_BB
                dx_BB = A_BB @ v_BB

                dquat_BB = 0.5 * Q_BB @ v_BB[3:6] # shape (4,)
                dQ_BB = get_dQ_matrix(dquat_BB) # shape (4,3)
                dA_BB = np.zeros((7,6), dtype=config.np_dtype)
                dA_BB[3:7,3:6] = dQ_BB

                SF1 = robot_SFs[bb_key]

                # Pass parameter values to cvxpy problem
                D_BB_sqrt = BB_coefs.coefs_sqrt[bb_key]
                ellipse_Q_sqrt_np = D_BB_sqrt @ R_BB.T
                ellipse_Q_np = ellipse_Q_sqrt_np.T @ ellipse_Q_sqrt_np
                _ellipse_Q_sqrt.value = ellipse_Q_sqrt_np
                _ellipse_b.value = -2 * ellipse_Q_np @ P_BB
                _ellipse_c.value = P_BB.T @ ellipse_Q_np @ P_BB

                for (ll, obs_key) in enumerate(obstacle_SFs.keys()):
                    SF2 = obstacle_SFs[obs_key]
                    all_vertices_in_world = obs_col.face_equations[obs_key]["vertices_in_world"]
                    min_distance = np.linalg.norm(all_vertices_in_world - P_BB, axis=1).min()
                    A_obs_np = obs_col.face_equations[obs_key]["A"]
                    b_obs_np = obs_col.face_equations[obs_key]["b"]

                    if min_distance < 10:

                        # solve cvxpy problem
                        time_cvxpy_tmp -= time.time()
                        cp_problems[bb_key][obs_key].solve(warm_start=True, solver=cp_solver)
                        time_cvxpy_tmp += time.time()

                        p_sol_np = np.squeeze(_p.value)
                        time_diff_helper_tmp -= time.time()
                        alpha, alpha_dx, alpha_dxdx = doh.getGradientAndHessian3d(p_sol_np, SF1, P_BB, quat_BB, 
                                                                                SF2, np.zeros(3), np.array([0,0,0,1]))
                        time_diff_helper_tmp += time.time()

                        h = alpha - alpha0
                        h_dx = alpha_dx
                        h_dxdx = alpha_dxdx

                        dh = h_dx @ dx_BB
                        phi1 = dh + gamma1 * h
                        
                        C[kk*n_obs+ll,:] = h_dx @ A_BB @ J_BB
                        lb[kk*n_obs+ll] = - gamma2*phi1 - gamma1*dh - dx_BB.T @ h_dxdx @ dx_BB - h_dx @ dA_BB @ v_BB \
                            - h_dx @ A_BB @ dJdq_BB  + compensation
                        ub[kk*n_obs+ll] = np.inf

                        CBF_tmp[kk*n_obs+ll] = h
                        phi1_tmp[kk*n_obs+ll] = phi1
                    else:
                        CBF_tmp[kk*n_obs+ll] = 0
                        phi1_tmp[kk*n_obs+ll] = 0

            # CBF-QP constraints
            print(np.min(CBF_tmp))
            g = -u_nominal
            C[n_CBF:n_CBF+n_controls,:] = np.eye(n_controls, dtype=config.np_dtype)
            lb[n_CBF:] = joint_acc_lb
            ub[n_CBF:] = joint_acc_ub
            cbf_qp.update(g=g, C=C, l=lb, u=ub)
            time_cbf_qp_start = time.time()
            cbf_qp.solve()
            time_cbf_qp_end = time.time()
            u = cbf_qp.results.x
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
        u = M_mj @ u + nle_mj
        u = u[:7]
        finger_pos = 0.01
        info = env.step(tau=u, finger_pos=finger_pos)
        time.sleep(max(0,dt-time_control_loop_end+time_control_loop_start))

        # Record
        P_EE_prev = P_EE
        joint_angles[i,:] = q
        controls[i,:] = u
        desired_controls[i,:] = u_nominal
        cbf_values[i,:] = CBF_tmp
        phi1s[i,:] = phi1_tmp
        phi2s[i,:] = phi2_tmp
        time_cvxpy[i] = time_cvxpy_tmp
        time_diff_helper[i] = time_diff_helper_tmp
        time_cbf_qp[i] = time_cbf_qp_end - time_cbf_qp_start
        time_control_loop[i] = time_control_loop_end - time_control_loop_start

    # Close the environment
    env.close()

    # Save summary
    print("==> Save results")
    summary = {"times": times,
               "joint_angles": joint_angles,
               "controls": controls,
               "desired_controls": desired_controls,
               "phi1s": phi1s,
               "phi2s": phi2s,
               "cbf_values": cbf_values,
               "time_cvxpy": time_cvxpy,
               "time_diff_helper": time_diff_helper,
               "time_cbf_qp": time_cbf_qp}
    save_dict(summary, os.path.join(results_dir, 'summary.pkl'))

    print("==> Save all_info")
    save_dict(all_info, os.path.join(results_dir, 'all_info.pkl'))

    # Print solving time
    print("==> Control loop solving time: {:.5f} s".format(np.mean(time_control_loop)))
    print("==> CVXPY solving time: {:.5f} s".format(np.mean(time_cvxpy)))
    print("==> Diff helper solving time: {:.5f} s".format(np.mean(time_diff_helper)))
    print("==> CBF-QP solving time: {:.5f} s".format(np.mean(time_cbf_qp)))
    
    # Visualization
    print("==> Draw plots")
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams.update({"text.usetex": True,
                         "text.latex.preamble": r"\usepackage{amsmath}"})
    plt.rcParams.update({'pdf.fonttype': 42})

    fig, ax = plt.subplots(figsize=(10,8), dpi=config.dpi, frameon=True)
    plt.plot(times, joint_angles[:,0], linestyle="-", label=r"$q_1$")
    plt.plot(times, joint_angles[:,1], linestyle="-", label=r"$q_2$")
    plt.plot(times, joint_angles[:,2], linestyle="-", label=r"$q_3$")
    plt.plot(times, joint_angles[:,3], linestyle="-", label=r"$q_4$")
    plt.plot(times, joint_angles[:,4], linestyle="-", label=r"$q_5$")
    plt.plot(times, joint_angles[:,5], linestyle="-", label=r"$q_6$")
    plt.plot(times, joint_angles[:,6], linestyle="-", label=r"$q_7$")
    plt.plot(times, joint_angles[:,7], linestyle="-", label=r"$q_8$")
    plt.plot(times, joint_angles[:,8], linestyle="-", label=r"$q_9$")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'plot_joint_angles.pdf'))
    plt.close(fig)

    for i in range(7):
        fig, ax = plt.subplots(figsize=(10,8), dpi=config.dpi, frameon=True)
        plt.plot(times, desired_controls[:,i], color="tab:blue", linestyle=":", 
                label="u_{:d} nominal".format(i+1))
        plt.plot(times, controls[:,i], color="tab:blue", linestyle="-", label="u_{:d}".format(i+1))
        plt.axhline(y = joint_acc_lb[i], color = 'black', linestyle = 'dotted', linewidth = 2)
        plt.axhline(y = joint_acc_ub[i], color = 'black', linestyle = 'dotted', linewidth = 2)
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

    for i in range(n_CBF):
        fig, ax = plt.subplots(figsize=(10,8), dpi=config.dpi, frameon=True)
        plt.plot(times, phi1s[:,i], label="phi1")
        plt.plot(times, phi2s[:,i], label="phi2")
        plt.axhline(y = 0.0, color = 'black', linestyle = 'dotted', linewidth = 2)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'plot_phi_{:d}.pdf'.format(i+1)))
        plt.close(fig)
    
    for i in range(n_CBF):
        fig, ax = plt.subplots(figsize=(10,8), dpi=config.dpi, frameon=True)
        plt.plot(times, cbf_values[:,i], label="CBF")
        plt.axhline(y = 0.0, color = 'black', linestyle = 'dotted', linewidth = 2)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'plot_cbf_{:d}.pdf'.format(i+1)))
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(10,8), dpi=config.dpi, frameon=True)
    plt.plot(times, time_cvxpy, label="cvxpy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'plot_time_cvxpy.pdf'))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10,8), dpi=config.dpi, frameon=True)
    plt.plot(times, time_diff_helper, label="diff helper")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'plot_time_diff_helper.pdf'))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10,8), dpi=config.dpi, frameon=True)
    plt.plot(times, time_cbf_qp, label="CBF-QP")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'plot_time_cbf_qp.pdf'))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10,8), dpi=config.dpi, frameon=True)
    plt.plot(times, time_control_loop, label="control loop")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'plot_time_control_loop.pdf'))
    plt.close(fig)

    print("==> Done!")
