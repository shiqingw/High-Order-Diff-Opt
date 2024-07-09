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
import scalingFunctionsHelperPy as sfh
import HOCBFHelperPy as hh
from cores.utils.rotation_utils import get_quat_from_rot_matrix
from cores.configuration.configuration import Configuration
from liegroups import SO3
from cores.utils.trajectory_utils import PositionTrapezoidalTrajectory, OrientationTrapezoidalTrajectory
from cores.obstacle_collections.polytope_collection import PolytopeCollection
import multiprocessing
from scipy.spatial.transform import Rotation

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_num', default=7, type=int, help='test case number')
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

    # Joint input_torque limits
    input_torque_limits_config = test_settings["input_torque_limits_config"]
    input_torque_lb = np.array(input_torque_limits_config["lb"], dtype=config.np_dtype)
    input_torque_ub = np.array(input_torque_limits_config["ub"], dtype=config.np_dtype)

    # Create and reset simulation
    cam_distance = simulator_config["cam_distance"]
    cam_azimuth = simulator_config["cam_azimuth"]
    cam_elevation = simulator_config["cam_elevation"]
    cam_lookat = simulator_config["cam_lookat"]
    base_pos = simulator_config["base_pos"]
    base_quat = simulator_config["base_quat"]
    initial_joint_angles = test_settings["initial_joint_angles"]
    dt = 1.0/240.0

    env = FR3MuJocoEnv(xml_name="fr3_on_table_with_bounding_boxes_circulation_polytope3", base_pos=base_pos, base_quat=base_quat,
                    cam_distance=cam_distance, cam_azimuth=cam_azimuth, cam_elevation=cam_elevation, cam_lookat=cam_lookat, dt=dt)
    info = env.reset(initial_joint_angles)

    # cvxpy config
    cvxpy_config = test_settings["cvxpy_config"]
    obstacle_kappa = cvxpy_config["obstacle_kappa"]

    # Load the bounding shape coefficients
    BB_coefs = BoundingShapeCoef()

    # Obstacle
    obstacle_config = test_settings["obstacle_config"]
    n_obstacle = len(obstacle_config)
    obs_col = PolytopeCollection(3, n_obstacle, obstacle_config)
    id_geom_offset = 0
    for (i, obs_key) in enumerate(obs_col.face_equations.keys()):
        # add visual ellipsoids
        all_points = obs_col.face_equations[obs_key]["vertices_in_world"]
        for j in range(len(all_points)):
            env.add_visual_ellipsoid(0.01*np.ones(3), all_points[j], np.eye(3), np.array([1,0,0,1]),id_geom_offset=id_geom_offset)
            id_geom_offset = env.viewer.user_scn.ngeom

    # Compute desired trajectory
    t_final = 30
    t_1 = 10
    t_2 = 20
    P_EE_0 = np.array([-0.10, 0.26, 1.31])
    P_EE_1 = np.array([-0.07, 0.61, 0.86])
    P_EE_2 = np.array([0.20, 0.25, 0.86])
    via_points = np.array([P_EE_0, P_EE_1, P_EE_2, P_EE_2])
    target_time = np.array([0, t_1, t_2, t_final])
    Ts = 0.01
    traj_line = PositionTrapezoidalTrajectory(via_points, target_time, T_antp=0.2, Ts=Ts)

    roll = np.pi
    pitch = 0
    yaw = np.pi/3
    R_EE_0 = Rotation.from_euler('xyz', [roll, pitch, yaw]).as_matrix()

    roll = np.pi
    pitch = 0
    yaw = np.pi/3
    R_EE_1 = Rotation.from_euler('xyz', [roll, pitch, yaw]).as_matrix()

    roll = np.pi
    pitch = 0
    yaw = np.pi/6
    R_EE_2 = Rotation.from_euler('xyz', [roll, pitch, yaw]).as_matrix()

    orientations = np.array([R_EE_0, R_EE_1, R_EE_2, R_EE_2], dtype=config.np_dtype)
    target_time = np.array([0, t_1, t_2, t_final])
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
    n_selected_BBs = len(selected_BBs)
    n_controls = 7

    # Define proxuite problem
    print("==> Define proxuite problem")
    n_CBF = n_selected_BBs*n_obstacle
    cbf_qp = init_proxsuite_qp(n_v=n_controls, n_eq=0, n_in=n_controls+n_CBF)

    # Create workers
    print("==> Create workers")
    n_threads = max(multiprocessing.cpu_count() -1, 1)
    probs = hh.Problem3dCollection(n_threads)
    for (i, bb_key) in enumerate(selected_BBs):
        for (j, obs_key) in enumerate(obs_col.face_equations.keys()):
            frame_id = i

            ellipsoid_quadratic_coef = BB_coefs.coefs[bb_key]
            SF_rob = sfh.Ellipsoid3d(True, ellipsoid_quadratic_coef, np.zeros(3))

            A_obs_np = obs_col.face_equations[obs_key]["A"]
            b_obs_np = obs_col.face_equations[obs_key]["b"]
            obs_kappa = obstacle_kappa
            vertices = obs_col.face_equations[obs_key]["vertices_in_world"]
            SF_obs = sfh.LogSumExp3d(False, A_obs_np, b_obs_np, obs_kappa)

            prob = hh.ElliposoidAndLogSumExp3dPrb(SF_rob, SF_obs, vertices)

            probs.addProblem(prob, frame_id)

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
    time_cvxpy_and_diff_helper = np.zeros(horizon, dtype=config.np_dtype)
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
        q = info["q"][:7]
        dq = info["dq"][:7]
        nle = info["nle"][:7]
        Minv = info["Minv"][:7,:7]
        M = info["M"][:7,:7]
        G = info["G"][:7]

        Minv_mj = info["Minv_mj"][:7,:7]
        M_mj = info["M_mj"][:7,:7]
        nle_mj = info["nle_mj"][:7]

        P_EE = info["P_EE"]
        R_EE = info["R_EE"]
        J_EE = info["J_EE"][:,:7]
        
        dJdq_EE = info["dJdq_EE"][:7]
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
        W = np.diag(1.0/(joint_ub-joint_lb))[:7,:7]
        q_bar = 1/2*(joint_ub+joint_lb)[:7]
        e_joint = W @ (q - q_bar)
        e_joint_dot = W @ dq
        Kp_joint = 80*np.diag([1, 1, 1, 1, 1, 1, 1]).astype(config.np_dtype)
        Kd_joint = 40*np.diag([1, 1, 1, 1, 1, 1, 1]).astype(config.np_dtype)
        q_dtdt = q_dtdt_task + S_null @ (- Kp_joint @ e_joint - Kd_joint @ e_joint_dot)

        # Map to torques
        u_nominal = q_dtdt

        all_input_data = []
        time_cvxpy_and_diff_helper_tmp = 0
        if CBF_config["active"]:
            # Matrices for the CBF-QP constraints
            C = np.zeros([n_controls+n_CBF, n_controls], dtype=config.np_dtype)
            lb = np.zeros(n_controls+n_CBF, dtype=config.np_dtype)
            ub = np.zeros(n_controls+n_CBF, dtype=config.np_dtype)
            CBF_tmp = np.zeros(n_CBF, dtype=config.np_dtype)
            phi1_tmp = np.zeros(n_CBF, dtype=config.np_dtype)
            phi2_tmp = np.zeros(n_CBF, dtype=config.np_dtype)

            all_P_np = np.zeros([n_selected_BBs, 3], dtype=config.np_dtype)
            all_quat_np = np.zeros([n_selected_BBs, 4], dtype=config.np_dtype)
            all_J_np = np.zeros([n_selected_BBs, 6, 7], dtype=config.np_dtype)
            all_dJdq_np = np.zeros([n_selected_BBs, 6], dtype=config.np_dtype)

            for (ii, bb_key) in enumerate(selected_BBs):
                all_P_np[ii, :] = info["P_"+bb_key].astype(config.np_dtype)
                all_J_np[ii, :, :] = info["J_"+bb_key][:,:7].astype(config.np_dtype)
                all_quat_np[ii, :] = get_quat_from_rot_matrix(info["R_"+bb_key]).astype(config.np_dtype)
                all_dJdq_np[ii, :] = info["dJdq_"+bb_key][:7].astype(config.np_dtype)
            
            
            time_cvxpy_and_diff_helper_tmp -= time.time()
            all_h_np, _, _, all_phi1_np, all_actuation_np, all_lb_np, all_ub_np = \
                probs.getCBFConstraints(dq, all_P_np, all_quat_np, all_J_np, all_dJdq_np, alpha0, gamma1, gamma2, compensation)
            time_cvxpy_and_diff_helper_tmp += time.time()

            C[0:n_CBF,:] = all_actuation_np
            lb[0:n_CBF] = all_lb_np
            ub[0:n_CBF] = all_ub_np
            CBF_tmp = all_h_np
            phi1_tmp = all_phi1_np

            # CBF-QP constraints
            print(np.min(CBF_tmp))
            g = -u_nominal
            C[n_CBF:n_CBF+n_controls,:] = M_mj
            lb[n_CBF:] = input_torque_lb[:7] - nle_mj
            ub[n_CBF:] = input_torque_ub[:7] - nle_mj
            cbf_qp.update(g=g, C=C, l=lb, u=ub)
            time_cbf_qp_start = time.time()
            cbf_qp.solve()
            time_cbf_qp_end = time.time()
            u = cbf_qp.results.x
            for kk in range(n_CBF):
                phi2_tmp[kk] = C[kk,:] @ u - lb[kk]

        else:
            u = u_nominal
            time_cbf_qp_start = 0
            time_cbf_qp_end = 0
            CBF_tmp = np.zeros(n_CBF, dtype=config.np_dtype)
            phi1_tmp = np.zeros(n_CBF, dtype=config.np_dtype)
            phi2_tmp = np.zeros(n_CBF, dtype=config.np_dtype)

        # Step the environment
        time_control_loop_end = time.time()
        u = M_mj @ u + nle_mj
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
        time_cvxpy_and_diff_helper[i] = time_cvxpy_and_diff_helper_tmp
        time_cbf_qp[i] = time_cbf_qp_end - time_cbf_qp_start
        time_control_loop[i] = time_control_loop_end - time_control_loop_start

    # Close the environment
    env.close()

    print("==> Save results")
    summary = {"times": times,
               "joint_angles": joint_angles,
               "controls": controls,
               "desired_controls": desired_controls,
               "phi1s": phi1s,
               "phi2s": phi2s,
               "cbf_values": cbf_values,
               "time_cvxpy_and_diff_helper": time_cvxpy_and_diff_helper,
               "time_cbf_qp": time_cbf_qp}
    save_dict(summary, os.path.join(results_dir, 'summary.pkl'))

    print("==> Save all_info")
    save_dict(all_info, os.path.join(results_dir, 'all_info.pkl'))

    # Print solving time
    print("==> Control loop solving time: {:.5f} s".format(np.mean(time_control_loop)))
    print("==> CVXPY and diff opt solving time: {:.5f} s".format(np.mean(time_cvxpy_and_diff_helper)))
    print("==> CBF-QP solving time: {:.5f} s".format(np.mean(time_cbf_qp)))

    # Visualization
    print("==> Draw plots")
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams.update({"text.usetex": True,
                         "text.latex.preamble": r"\usepackage{amsmath}"})
    plt.rcParams.update({'pdf.fonttype': 42})

    for i in range(7):
        fig, ax = plt.subplots(figsize=(10,8), dpi=config.dpi, frameon=True)
        plt.plot(times, joint_angles[:,i], color="tab:blue", linestyle="-", label="u_{:d}".format(i+1))
        plt.axhline(y = joint_lb[i], color = 'black', linestyle = 'dotted', linewidth = 2)
        plt.axhline(y = joint_ub[i], color = 'black', linestyle = 'dotted', linewidth = 2)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'plot_q_{:d}.pdf'.format(i+1)))
        plt.close(fig)


    for i in range(7):
        fig, ax = plt.subplots(figsize=(10,8), dpi=config.dpi, frameon=True)
        plt.plot(times, desired_controls[:,i], color="tab:blue", linestyle=":", 
                label="u_{:d} nominal".format(i+1))
        plt.plot(times, controls[:,i], color="tab:blue", linestyle="-", label="u_{:d}".format(i+1))
        plt.axhline(y = input_torque_lb[i], color = 'black', linestyle = 'dotted', linewidth = 2)
        plt.axhline(y = input_torque_ub[i], color = 'black', linestyle = 'dotted', linewidth = 2)
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
    plt.plot(times, time_cvxpy_and_diff_helper, label="cvxpy and diff helper")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'plot_time_cvxpy_and_diff_helper.pdf'))
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
