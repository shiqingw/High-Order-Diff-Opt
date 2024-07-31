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
from cores.utils.utils import seed_everything, save_dict
from cores.utils.proxsuite_utils import init_proxsuite_qp
import scalingFunctionsHelperPy as sfh
import HOCBFHelperPy as hh
from cores.utils.rotation_utils import get_quat_from_rot_matrix
from cores.utils.utils import get_facial_equations
from cores.configuration.configuration import Configuration
from liegroups import SO3
from cores.utils.trajectory_utils import PositionTrapezoidalTrajectory, OrientationTrapezoidalTrajectory
from cores.obstacle_collections.polytope_collection import PolytopeCollection
import multiprocessing
from scipy.spatial.transform import Rotation

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_num', default=1, type=int, help='test case number')
    args = parser.parse_args()

    # Create result directory
    exp_num = args.exp_num
    results_dir = "{}/eg3_results/{:03d}".format(str(Path(__file__).parent.parent), exp_num)
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

    # Joint limits
    joint_limits_config = test_settings["joint_limits_config"]
    joint_lb = np.array(joint_limits_config["lb"], dtype=config.np_dtype)
    joint_ub = np.array(joint_limits_config["ub"], dtype=config.np_dtype)

    # Joint input_torque limits
    input_torque_limits_config = test_settings["input_torque_limits_config"]
    input_torque_lb = np.array(input_torque_limits_config["lb"], dtype=config.np_dtype)
    input_torque_ub = np.array(input_torque_limits_config["ub"], dtype=config.np_dtype)

    # Joint acceleratin limits
    joint_acc_limits_config = test_settings["joint_acceleration_limits_config"]
    joint_acc_lb = np.array(joint_acc_limits_config["lb"], dtype=config.np_dtype)
    joint_acc_ub = np.array(joint_acc_limits_config["ub"], dtype=config.np_dtype)

    # End-effector velocity limits
    v_EE_limits_config = test_settings["end_effector_velocity_limits_config"]
    v_EE_lb = np.array(v_EE_limits_config["lb"], dtype=config.np_dtype)
    v_EE_ub = np.array(v_EE_limits_config["ub"], dtype=config.np_dtype)

    # Create and reset simulation
    cam_distance = simulator_config["cam_distance"]
    cam_azimuth = simulator_config["cam_azimuth"]
    cam_elevation = simulator_config["cam_elevation"]
    cam_lookat = simulator_config["cam_lookat"]
    base_pos = simulator_config["base_pos"]
    base_quat = simulator_config["base_quat"]
    initial_joint_angles = test_settings["initial_joint_angles"]
    dt = 1.0/240.0

    env = FR3MuJocoEnv(xml_name="fr3_on_table_with_bounding_boxes_wiping", base_pos=base_pos, base_quat=base_quat,
                    cam_distance=cam_distance, cam_azimuth=cam_azimuth, cam_elevation=cam_elevation, cam_lookat=cam_lookat, dt=dt)
    info = env.reset(initial_joint_angles)

    # Define the robot SF
    robot_SFs = []
    eraser_bb_size_2d = np.array([0.088, 0.035])
    ellipsoid_quadratic_coef_2d = np.diag(1/eraser_bb_size_2d**2)
    eraser_bb_size_3d = np.array([0.088, 0.035, 0.01])
    SF_rob = sfh.Ellipsoid2d(True, ellipsoid_quadratic_coef_2d, np.zeros(2))
    robot_SFs.append(SF_rob)

    id_geom_offset = 0
    eraser_bb_id_offset_2d = id_geom_offset
    eraser_bb_id_offset_3d = id_geom_offset + 1
    P_EE = info["P_EE"]
    R_EE = info["R_EE"]
    theta_2d = np.arctan2(R_EE[1,0], R_EE[0,0])
    R_2d_to_3d = np.array([[np.cos(theta_2d), -np.sin(theta_2d), 0],
                            [np.sin(theta_2d), np.cos(theta_2d), 0],
                            [0, 0, 1]], dtype=config.np_dtype)
    env.add_visual_ellipsoid(eraser_bb_size_3d, P_EE, R_2d_to_3d, np.array([0,1,0,1]), id_geom_offset=eraser_bb_id_offset_2d)
    env.add_visual_ellipsoid(eraser_bb_size_3d, P_EE, R_EE, np.array([1,0,0,1]), id_geom_offset=eraser_bb_id_offset_3d)
    id_geom_offset = env.viewer.user_scn.ngeom

    # Define the boundaries
    base_pos = np.array(base_pos)
    corners_config = test_settings["corners"]
    corner1_2d = np.array(corners_config["corner1"], dtype=config.np_dtype) + base_pos[0:2]
    corner2_2d = np.array(corners_config["corner2"], dtype=config.np_dtype) + base_pos[0:2]
    corner3_2d = np.array(corners_config["corner3"], dtype=config.np_dtype) + base_pos[0:2]
    corner4_2d = np.array(corners_config["corner4"], dtype=config.np_dtype) + base_pos[0:2]
    corners_2d = np.array([corner1_2d, corner2_2d, corner3_2d, corner4_2d])

    corner1_3d = np.array([corner1_2d[0], corner1_2d[1], base_pos[2]])
    corner2_3d = np.array([corner2_2d[0], corner2_2d[1], base_pos[2]])
    corner3_3d = np.array([corner3_2d[0], corner3_2d[1], base_pos[2]])
    corner4_3d = np.array([corner4_2d[0], corner4_2d[1], base_pos[2]])
    corners_3d = np.array([corner1_3d, corner2_3d, corner3_3d, corner4_3d])

    for i in range(len(corners_3d)):
        env.add_visual_ellipsoid(0.01*np.ones(3), corners_3d[i], np.eye(3), np.array([1,0,0,1]), id_geom_offset)
        id_geom_offset = env.viewer.user_scn.ngeom
    for i in range(len(corners_3d)):
        env.add_visual_capsule(corners_3d[i%len(corners_3d)], corners_3d[(i+1)%len(corners_3d)], 0.004, np.array([0,0,1,1]), id_geom_offset)
        id_geom_offset = env.viewer.user_scn.ngeom 

    hyperplane_SFs = []
    A_tmp, b_tmp = get_facial_equations(corners_2d) # need to inverse the sign
    for i in range(A_tmp.shape[0]):
        hyperplane_SFs.append(sfh.Hyperplane2d(False, -A_tmp[i], -b_tmp[i]))
    n_hyperplane = len(hyperplane_SFs)

    # Define the obstacle
    obstacle_config = test_settings["obstacle_config"]
    n_polytope = len(obstacle_config)
    obs_col = PolytopeCollection(2, n_polytope, obstacle_config)

    for (i, obs_key) in enumerate(obs_col.face_equations.keys()):
        # add visual ellipsoids
        all_points = obs_col.face_equations[obs_key]["vertices_in_world"]
        for j in range(len(all_points)):
            point = np.array([all_points[j,0], all_points[j,1], base_pos[2]])
            env.add_visual_ellipsoid(0.01*np.ones(3), point, np.eye(3), np.array([1,0,0,1]),id_geom_offset=id_geom_offset)
            id_geom_offset = env.viewer.user_scn.ngeom
        # add visual capsules
        for j in range(len(all_points)):
            point1 = np.array([all_points[j,0], all_points[j,1], base_pos[2]])
            point2 = np.array([all_points[(j+1)%len(all_points),0], all_points[(j+1)%len(all_points),1], base_pos[2]])
            env.add_visual_capsule(point1, point2, 0.004, np.array([0,0,1,1]), id_geom_offset)
            id_geom_offset = env.viewer.user_scn.ngeom

    polytope_SFs = []
    SF_config = test_settings["SF_config"]
    obstacle_kappa = SF_config["obstacle_kappa"]
    for (i, obs_key) in enumerate(obs_col.face_equations.keys()):
        A_obs_np = obs_col.face_equations[obs_key]["A"]
        b_obs_np = obs_col.face_equations[obs_key]["b"]
        obs_kappa = obstacle_kappa
        SF_obs = sfh.LogSumExp2d(False, A_obs_np, b_obs_np, obs_kappa)
        polytope_SFs.append(SF_obs)

    # Define problems
    n_threads = max(multiprocessing.cpu_count() -1, 1)
    probs = hh.Problem2dCollection(n_threads)

    for i in range(len(robot_SFs)):
        SF_rob = robot_SFs[i]
        frame_id = i
        for (j, obs_key) in enumerate(obs_col.face_equations.keys()):
            SF_obs = polytope_SFs[j]
            vertices = obs_col.face_equations[obs_key]["vertices_in_world"]
            prob = hh.EllipsoidAndLogSumExp2dPrb(SF_rob, SF_obs, vertices)
            probs.addProblem(prob, frame_id)

        for j in range(len(hyperplane_SFs)):
            SF_obs = hyperplane_SFs[j]
            prob = hh.EllipsoidAndHyperplane2dPrb(SF_rob, SF_obs)
            probs.addProblem(prob, frame_id)

    # CBF parameters
    CBF_config = test_settings["CBF_config"]
    alpha0 = CBF_config["alpha0"]
    f0 = CBF_config["f0"]
    gamma1 = CBF_config["gamma1"]
    gamma2 = CBF_config["gamma2"]
    compensation = CBF_config["compensation"] 
    n_controls = 7

    # Define proxuite problem
    print("==> Define proxuite problem")
    n_obstacle = n_polytope + n_hyperplane
    n_robot = len(robot_SFs)
    n_CBF = n_robot*n_obstacle
    n_in = n_controls + n_CBF + 2
    n_eq = 3
    cbf_qp = init_proxsuite_qp(n_v=n_controls, n_eq=n_eq, n_in=n_in)

    # Compute desired trajectory
    t_final = 4
    P_EE_0 = info['P_EE'].copy()
    P_EE_pre_cleaning = np.array([0.50, 0.5, 0.02]) + base_pos
    via_points = np.array([P_EE_0, P_EE_pre_cleaning])
    target_time = np.array([0, t_final])
    Ts = 0.01
    traj_position = PositionTrapezoidalTrajectory(via_points, target_time, T_antp=0.2, Ts=Ts)

    R_EE_start = info['R_EE'].copy()
    roll = np.pi
    pitch = 0
    yaw = 0
    R_EE_pre_grasping = Rotation.from_euler('xyz', [roll, pitch, yaw]).as_matrix()
    orientations = np.array([R_EE_start, R_EE_pre_grasping])
    target_time = np.array([0, t_final])
    traj_orientation = OrientationTrapezoidalTrajectory(orientations, target_time, Ts=Ts)

    # Visualize the trajectory
    N = 10
    len_traj = len(traj_position.t)
    sampled = np.linspace(0, len_traj-1, N).astype(int)
    sampled = traj_position.pd[sampled]
    for i in range(N-1):
        env.add_visual_capsule(sampled[i], sampled[i+1], 0.004, np.array([0,0,1,1]), id_geom_offset)
        id_geom_offset = env.viewer.user_scn.ngeom 
    env.viewer.sync()

    # Control from initial pose to pre-cleaning pose
    T = t_final + 1
    horizon = int(T/dt)
    P_EE_prev = info['P_EE'].copy()
    for i in range(horizon):
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

        # Visualize bounding shape of the eraser
        theta_2d = np.arctan2(R_EE[1,0], R_EE[0,0])
        R_2d_to_3d = np.array([[np.cos(theta_2d), -np.sin(theta_2d), 0],
                                [np.sin(theta_2d), np.cos(theta_2d), 0],
                                [0, 0, 1]], dtype=config.np_dtype)
        env.add_visual_ellipsoid(eraser_bb_size_3d, P_EE, R_2d_to_3d, np.array([0,1,0,1]), id_geom_offset=eraser_bb_id_offset_2d)
        env.add_visual_ellipsoid(eraser_bb_size_3d, P_EE, R_EE, np.array([1,0,0,1]), id_geom_offset=eraser_bb_id_offset_3d)

        # Primary obejctive: tracking control
        K_p_pos = np.diag([100,100,100]).astype(config.np_dtype)
        K_d_pos = np.diag([50,50,50]).astype(config.np_dtype)
        traj_pos, traj_pos_dt, traj_pos_dtdt = traj_position.get_traj_and_ders(i*dt)
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
        ddq_task = S_pinv @ (v_EE_dt_desired - dJdq_EE)

        # Secondary objective: encourage the joints to remain close to the initial configuration
        W = np.diag(1.0/(joint_ub-joint_lb))[:7,:7]
        q_bar = 1/2*(joint_ub+joint_lb)[:7]
        e_joint = W @ (q - q_bar)
        e_joint_dot = W @ dq
        Kp_joint = 80*np.diag([1, 1, 1, 1, 1, 1, 1]).astype(config.np_dtype)
        Kd_joint = 40*np.diag([1, 1, 1, 1, 1, 1, 1]).astype(config.np_dtype)
        ddq = ddq_task + S_null @ (- Kp_joint @ e_joint - Kd_joint @ e_joint_dot)

        # Step the environment
        time_control_loop_end = time.time()
        u = M_mj @ ddq + nle_mj
        finger_pos = 0.02
        info = env.step(tau=u, finger_pos=finger_pos)
        time.sleep(max(0,dt-time_control_loop_end+time_control_loop_start))

        P_EE_prev = P_EE

    # Define cleaning trajectories
    t_1 = 1
    t_2 = 6
    t_final = 11

    # Translational trajectory
    P_EE_start = info['P_EE'].copy()
    P_EE_start_cleaning = np.array([0.50, 0.5, 0.02]) + base_pos
    P_EE_end_cleaning = np.array([-0.1, 0.5, 0.02]) + base_pos
    via_points = np.array([P_EE_start, P_EE_start_cleaning, P_EE_end_cleaning, P_EE_start_cleaning])
    target_time = np.array([0, t_1, t_2, t_final])
    Ts = 0.01
    traj_position = PositionTrapezoidalTrajectory(via_points, target_time, T_antp=0.2, Ts=Ts)

    # Rotational trajectory
    R_EE_start = info['R_EE'].copy()
    roll = np.pi
    pitch = 0
    yaw = 0
    R_EE_cleaning = Rotation.from_euler('xyz', [roll, pitch, yaw]).as_matrix()
    orientations = np.array([R_EE_start, R_EE_cleaning, R_EE_cleaning])
    target_time = np.array([0, t_1, t_final])
    traj_orientation = OrientationTrapezoidalTrajectory(orientations, target_time, Ts=Ts)

    # Visualize the trajectory
    if env.ngeom >= env.maxgeom:
        env.viewer.user_scn.ngeom -= env.maxgeom
        env.ngeom = 0
    else:
        env.viewer.user_scn.ngeom -= env.ngeom
        env.ngeom = 0
    id_geom_offset = env.viewer.user_scn.ngeom 
    N = 100
    len_traj = len(traj_position.t)
    sampled = np.linspace(0, len_traj-1, N).astype(int)
    sampled = traj_position.pd[sampled]
    for i in range(N-1):
        env.add_visual_capsule(sampled[i], sampled[i+1], 0.004, np.array([0,0,1,1]), id_geom_offset)
        id_geom_offset = env.viewer.user_scn.ngeom 
    env.viewer.sync()

    # Control from initial pose to pre-cleaning pose
    T = t_final + 1
    horizon = int(T/dt)
    time_all = np.zeros(horizon, dtype=config.np_dtype)
    P_EE_all = np.zeros([horizon, 3], dtype=config.np_dtype)
    P_EE_desired_all = np.zeros([horizon, 3], dtype=config.np_dtype)
    time_diff_opt = np.zeros(horizon, dtype=config.np_dtype)
    phi1s = np.zeros([horizon, n_CBF], dtype=config.np_dtype)
    phi2s = np.zeros([horizon, n_CBF], dtype=config.np_dtype)
    cbf_values = np.zeros([horizon, n_CBF], dtype=config.np_dtype)

    P_EE_prev = info['P_EE'].copy()
    for i in range(horizon):
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

        tau_ext = (info["qfrc_constraint"] + info["qfrc_smooth"])[:7]

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

        # Visualize bounding shape of the eraser
        theta_2d = np.arctan2(R_EE[1,0], R_EE[0,0])
        R_2d_to_3d = np.array([[np.cos(theta_2d), -np.sin(theta_2d), 0],
                                [np.sin(theta_2d), np.cos(theta_2d), 0],
                                [0, 0, 1]], dtype=config.np_dtype)
        env.add_visual_ellipsoid(eraser_bb_size_3d, P_EE, R_2d_to_3d, np.array([0,1,0,1]), id_geom_offset=eraser_bb_id_offset_2d)
        env.add_visual_ellipsoid(eraser_bb_size_3d, P_EE, R_EE, np.array([1,0,0,1]), id_geom_offset=eraser_bb_id_offset_3d)

        # Primary obejctive: tracking control
        K_p_pos = np.diag([100,100,100]).astype(config.np_dtype)
        K_d_pos = np.diag([50,50,50]).astype(config.np_dtype)
        traj_pos, traj_pos_dt, traj_pos_dtdt = traj_position.get_traj_and_ders(i*dt)
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
        ddq_task = S_pinv @ (v_EE_dt_desired - dJdq_EE)

        # Secondary objective: encourage the joints to remain close to the initial configuration
        W = np.diag(1.0/(joint_ub-joint_lb))[:7,:7]
        q_bar = 1/2*(joint_ub+joint_lb)[:7]
        e_joint = W @ (q - q_bar)
        e_joint_dot = W @ dq
        Kp_joint = 80*np.diag([1, 1, 1, 1, 1, 1, 1]).astype(config.np_dtype)
        Kd_joint = 40*np.diag([1, 1, 1, 1, 1, 1, 1]).astype(config.np_dtype)
        ddq_desired = ddq_task + S_null @ (- Kp_joint @ e_joint - Kd_joint @ e_joint_dot)

        time_diff_helper_tmp = 0
        if (CBF_config["active"]):
            C = np.zeros((n_in, n_controls), dtype=config.np_dtype)
            lb = np.zeros(n_in, dtype=config.np_dtype)
            ub = np.zeros(n_in, dtype=config.np_dtype)
            A = np.zeros((n_eq, n_controls), dtype=config.np_dtype)
            b = np.zeros(n_eq, dtype=config.np_dtype)

            CBF_tmp = np.zeros(n_CBF, dtype=config.np_dtype)
            phi1_tmp = np.zeros(n_CBF, dtype=config.np_dtype)
            phi2_tmp = np.zeros(n_CBF, dtype=config.np_dtype)

            all_P_np = np.zeros([1, 3], dtype=config.np_dtype)
            all_theta_np = np.zeros([1], dtype=config.np_dtype)
            all_J_np = np.zeros([1, 6, 7], dtype=config.np_dtype)
            all_dJdq_np = np.zeros([1, 6], dtype=config.np_dtype)

            all_P_np[0] = P_EE.copy()
            all_theta_np[0] = theta_2d
            all_J_np[0] = J_EE.copy()
            all_dJdq_np[0] = dJdq_EE.copy()

            time_diff_helper_tmp -= time.time()
            all_h_np, all_h_dx, all_h_dxdx, all_phi1_np, all_actuation_np, all_lb_np, all_ub_np = \
                probs.getCBFConstraints(dq, all_P_np, all_theta_np, all_J_np, all_dJdq_np, alpha0, gamma1, gamma2, compensation)
            time_diff_helper_tmp += time.time()

            # CBF-QP constraints
            C[0:n_CBF,:] = all_actuation_np
            lb[0:n_CBF] = all_lb_np
            ub[0:n_CBF] = all_ub_np
            CBF_tmp = all_h_np
            phi1_tmp = all_phi1_np

            C[n_CBF:n_CBF+n_controls,:] = np.eye(n_controls)
            lb[n_CBF:n_CBF+n_controls] = joint_acc_lb[:7]
            ub[n_CBF:n_CBF+n_controls] = joint_acc_ub[:7]

            h_v_lb = v_EE[0:2] - v_EE_lb[0:2]
            h_v_ub = v_EE_ub[0:2] - v_EE[0:2]
            C[n_CBF+n_controls:n_CBF+n_controls+2, :] = J_EE[0:2,:]
            lb[n_CBF+n_controls:n_CBF+n_controls+2] = -20*h_v_lb - dJdq_EE[0:2]
            ub[n_CBF+n_controls:n_CBF+n_controls+2] = 20*h_v_ub - dJdq_EE[0:2]

            A = J_EE[[2,3,4],:]
            b = v_EE_dt_desired[[2,3,4]] - dJdq_EE[[2,3,4]]
            
            g = -ddq_desired

            cbf_qp.update(g=g, C=C, l=lb, u=ub, A=A, b=b)
            cbf_qp.solve()
            ddq = cbf_qp.results.x
            for kk in range(n_CBF):
                phi2_tmp[kk] = C[kk,:] @ ddq - lb[kk]

        else:
            ddq = ddq_desired
            phi1_tmp = np.zeros(n_CBF)
            phi2_tmp = np.zeros(n_CBF)
            CBF_tmp = np.zeros(n_CBF, dtype=config.np_dtype)

        # print(ddq)
        u = M_mj @ ddq + nle_mj

        # Step the environment
        time_control_loop_end = time.time()
        finger_pos = 0.02
        info = env.step(tau=u, finger_pos=finger_pos)
        time.sleep(max(0,dt-time_control_loop_end+time_control_loop_start))

        P_EE_prev = P_EE

        # record data
        time_all[i] = i*dt
        P_EE_all[i,:] = P_EE.copy()
        P_EE_desired_all[i,:] = traj_pos.copy()
        time_diff_opt[i] = time_diff_helper_tmp
        phi1s[i,:] = phi1_tmp
        phi2s[i,:] = phi2_tmp
        cbf_values[i,:] = CBF_tmp

    # Close the environment
    env.close()

    # print("==> Save results")
    # summary = {"time": time_all,
    #            "P_EE": P_EE_all,
    #            "P_EE_desired": P_EE_desired_all}
    # save_dict(summary, os.path.join(results_dir, 'summary.pkl'))

    print("Time diff helper: ", time_diff_opt.mean())

    # Visualization
    print("==> Draw plots")
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams.update({"text.usetex": True,
                         "text.latex.preamble": r"\usepackage{amsmath}"})
    plt.rcParams.update({'pdf.fonttype': 42})

    fig, ax = plt.subplots(figsize=(10,8), dpi=config.dpi, frameon=True)
    ax.plot(time_all, P_EE_all[:,0], label='x')
    ax.plot(time_all, P_EE_desired_all[:,0], label='x_d')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'plot_P_EE_x.pdf'))

    fig, ax = plt.subplots(figsize=(10,8), dpi=config.dpi, frameon=True)
    ax.plot(time_all, P_EE_all[:,1], label='y')
    ax.plot(time_all, P_EE_desired_all[:,1], label='y_d')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'plot_P_EE_y.pdf'))

    fig, ax = plt.subplots(figsize=(10,8), dpi=config.dpi, frameon=True)
    ax.plot(time_all, P_EE_all[:,2], label='z')
    ax.plot(time_all, P_EE_desired_all[:,2], label='z_d')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'plot_P_EE_z.pdf'))

    for i in range(n_CBF):
        fig, ax = plt.subplots(figsize=(10,8), dpi=config.dpi, frameon=True)
        plt.plot(time_all, phi1s[:,i], label="phi1")
        plt.plot(time_all, phi2s[:,i], label="phi2")
        plt.axhline(y = 0.0, color = 'black', linestyle = 'dotted', linewidth = 2)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'plot_phi_{:d}.pdf'.format(i+1)))
        plt.close(fig)
    
    for i in range(n_CBF):
        fig, ax = plt.subplots(figsize=(10,8), dpi=config.dpi, frameon=True)
        plt.plot(time_all, cbf_values[:,i], label="CBF")
        plt.axhline(y = 0.0, color = 'black', linestyle = 'dotted', linewidth = 2)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'plot_cbf_{:d}.pdf'.format(i+1)))
        plt.close(fig)