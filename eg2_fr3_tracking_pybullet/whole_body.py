import json
import sys
import os
import argparse
import shutil
import numpy as np
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import matplotlib.pyplot as plt
import cvxpy as cp
import time
import sympy as sp
import pybullet as p
import pickle
from PIL import Image

from fr3_envs.fr3_env_cam_collision import FR3CameraSimCollision
from fr3_envs.bounding_shape_coef import BoundingShapeCoef
from cores.utils.utils import seed_everything, save_dict, load_dict
from cores.utils.proxsuite_utils import init_proxsuite_qp
import cores_cpp.diffOptCpp as DOC
from cores.utils.rotation_utils import get_quat_from_rot_matrix, get_Q_matrix_from_quat, get_dQ_matrix
from cores.utils.control_utils import get_torque_to_track_traj_const_ori
from cores.configuration.configuration import Configuration

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_num', default=1, type=int, help='test case number')
    args = parser.parse_args()

    # Create result directory
    exp_num = args.exp_num
    results_dir = "{}/eg2_results/{:03d}".format(str(Path(__file__).parent.parent), exp_num)
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
    screenshot_config = test_settings["screenshot_config"]
    controller_config = test_settings["controller_config"]
    CBF_config = test_settings["CBF_config"]
    camera_config = test_settings["camera_config"]
    trajectory_config = test_settings["trajectory_config"]

    # Joint limits
    joint_limits_config = test_settings["joint_limits_config"]
    joint_lb = np.array(joint_limits_config["lb"], dtype=config.np_dtype)
    joint_ub = np.array(joint_limits_config["ub"], dtype=config.np_dtype)

    # Input torque limits
    input_torque_limits_config = test_settings["input_torque_limits_config"]
    input_torque_lb = np.array(input_torque_limits_config["lb"], dtype=config.np_dtype)
    input_torque_ub = np.array(input_torque_limits_config["ub"], dtype=config.np_dtype)

    # Create and reset simulation
    enable_gui_camera_data = simulator_config["enable_gui_camera_data"]
    obs_urdf = simulator_config["obs_urdf"]
    cameraDistance = simulator_config["cameraDistance"]
    cameraYaw = simulator_config["cameraYaw"]
    cameraPitch = simulator_config["cameraPitch"]
    lookat = simulator_config["lookat"]
    crude_type = simulator_config["crude_type"]

    if test_settings["record"] == 1:
        env = FR3CameraSimCollision(camera_config, enable_gui_camera_data, render_mode="human",
                                     record_path=os.path.join(results_dir, 'record.mp4'), crude_type=crude_type)
    else:
        env = FR3CameraSimCollision(camera_config, enable_gui_camera_data, render_mode="human",
                                     record_path=None, crude_type=crude_type)
    
    info = env.reset(cameraDistance = cameraDistance,
                     cameraYaw = cameraYaw,
                     cameraPitch = cameraPitch,
                     lookat = lookat,
                     target_joint_angles = test_settings["initial_joint_angles"])
    
    # Sreenshot config
    save_every = test_settings["save_every"]
    cameraDistance = screenshot_config["cameraDistance"]
    cameraYaw = screenshot_config["cameraYaw"]
    cameraPitch = screenshot_config["cameraPitch"]
    lookat = screenshot_config["lookat"]
    pixelWidth = screenshot_config["pixelWidth"]
    pixelHeight = screenshot_config["pixelHeight"]
    nearPlane = screenshot_config["nearPlane"]
    farPlane = screenshot_config["farPlane"]
    fov = screenshot_config["fov"]
    viewMatrix = p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=lookat,
                                                     distance=cameraDistance, 
                                                     yaw=cameraYaw, 
                                                     pitch=cameraPitch,
                                                     roll = 0,
                                                     upAxisIndex = 2)
    projectionMatrix = p.computeProjectionMatrixFOV(fov, pixelWidth / pixelHeight, nearPlane, farPlane)

    # Load the obstacle
    obstacle_config = test_settings["obstacle_config"]
    obs_pos_np = np.array(obstacle_config["center"], dtype=config.np_dtype)
    obs_quat_np = np.array([0, 0, 0, 1], dtype=config.np_dtype) # (x, y, z, w)
    obs_radius_np = obstacle_config["radius"]
    obs_coef_np = np.eye(3)/obs_radius_np**2
    obs_id = p.loadURDF(obs_urdf, obs_pos_np, obs_quat_np, useFixedBase=True)

    # Load the bounding shape coefficients
    BB_coefs = BoundingShapeCoef()

    # Visualize desired trajectory
    traj_center = np.array(trajectory_config["center"], dtype=config.np_dtype)
    traj_radius = trajectory_config["radius"]
    traj_angular_velocity = trajectory_config["angular_velocity"]
    N = 100
    theta = np.linspace(0, 2*np.pi, N)
    circle = np.zeros((N, 3), dtype=config.np_dtype)
    circle[:,0] = traj_center[0]
    circle[:,1] = traj_center[1] + traj_radius * np.cos(theta)
    circle[:,2] = traj_center[2] + traj_radius * np.sin(theta)
    p.addUserDebugPoints(circle, [[1, 0, 0]]*N, 4, 0)

    # Compute desired trajectory
    horizon = test_settings["horizon_length"]
    dt = 1.0/240.0
    t = np.linspace(0, horizon*dt, horizon+1)
    traj = np.repeat(traj_center.reshape(1,3), horizon+1, axis=0)
    traj[:,1] += traj_radius * np.cos(traj_angular_velocity*t)
    traj[:,2] += traj_radius * np.sin(traj_angular_velocity*t)

    traj_dt = np.zeros([horizon+1, 3], dtype=config.np_dtype)
    traj_dt[:,1] = -traj_radius * traj_angular_velocity * np.sin(traj_angular_velocity*t)
    traj_dt[:,2] = traj_radius * traj_angular_velocity * np.cos(traj_angular_velocity*t)

    traj_dtdt = np.zeros([horizon+1, 3], dtype=config.np_dtype)
    traj_dtdt[:,1] = -traj_radius * traj_angular_velocity**2 * np.cos(traj_angular_velocity*t)
    traj_dtdt[:,2] = -traj_radius * traj_angular_velocity**2 * np.sin(traj_angular_velocity*t)

    # CBF parameters
    CBF_config = test_settings["CBF_config"]
    alpha0 = CBF_config["alpha0"]
    gamma1 = CBF_config["gamma1"]
    gamma2 = CBF_config["gamma2"]
    compensation = CBF_config["compensation"] 
    selected_BBs = CBF_config["selected_bbs"]
    n_controls = 9

    # Define proxuite problem
    print("==> Define proxuite problem")
    n_CBF = len(selected_BBs)
    cbf_qp = init_proxsuite_qp(n_v=n_controls, n_eq=0, n_in=n_controls+n_CBF)

    # Create records
    print("==> Create records")
    times = np.linspace(0, (horizon-1)*dt, horizon)
    joint_angles = np.zeros([horizon, n_controls], dtype=config.np_dtype)
    controls = np.zeros([horizon, n_controls], dtype=config.np_dtype)
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

        P_EE = info["P_EE"]
        R_EE = info["R_EE"]
        J_EE = info["J_EE"]
        dJ_EE = info["dJ_EE"]
        v_EE = J_EE @ dq

        # Primary obejctive: tracking control
        Kp = np.diag([20,20,20,100,100,100])
        Kd = np.diag([20,20,20,100,100,100])
        R_d = np.diag([1,1,-1])
        G, u_task = get_torque_to_track_traj_const_ori(traj[i,:], traj_dt[i,:], traj_dtdt[i,:], R_d, Kp, Kd, Minv, J_EE, dJ_EE, dq, P_EE, R_EE)

        # Secondary objective: encourage the joints to remain close to the initial configuration
        W = np.diag(1.0/(joint_ub-joint_lb))
        q_bar = np.array(test_settings["initial_joint_angles"], dtype=config.np_dtype)
        eq = W @ (q - q_bar)
        deq = W @ dq
        Kp = np.diag([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 50., 50.])
        Kd = np.diag([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 10., 10.])
        u_joint = M @ (- Kd @ deq - Kp @ eq) # larger control only for the fingers

        # Compute the input torque
        Gpinv = G.T @ np.linalg.pinv(G @ G.T + 0.2* np.eye(G.shape[0]))
        u_nominal = nle + Gpinv @ u_task + (np.eye(len(q)) - Gpinv @ G) @ u_joint 

        time_diff_helper_tmp = 0
        if CBF_config["active"]:
            # Matrices for the CBF-QP constraints
            C = np.zeros([n_controls+n_CBF, n_controls], dtype=config.np_dtype)
            lb = np.zeros(n_controls+n_CBF, dtype=config.np_dtype)
            ub = np.zeros(n_controls+n_CBF, dtype=config.np_dtype)
            CBF_tmp = np.zeros(n_CBF, dtype=config.np_dtype)
            phi1_tmp = np.zeros(n_CBF, dtype=config.np_dtype)
            phi2_tmp = np.zeros(n_CBF, dtype=config.np_dtype)

            for kk in range(len(selected_BBs)):
                name_BB = selected_BBs[kk]
                P_BB = info["P_"+name_BB]
                R_BB = info["R_"+name_BB]
                J_BB = info["J_"+name_BB]
                dJ_BB = info["dJ_"+name_BB]
                v_BB = J_BB @ dq
                D_BB = BB_coefs.coefs[name_BB]
                quat_BB = get_quat_from_rot_matrix(R_BB)
                time_diff_helper_tmp -= time.time()
                alpha, _, alpha_dx_tmp, alpha_dxdx_tmp = DOC.getGradientAndHessianEllipsoids(P_BB, quat_BB, D_BB, 
                                R_BB, obs_coef_np, obs_pos_np)
                time_diff_helper_tmp += time.time()
                
                # Order of parameters in alpha_dx_tmp and alpha_dxdx_tmp: [qx, qy, qz, qw, x, y, z]
                # Convert to the order of [x, y, z, qx, qy, qz, qw]
                alpha_dx = np.zeros(7, dtype=config.np_dtype)
                alpha_dx[0:3] = alpha_dx_tmp[4:7]
                alpha_dx[3:7] = alpha_dx_tmp[0:4]

                alpha_dxdx = np.zeros((7, 7), dtype=config.np_dtype)
                alpha_dxdx[0:3,0:3] = alpha_dxdx_tmp[4:7,4:7]
                alpha_dxdx[3:7,3:7] = alpha_dxdx_tmp[0:4,0:4]
                alpha_dxdx[0:3,3:7] = alpha_dxdx_tmp[4:7,0:4]
                alpha_dxdx[3:7,0:3] = alpha_dxdx_tmp[0:4,4:7]

                # CBF-QP constraints
                dx = np.zeros(7, dtype=config.np_dtype)
                dx[0:3] = v_BB[0:3]
                Q = get_Q_matrix_from_quat(quat_BB) # shape (4,3)
                dquat = 0.5 * Q @ v_BB[3:6] # shape (4,)
                dx[3:7] = dquat 
                dCBF =  alpha_dx @ dx # scalar
                CBF = alpha - alpha0
                phi1 = dCBF + gamma1 * CBF

                dQ = get_dQ_matrix(dquat) # shape (4,3)
                tmp_vec = np.zeros(7, dtype=config.np_dtype)
                tmp_vec[3:7] = 0.5 * dQ @ v_BB[3:6] # shape (4,)
                tmp_mat = np.zeros((7,6), dtype=config.np_dtype)
                tmp_mat[0:3,0:3] = np.eye(3, dtype=config.np_dtype)
                tmp_mat[3:7,3:6] = 0.5 * Q

                C[kk,:] = alpha_dx @ tmp_mat @ J_BB @ Minv
                lb[kk] = - gamma2*phi1 - gamma1*dCBF - dx.T @ alpha_dxdx @ dx - alpha_dx @ tmp_vec \
                        - alpha_dx @ tmp_mat @ dJ_BB @ dq + alpha_dx @ tmp_mat @ J_BB @ Minv @ nle + compensation
                ub[kk] = np.inf

                CBF_tmp[kk] = CBF
                phi1_tmp[kk] = phi1

            # CBF-QP constraints
            g = -u_nominal
            C[n_CBF:n_CBF+n_controls,:] = np.eye(n_controls, dtype=config.np_dtype)
            lb[n_CBF:] = input_torque_lb
            ub[n_CBF:] = input_torque_ub
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
        info = env.step(action=u, return_image=False)
        time.sleep(max(0,dt-time_control_loop_end+time_control_loop_start))

        # Record
        joint_angles[i,:] = q
        controls[i,:] = u
        desired_controls[i,:] = u_nominal
        cbf_values[i,:] = CBF_tmp
        phi1s[i,:] = phi1_tmp
        phi2s[i,:] = phi2_tmp
        time_diff_helper[i] = time_diff_helper_tmp
        time_cbf_qp[i] = time_cbf_qp_end - time_cbf_qp_start
        time_control_loop[i] = time_control_loop_end - time_control_loop_start

        if test_settings["save_screeshot"] == 1 and i % save_every == 0:
            screenshot = p.getCameraImage(pixelWidth,
                                            pixelHeight,
                                            viewMatrix=viewMatrix,
                                            projectionMatrix=projectionMatrix,
                                            flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
                                            renderer=p.ER_BULLET_HARDWARE_OPENGL
                                            )
            screenshot = np.reshape(screenshot[2], (pixelHeight, pixelWidth, 4))
            screenshot = screenshot.astype(np.uint8)
            screenshot = Image.fromarray(screenshot)
            screenshot = screenshot.convert('RGB')
            screenshot.save(results_dir+'/screenshot_'+'{:04d}.{}'.format(i, test_settings["image_save_format"]))


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

    for i in range(n_controls):
        fig, ax = plt.subplots(figsize=(10,8), dpi=config.dpi, frameon=True)
        plt.plot(times, desired_controls[:,i], color="tab:blue", linestyle=":", 
                label="u_{:d} nominal".format(i+1))
        plt.plot(times, controls[:,i], color="tab:blue", linestyle="-", label="u_{:d}".format(i+1))
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'plot_controls_{:d}.pdf'.format(i+1)))
        plt.close(fig)

    for i in range(n_CBF):
        fig, ax = plt.subplots(figsize=(10,8), dpi=config.dpi, frameon=True)
        plt.plot(times, cbf_values[:,i], label="CBF")
        plt.plot(times, phi1s[:,i], label="phi1")
        plt.plot(times, phi2s[:,i], label="phi2")
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

    # Print solving time
    print("==> Control loop solving time: {:.5f} s".format(np.mean(time_control_loop)))
    print("==> CVXPY solving time: {:.5f} s".format(np.mean(time_cvxpy)))
    print("==> Diff helper solving time: {:.5f} s".format(np.mean(time_diff_helper)))
    print("==> CBF-QP solving time: {:.5f} s".format(np.mean(time_cbf_qp)))


    with open(os.path.join("/Users/shiqing/Desktop/Generalized-Diff-Opt", 'all_info.pkl'), 'wb') as f:
        pickle.dump(all_info, f)
    print("==> all_info saved")
    print("==> Done!")
