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
import scalingFunctionHelper as doh
from cores.utils.rotation_utils import get_quat_from_rot_matrix, get_Q_matrix_from_quat, get_dQ_matrix
from cores.configuration.configuration import Configuration
from scipy.spatial.transform import Rotation
from liegroups import SO3

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_num', default=2, type=int, help='test case number')
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
    controller_config = test_settings["controller_config"]
    CBF_config = test_settings["CBF_config"]
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
    # env = FR3MuJocoEnv(xml_name="fr3_on_table_with_bounding_boxes_circulation_ellipsoid", base_pos=base_pos, base_quat=base_quat,
    #                 cam_distance=cam_distance, cam_azimuth=cam_azimuth, cam_elevation=cam_elevation, cam_lookat=cam_lookat, dt=dt)
    info = env.reset(initial_joint_angles)
    
    time_prev = time.time()
    while True:
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

        if time.time() - time_prev > 0.1:
            # keep only to the second decimal
            print("q:", f"[{q[0]:.2f}, {q[1]:.2f}, {q[2]:.2f}, {q[3]:.2f}, {q[4]:.2f}, {q[5]:.2f}, {q[6]:.2f}, {q[7]:.2f}, {q[8]:.2f}]")
            time_prev = time.time()

        # Primary obejctive: tracking control
        P_d = np.array([0.2, 0.2, 0.86])
        # P_d = np.array([0.1, 0.0, 1.8])
        # P_d = np.array([0.5, 0.0, 0.85])
        # P_d = np.array([0.25, 0.4, 0.85])

        K_p_pos = np.diag([100,100,100]).astype(config.np_dtype)
        K_d_pos = np.diag([50,50,50]).astype(config.np_dtype)
        e_pos = P_EE - P_d # shape (3,)
        e_pos_dt = v_EE[:3] # shape (3,)
        v_dt = - K_p_pos @ e_pos - K_d_pos @ e_pos_dt

        R_d = np.array([[1, 0, 0],
                        [0, -1, 0],
                        [0, 0, -1]], dtype=config.np_dtype)
        # roll = np.pi
        # pitch = -np.pi/4
        # yaw = 0
        # R_d = Rotation.from_euler('xyz', [roll, pitch, yaw]).as_matrix()

        # Create a Rotation object from RPY
        K_p_rot = np.diag([200,200,200]).astype(config.np_dtype)
        K_d_rot = np.diag([100,100,100]).astype(config.np_dtype)
        e_rot = SO3(R_EE @ R_d.T).log() # shape (3,)
        e_rot_dt = v_EE[3:] # shape (3,)
        omega_dt = - K_p_rot @ e_rot - K_d_rot @ e_rot_dt

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
        u = nle + M_mj @ q_dtdt

        # Step the environment
        u = u[:7]
        finger_pos = 0.01
        info = env.step(tau=u, finger_pos=finger_pos)

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
               "smooth_mins": smooth_mins,
               "time_cvxpy": time_cvxpy,
               "time_diff_helper": time_diff_helper,
               "time_cbf_qp": time_cbf_qp}
    save_dict(summary, os.path.join(results_dir, 'summary.pkl'))

    print("==> Save all_info")
    save_dict(all_info, os.path.join(results_dir, 'all_info.pkl'))
    
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

    fig, ax = plt.subplots(figsize=(10,8), dpi=config.dpi, frameon=True)
    plt.plot(times[0:horizon], smooth_mins, label="smooth_mins")
    plt.axhline(y = 0.0, color = 'black', linestyle = 'dotted', linewidth = 2)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'plot_smooth_mins.pdf'))
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

    # Print solving time
    print("==> Control loop solving time: {:.5f} s".format(np.mean(time_control_loop)))
    print("==> CVXPY solving time: {:.5f} s".format(np.mean(time_cvxpy)))
    print("==> Diff helper solving time: {:.5f} s".format(np.mean(time_diff_helper)))
    print("==> CBF-QP solving time: {:.5f} s".format(np.mean(time_cbf_qp)))

    print("==> Done!")
