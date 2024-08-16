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
import scipy.sparse as sparse

from fr3_envs.fr3_mj_env_collision_flying_ball import FR3MuJocoEnv
from fr3_envs.bounding_shape_coef_mj import BoundingShapeCoef
from cores.utils.utils import seed_everything, save_dict
from cores.utils.osqp_utils import init_osqp
import scalingFunctionsHelperPy as sfh
import HOCBFHelperPy as hh
from cores.utils.rotation_utils import get_quat_from_rot_matrix
from cores.configuration.configuration import Configuration
from liegroups import SO3
import multiprocessing
from scipy.spatial.transform import Rotation

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_num', default=2, type=int, help='test case number')
    args = parser.parse_args()

    # Create result directory
    exp_num = args.exp_num
    results_dir = "{}/eg9_results/{:03d}".format(str(Path(__file__).parent.parent), exp_num)
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

    # Joint acceleration limits
    joint_acc_limits_config = test_settings["joint_acceleration_limits_config"]
    joint_acc_lb = np.array(joint_acc_limits_config["lb"], dtype=config.np_dtype)
    joint_acc_ub = np.array(joint_acc_limits_config["ub"], dtype=config.np_dtype)

    # Joint velocity limits
    joint_vel_limits_config = test_settings["joint_velocity_limits_config"]
    joint_vel_lb = np.array(joint_vel_limits_config["lb"], dtype=config.np_dtype)
    joint_vel_ub = np.array(joint_vel_limits_config["ub"], dtype=config.np_dtype)

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
    ball_pos = test_settings["ball_pos"]
    ball_vel = test_settings["ball_vel"]
    dt = 1.0/240.0

    env = FR3MuJocoEnv(xml_name="fr3_on_table_with_bounding_boxes_flying_ball", base_pos=base_pos, base_quat=base_quat,
                    cam_distance=cam_distance, cam_azimuth=cam_azimuth, cam_elevation=cam_elevation, cam_lookat=cam_lookat, dt=dt)
    info = env.reset(initial_joint_angles, ball_pos, ball_vel)
    q_d = np.array(initial_joint_angles[:7], dtype=config.np_dtype)

    # Load the bounding shape coefficients
    BB_coefs = BoundingShapeCoef()

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

    # Define SFs
    # Define a current ball and a future ball
    obstacle_SFs = []
    future_dts = np.array([0, 0.10])
    for i in range(len(future_dts)):
        ball_radius = 0.05
        SF_ball = sfh.Ellipsoid3d(True, np.eye(3)/ball_radius**2, np.zeros(3))
        obstacle_SFs.append(SF_ball)

    robot_SFs = []
    for (i, bb_key) in enumerate(selected_BBs):
        ellipsoid_quadratic_coef = BB_coefs.coefs[bb_key]
        SF_rob = sfh.Ellipsoid3d(True, ellipsoid_quadratic_coef, np.zeros(3))
        robot_SFs.append(SF_rob)

    # Define problems
    n_threads = min(max(multiprocessing.cpu_count() -1, 1), len(selected_BBs))
    probs = hh.Problem3dCollectionMovingObstacle(n_threads)

    for j in range(len(obstacle_SFs)):
        SF_obs = obstacle_SFs[j]
        obs_frame_id = j
        for i in range(len(selected_BBs)):
            SF_rob = robot_SFs[i]
            rob_frame_id = i

            prob = hh.EllipsoidAndEllipsoid3dPrb(SF_rob, SF_obs)
            probs.addProblem(prob, rob_frame_id, obs_frame_id)
    
    # Define proxuite problem
    print("==> Define proxuite problem")
    n_obstacle = len(obstacle_SFs)
    n_robots = len(robot_SFs)
    n_CBF = n_selected_BBs*n_obstacle
    n_in = n_CBF + n_controls + n_controls + n_controls # limits on torque, dq, q
    P_diag = [1]*n_controls + [10000]*(len(future_dts) - 1)
    n_v = n_controls + len(future_dts) - 1
    cbf_qp = init_osqp(n_v=n_v, n_in=n_in, P_diag=P_diag)

    # Create records
    print("==> Create records")
    t_final = 2.0
    horizon = int(t_final/dt)
    times = np.linspace(0, (horizon-1)*dt, horizon)
    joint_angles = np.zeros([horizon, n_controls], dtype=config.np_dtype)
    joint_velocities = np.zeros([horizon, n_controls], dtype=config.np_dtype)
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
    id_geom_offset = 0
    P_EE_prev = info["P_EE"]
    time_prev = time.time()
    print("==> Forward simulate the system")
    try:
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
            W = np.diag(1.0/(joint_ub-joint_lb))[:7,:7]
            e_joint = W @ (q - q_d)
            e_joint_dot = W @ dq
            Kp_joint = 80*np.diag([1, 1, 1, 1, 1, 1, 1]).astype(config.np_dtype)
            Kd_joint = 40*np.diag([1, 1, 1, 1, 1, 1, 1]).astype(config.np_dtype)
            q_dtdt = - Kp_joint @ e_joint - Kd_joint @ e_joint_dot

            # Map to torques
            u_nominal = q_dtdt

            all_input_data = []
            time_cvxpy_and_diff_helper_tmp = 0
            if CBF_config["active"]:
                # Matrices for the CBF-QP constraints
                C = np.zeros([n_in, n_v], dtype=config.np_dtype)
                lb = np.zeros(n_in, dtype=config.np_dtype)
                ub = np.zeros(n_in, dtype=config.np_dtype)
                CBF_tmp = np.zeros(n_CBF, dtype=config.np_dtype)
                phi1_tmp = np.zeros(n_CBF, dtype=config.np_dtype)
                phi2_tmp = np.zeros(n_CBF, dtype=config.np_dtype)

                all_P_rob_np = np.zeros([n_selected_BBs, 3], dtype=config.np_dtype)
                all_quat_rob_np = np.zeros([n_selected_BBs, 4], dtype=config.np_dtype)
                all_J_rob_np = np.zeros([n_selected_BBs, 6, 7], dtype=config.np_dtype)
                all_dJdq_rob_np = np.zeros([n_selected_BBs, 6], dtype=config.np_dtype)
                all_P_obs_np = np.zeros([n_obstacle, 3], dtype=config.np_dtype)
                all_quat_obs_np = np.zeros([n_obstacle, 4], dtype=config.np_dtype)
                all_v_obs_np = np.zeros([n_obstacle, 3], dtype=config.np_dtype)
                all_omega_obs_np = np.zeros([n_obstacle, 3], dtype=config.np_dtype)
                all_v_dot_obs_np = np.zeros([n_obstacle, 3], dtype=config.np_dtype)
                all_omega_dot_obs_np = np.zeros([n_obstacle, 3], dtype=config.np_dtype)

                for (ii, bb_key) in enumerate(selected_BBs):
                    all_P_rob_np[ii, :] = info["P_"+bb_key].astype(config.np_dtype)
                    all_J_rob_np[ii, :, :] = info["J_"+bb_key][:,:7].astype(config.np_dtype)
                    all_quat_rob_np[ii, :] = get_quat_from_rot_matrix(info["R_"+bb_key]).astype(config.np_dtype)
                    all_dJdq_rob_np[ii, :] = info["dJdq_"+bb_key][:7].astype(config.np_dtype)
                
                for (ii, dt_ahead) in enumerate(future_dts):
                    future_pos, future_vel = env.ball_future_pos_and_vel(dt_ahead)
                    all_P_obs_np[ii, :] = future_pos.astype(config.np_dtype)
                    all_quat_obs_np[ii, :] = np.array([0, 0, 0, 1], dtype=config.np_dtype)
                    all_v_obs_np[ii, :] = future_vel.astype(config.np_dtype)
                    all_omega_obs_np[ii, :] = np.zeros(3, dtype=config.np_dtype)
                    all_v_dot_obs_np[ii, :] = np.array([0, 0, -9.81], dtype=config.np_dtype)
                    all_omega_dot_obs_np[ii, :] = np.zeros(3, dtype=config.np_dtype)
                
                time_cvxpy_and_diff_helper_tmp -= time.time()
                all_h_np, all_phi1_np, all_actuation_np, all_lb_np, all_ub_np = \
                    probs.getCBFConstraints(dq, all_P_rob_np, all_quat_rob_np, all_J_rob_np, all_dJdq_rob_np, 
                                            all_P_obs_np, all_quat_obs_np, all_v_obs_np, all_omega_obs_np, 
                                            all_v_dot_obs_np, all_omega_dot_obs_np,
                                            alpha0, gamma1, gamma2, compensation)
                time_cvxpy_and_diff_helper_tmp += time.time()

                # CBF constraints
                C[0:n_CBF,0:n_controls] = all_actuation_np
                for ii in range(len(future_dts)-1):
                    C[(ii+1)*n_robots:(ii+2)*n_robots, n_controls+ii] = -np.ones(n_robots)
                lb[0:n_CBF] = all_lb_np
                ub[0:n_CBF] = all_ub_np
                CBF_tmp = all_h_np
                phi1_tmp = all_phi1_np
                print(np.min(CBF_tmp))

                # torque limits
                C[n_CBF:n_CBF+n_controls,0:n_controls] = M_mj
                lb[n_CBF:n_CBF+n_controls] = input_torque_lb[:7] - nle_mj
                ub[n_CBF:n_CBF+n_controls] = input_torque_ub[:7] - nle_mj

                # dq limits
                h_dq_lb = dq[:7] - joint_vel_lb[:7]
                h_dq_ub = joint_vel_ub[:7] - dq[:7]
                C[n_CBF+n_controls:n_CBF+2*n_controls,0:n_controls] = np.eye(7)
                lb[n_CBF+n_controls:n_CBF+2*n_controls] = - 100*h_dq_lb
                ub[n_CBF+n_controls:n_CBF+2*n_controls] = 100*h_dq_ub

                # q limits
                phi1_q_lb = q[:7] - joint_lb[:7]
                dphi1_q_lb = dq[:7]
                phi2_q_lb = dphi1_q_lb + 20*phi1_q_lb

                phi1_q_ub = joint_ub[:7] - q[:7]
                dphi1_q_ub = -dq[:7]
                phi2_q_ub = dphi1_q_ub + 20*phi1_q_ub

                C[n_CBF+2*n_controls:n_CBF+3*n_controls,0:n_controls] = np.eye(7)
                lb[n_CBF+2*n_controls:n_CBF+3*n_controls] = - 20*dphi1_q_lb - 20*phi2_q_lb
                ub[n_CBF+2*n_controls:n_CBF+3*n_controls] = 20*dphi1_q_ub + 20*phi2_q_ub

                # Put together
                time_cbf_qp_start = time.time()
                data = C.flatten()
                rows, cols = np.indices(C.shape)
                row_indices = rows.flatten()
                col_indices = cols.flatten()
                Ax = sparse.csc_matrix((data, (row_indices, col_indices)), shape=C.shape)
                g = np.zeros(n_v)
                g[0:n_controls] = -u_nominal
                cbf_qp.update(q=g, l=lb, u=ub, Ax=Ax.data)
                results = cbf_qp.solve()
                time_cbf_qp_end = time.time()
                u = results.x[0:n_controls]
                for kk in range(n_CBF):
                    phi2_tmp[kk] = C[kk,0:n_controls] @ u - lb[kk]

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
            # time.sleep(max(0,dt-time_control_loop_end+time_control_loop_start))
            time.sleep(max(0,0.02-time_control_loop_end+time_control_loop_start))

            # Record
            P_EE_prev = P_EE
            joint_angles[i,:] = q
            joint_velocities[i,:] = dq
            controls[i,:] = u
            desired_controls[i,:] = u_nominal
            cbf_values[i,:] = CBF_tmp
            phi1s[i,:] = phi1_tmp
            phi2s[i,:] = phi2_tmp
            time_cvxpy_and_diff_helper[i] = time_cvxpy_and_diff_helper_tmp
            time_cbf_qp[i] = time_cbf_qp_end - time_cbf_qp_start
            time_control_loop[i] = time_control_loop_end - time_control_loop_start
    finally:
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
            plt.plot(times, joint_angles[:,i], color="tab:blue", linestyle="-", label="q_{:d}".format(i+1))
            plt.axhline(y = joint_lb[i], color = 'black', linestyle = 'dotted', linewidth = 2)
            plt.axhline(y = joint_ub[i], color = 'black', linestyle = 'dotted', linewidth = 2)
            plt.legend()
            plt.grid()
            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, 'plot_q_{:d}.pdf'.format(i+1)))
            plt.close(fig)
        
        for i in range(7):
            fig, ax = plt.subplots(figsize=(10,8), dpi=config.dpi, frameon=True)
            plt.plot(times, joint_velocities[:,i], color="tab:blue", linestyle="-", label="dq_{:d}".format(i+1))
            plt.axhline(y = joint_vel_lb[i], color = 'black', linestyle = 'dotted', linewidth = 2)
            plt.axhline(y = joint_vel_ub[i], color = 'black', linestyle = 'dotted', linewidth = 2)
            plt.legend()
            plt.grid()
            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, 'plot_dq_{:d}.pdf'.format(i+1)))
            plt.close(fig)


        for i in range(7):
            fig, ax = plt.subplots(figsize=(10,8), dpi=config.dpi, frameon=True)
            plt.plot(times, desired_controls[:,i], color="tab:blue", linestyle=":", 
                    label="u_{:d} nominal".format(i+1))
            plt.plot(times, controls[:,i], color="tab:blue", linestyle="-", label="\tau_{:d}".format(i+1))
            plt.axhline(y = input_torque_lb[i], color = 'black', linestyle = 'dotted', linewidth = 2)
            plt.axhline(y = input_torque_ub[i], color = 'black', linestyle = 'dotted', linewidth = 2)
            plt.legend()
            plt.grid()
            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, 'plot_controls_{:d}.pdf'.format(i+1)))
            plt.close(fig)

        fig, ax = plt.subplots(figsize=(10,8), dpi=config.dpi, frameon=True)
        plt.plot(times[0:horizon], phi1s, label="phi1")
        plt.plot(times[0:horizon], phi2s, label="phi2")
        plt.axhline(y = 0.0, color = 'black', linestyle = 'dotted', linewidth = 2)
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'plot_phi.pdf'))
        plt.close(fig)

        for i in range(n_CBF):
            fig, ax = plt.subplots(figsize=(10,8), dpi=config.dpi, frameon=True)
            plt.plot(times, phi1s[:,i], label="phi1")
            plt.plot(times, phi2s[:,i], label="phi2")
            plt.axhline(y = 0.0, color = 'black', linestyle = 'dotted', linewidth = 2)
            plt.legend()
            plt.grid()
            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, 'plot_phi_{:d}.pdf'.format(i+1)))
            plt.close(fig)
        
        for i in range(n_CBF):
            fig, ax = plt.subplots(figsize=(10,8), dpi=config.dpi, frameon=True)
            plt.plot(times, cbf_values[:,i], label="CBF")
            plt.axhline(y = 0.0, color = 'black', linestyle = 'dotted', linewidth = 2)
            plt.legend()
            plt.grid()
            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, 'plot_cbf_{:d}.pdf'.format(i+1)))
            plt.close(fig)


        fig, ax = plt.subplots(figsize=(10,8), dpi=config.dpi, frameon=True)
        plt.plot(times, time_cvxpy_and_diff_helper, label="cvxpy and diff helper")
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'plot_time_cvxpy_and_diff_helper.pdf'))
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(10,8), dpi=config.dpi, frameon=True)
        plt.plot(times, time_cbf_qp, label="CBF-QP")
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'plot_time_cbf_qp.pdf'))
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(10,8), dpi=config.dpi, frameon=True)
        plt.plot(times, time_control_loop, label="control loop")
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'plot_time_control_loop.pdf'))
        plt.close(fig)

        print("==> Done!")
