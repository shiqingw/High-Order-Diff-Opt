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
from liegroups import SO3
from cores.utils.trajectory_utils import PositionTrapezoidalTrajectory, OrientationTrapezoidalTrajectory
from cores.obstacle_collections.polytope_collection import PolytopeCollection
import scs
from scipy import sparse
import posix_ipc
import mmap
import multiprocessing
from concurrent.futures.process import ProcessPoolExecutor
import concurrent

def create_mapfile(shared_memory_name, size):
    shared_memory = posix_ipc.SharedMemory(shared_memory_name, posix_ipc.O_CREX, size=size)
    
    # Map the shared memory to a file descriptor
    mapfile = mmap.mmap(shared_memory.fd, size)
    shared_memory.close_fd()  # Close the file descriptor from posix_ipc

    return mapfile

def clean_shared_memory(shared_memory_name):
    try:
        posix_ipc.unlink_shared_memory(shared_memory_name)
    except posix_ipc.ExistentialError:
        pass

class SolverNode:
    def __init__(self, ellipsoid_quadratic_coef, A_obs_np, b_obs_np, obs_kappa, vertices, np_dtype, 
                 frame_id, n_frames, dq_shm_id, all_J_shm_id, all_quat_shm_id, all_P_shm_id, 
                 all_dJdq_shm_id, alpha0, gamma1, gamma2, compensation, cbf_id, all_h_shm_id,
                 all_h_dx_shm_id, all_h_dxdx_shm_id, all_phi1_shm_id, all_actuation_shm_id, 
                 all_lb_shm_id, all_ub_shm_id):
                 
        self.frame_id = frame_id
        self.n_frames = n_frames 
        self.ellipsoid_quadratic_coef = ellipsoid_quadratic_coef
        self.vertices = vertices 
        self.np_dtype = np_dtype
        self.SF1 = doh.Ellipsoid3d(True, ellipsoid_quadratic_coef, np.zeros(3))
        self.SF2 = doh.LogSumExp3d(False, A_obs_np, b_obs_np, obs_kappa)
        self.dq_shm_id = dq_shm_id
        self.all_J_shm_id = all_J_shm_id
        self.all_quat_shm_id = all_quat_shm_id
        self.all_R_shm_id = all_R_shm_id
        self.all_P_shm_id = all_P_shm_id
        self.all_dJdq_shm_id = all_dJdq_shm_id
        self.alpha0 = alpha0
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.compensation = compensation
        self.cbf_id = cbf_id
        self.all_h_shm_id = all_h_shm_id
        self.all_h_dx_shm_id = all_h_dx_shm_id
        self.all_h_dxdx_shm_id = all_h_dxdx_shm_id
        self.all_phi1_shm_id = all_phi1_shm_id
        self.all_actuation_shm_id = all_actuation_shm_id
        self.all_lb_shm_id = all_lb_shm_id
        self.all_ub_shm_id = all_ub_shm_id

        # Preparation for SCS
        self.n_exp_cones = A_obs_np.shape[0]
        self.n_vars = A_obs_np.shape[1]
        A_d = obs_kappa * A_obs_np
        b_d = obs_kappa * b_obs_np
        c_d = np.log(self.n_exp_cones)
        A_exp = sparse.lil_matrix((3 * self.n_exp_cones, self.n_exp_cones + self.n_vars))
        b_exp = np.zeros(3 * self.n_exp_cones)
        for i in range(self.n_exp_cones):
            A_exp[i*3, 0] = - A_d[i, 0]
            A_exp[i*3, 1] = - A_d[i, 1]
            A_exp[i*3, 2] = - A_d[i, 2]
            A_exp[i*3 + 2, i + self.n_vars] = -1
            b_exp[i*3] = b_d[i] - c_d
            b_exp[i*3 + 1] = 1

        self.A_scs = sparse.vstack([
            sparse.hstack([sparse.csc_matrix((1, self.n_vars)), np.ones((1, self.n_exp_cones))]), # positive cone
            A_exp, # exponential cones
            ], format="csc", dtype=np_dtype)
        self.b_scs = np.hstack([1, b_exp], dtype=np_dtype)

        Q_d = self.ellipsoid_quadratic_coef
        c_d = - Q_d @ np.zeros(3, dtype=self.np_dtype)
        P_scs = sparse.block_diag([Q_d, sparse.csc_matrix((self.n_exp_cones,self.n_exp_cones))], format="csc")
        c_scs = np.hstack([c_d, np.zeros(self.n_exp_cones)])
        data = dict(P=P_scs, A=self.A_scs, b=self.b_scs, c=c_scs)
        cone = dict(l=1, ep=self.n_exp_cones)
        self.solver = scs.SCS(data,
                              cone,
                              eps_abs=1e-5,
                              eps_rel=1e-5,
                              verbose=False
                              )

        self.scs_result_x = None
        self.scs_result_y = None
        self.scs_result_s = None

    def solve(self):
        # Get shared memory
        itemsize = np.dtype(self.np_dtype).itemsize
        dq_shm = posix_ipc.SharedMemory(self.dq_shm_id)
        dq_mapfile = mmap.mmap(dq_shm.fd, 7*itemsize)
        dq_shm.close_fd()
        # dq = np.memmap(dq_mapfile, dtype=self.np_dtype, mode='r', shape=(7,))
        dq = np.ndarray((7,), dtype=self.np_dtype, buffer=dq_mapfile)

        all_J_shm = posix_ipc.SharedMemory(self.all_J_shm_id)
        all_J_mapfile = mmap.mmap(all_J_shm.fd, self.n_frames*6*7*itemsize)
        all_J_shm.close_fd()
        # all_J = np.memmap(all_J_mapfile, dtype=self.np_dtype, mode='r', shape=(self.n_frames,6,7))
        all_J = np.ndarray((self.n_frames,6,7), dtype=self.np_dtype, buffer=all_J_mapfile)

        all_quat_shm = posix_ipc.SharedMemory(self.all_quat_shm_id)
        all_quat_mapfile = mmap.mmap(all_quat_shm.fd, self.n_frames*4*itemsize)
        all_quat_shm.close_fd()
        # all_quat = np.memmap(all_quat_mapfile, dtype=self.np_dtype, mode='r', shape=(self.n_frames,4))
        all_quat = np.ndarray((self.n_frames,4), dtype=self.np_dtype, buffer=all_quat_mapfile)

        all_R_shm = posix_ipc.SharedMemory(self.all_R_shm_id)
        all_R_mapfile = mmap.mmap(all_R_shm.fd, self.n_frames*3*3*itemsize)
        all_R_shm.close_fd()
        # all_R = np.memmap(all_R_mapfile, dtype=self.np_dtype, mode='r', shape=(self.n_frames,3,3))
        all_R = np.ndarray((self.n_frames,3,3), dtype=self.np_dtype, buffer=all_R_mapfile)

        all_P_shm = posix_ipc.SharedMemory(self.all_P_shm_id)
        all_P_mapfile = mmap.mmap(all_P_shm.fd, self.n_frames*3*itemsize)
        all_P_shm.close_fd()
        # all_P = np.memmap(all_P_mapfile, dtype=self.np_dtype, mode='r', shape=(self.n_frames,3))
        all_P = np.ndarray((self.n_frames,3), dtype=self.np_dtype, buffer=all_P_mapfile)

        all_dJdq_shm = posix_ipc.SharedMemory(self.all_dJdq_shm_id)
        all_dJdq_mapfile = mmap.mmap(all_dJdq_shm.fd, self.n_frames*6*itemsize)
        all_dJdq_shm.close_fd()
        # all_dJdq = np.memmap(all_dJdq_mapfile, dtype=self.np_dtype, mode='r', shape=(self.n_frames,6))
        all_dJdq = np.ndarray((self.n_frames,6), dtype=self.np_dtype, buffer=all_dJdq_mapfile)

        all_h_shm = posix_ipc.SharedMemory(self.all_h_shm_id)
        all_h_mapfile = mmap.mmap(all_h_shm.fd, self.n_frames*itemsize)
        all_h_shm.close_fd()
        # all_h = np.memmap(all_h_mapfile, dtype=self.np_dtype, mode='r', shape=(self.n_frames,))
        all_h = np.ndarray((self.n_frames,), dtype=self.np_dtype, buffer=all_h_mapfile)

        all_h_dx_shm = posix_ipc.SharedMemory(self.all_h_dx_shm_id)
        all_h_dx_mapfile = mmap.mmap(all_h_dx_shm.fd, self.n_frames*7*itemsize)
        all_h_dx_shm.close_fd()
        # all_h_dx = np.memmap(all_h_dx_mapfile, dtype=self.np_dtype, mode='r', shape=(self.n_frames,7))
        all_h_dx = np.ndarray((self.n_frames,7), dtype=self.np_dtype, buffer=all_h_dx_mapfile)

        all_h_dxdx_shm = posix_ipc.SharedMemory(self.all_h_dxdx_shm_id)
        all_h_dxdx_mapfile = mmap.mmap(all_h_dxdx_shm.fd, self.n_frames*7*7*itemsize)
        all_h_dxdx_shm.close_fd()
        # all_h_dxdx = np.memmap(all_h_dxdx_mapfile, dtype=self.np_dtype, mode='r', shape=(self.n_frames,7,7))
        all_h_dxdx = np.ndarray((self.n_frames,7,7), dtype=self.np_dtype, buffer=all_h_dxdx_mapfile)

        all_phi1_shm = posix_ipc.SharedMemory(self.all_phi1_shm_id)
        all_phi1_mapfile = mmap.mmap(all_phi1_shm.fd, self.n_frames*itemsize)
        all_phi1_shm.close_fd()
        # all_phi1 = np.memmap(all_phi1_mapfile, dtype=self.np_dtype, mode='r', shape=(self.n_frames,))
        all_phi1 = np.ndarray((self.n_frames,), dtype=self.np_dtype, buffer=all_phi1_mapfile)

        all_actuation_shm = posix_ipc.SharedMemory(self.all_actuation_shm_id)
        all_actuation_mapfile = mmap.mmap(all_actuation_shm.fd, self.n_frames*7*itemsize)
        all_actuation_shm.close_fd()
        # all_actuation = np.memmap(all_actuation_mapfile, dtype=self.np_dtype, mode='r', shape=(self.n_frames,7))
        all_actuation = np.ndarray((self.n_frames,7), dtype=self.np_dtype, buffer=all_actuation_mapfile)

        all_lb_shm = posix_ipc.SharedMemory(self.all_lb_shm_id)
        all_lb_mapfile = mmap.mmap(all_lb_shm.fd, self.n_frames*itemsize)
        all_lb_shm.close_fd()
        # all_lb = np.memmap(all_lb_mapfile, dtype=self.np_dtype, mode='r', shape=(self.n_frames,))
        all_lb = np.ndarray((self.n_frames,), dtype=self.np_dtype, buffer=all_lb_mapfile)

        all_ub_shm = posix_ipc.SharedMemory(self.all_ub_shm_id)
        all_ub_mapfile = mmap.mmap(all_ub_shm.fd, self.n_frames*itemsize)
        all_ub_shm.close_fd()
        # all_ub = np.memmap(all_ub_mapfile, dtype=self.np_dtype, mode='r', shape=(self.n_frames,))
        all_ub = np.ndarray((self.n_frames,), dtype=self.np_dtype, buffer=all_ub_mapfile)

        P_ = all_P[self.frame_id, :]
        min_distance = np.linalg.norm(self.vertices - P_, axis=1).min()
        print("min_distance: ", min_distance)
        if min_distance < 10:
            dq_ = dq
            R_ = all_R[self.frame_id, :, :]
            J_ = all_J[self.frame_id, :, :]
            dJdq_ = all_dJdq[self.frame_id, :]
            quat_ = all_quat[self.frame_id, :]
            v_ = J_ @ dq_ 
            A_ = np.zeros((7,6), dtype=self.np_dtype)
            Q_ = get_Q_matrix_from_quat(quat_) # shape (4,3)
            A_[0:3,0:3] = np.eye(3, dtype=self.np_dtype)
            A_[3:7,3:6] = Q_
            dx_ = A_ @ v_
            dquat_ = 0.5 * Q_ @ v_[3:6] # shape (4,)
            dQ_ = get_dQ_matrix(dquat_) # shape (4,3)
            dA_ = np.zeros((7,6), dtype=self.np_dtype)
            dA_[3:7,3:6] = dQ_

            c_d = - self.ellipsoid_quadratic_coef @ P_
            c_scs = np.hstack([c_d, np.zeros(self.n_exp_cones)])
            self.solver.update(c=c_scs)

            # Solve the SCS problem
            if (self.scs_result_x is not None) and (self.scs_result_y is not None) and (self.scs_result_s is not None):
                sol = self.solver.solve(warm_start=True,
                                   x = self.scs_result_x,
                                   y = self.scs_result_y,
                                   s = self.scs_result_s
                                   )
            else:
                sol = self.solver.solve()
            # Save for the warm start
            self.scs_result_x = sol["x"]
            self.scs_result_y = sol["y"]
            self.scs_result_s = sol["s"]

            # CBF constraint
            p_sol_np = sol["x"][:self.n_vars]
            alpha_, alpha_dx_, alpha_dxdx_ = doh.getGradientAndHessian3d(p_sol_np, self.SF1, P_, quat_, 
                                                                self.SF2, np.zeros(3), np.array([0,0,0,1]))
            
            alpha0_ = self.alpha0
            gamma1_ = self.gamma1
            gamma2_ = self.gamma2
            compensation_ = self.compensation

            h_ = alpha_ - alpha0_
            h_dx_ = alpha_dx_
            h_dxdx_ = alpha_dxdx_

            dh_ = h_dx_ @ dx_
            phi1_ = dh_ + gamma1_ * h_
            
            actuation_ = h_dx_ @ A_ @ J_
            lb_ = - gamma2_*phi1_ - gamma1_*dh_ - dx_.T @ h_dxdx_ @ dx_ - h_dx_ @ dA_ @ v_ \
                - h_dx_ @ A_ @ dJdq_ + compensation_
            ub_ = np.inf
            result_ = (h_, h_dx_, h_dxdx_, phi1_, actuation_, lb_, ub_)
        else:
            result_ = (0, np.zeros(7), np.zeros((7,7)), 0, np.zeros(9), 0, 0)
        
        # Update shared memory
        all_h[self.cbf_id] = result_[0]
        all_h_dx[self.cbf_id, :] = result_[1]
        all_h_dxdx[self.cbf_id, :, :] = result_[2]
        all_phi1[self.cbf_id] = result_[3]
        all_actuation[self.cbf_id, :] = result_[4]
        all_lb[self.cbf_id] = result_[5]
        all_ub[self.cbf_id] = result_[6]
        return

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

    env = FR3MuJocoEnv(xml_name="fr3_on_table_with_bounding_boxes_circulation_polytope2", base_pos=base_pos, base_quat=base_quat,
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
            env.add_visual_ellipsoid(0.05*np.ones(3), all_points[j], np.eye(3), np.array([1,0,0,1]),id_geom_offset=id_geom_offset)
            id_geom_offset = env.viewer.user_scn.ngeom

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
    n_selected_BBs = len(selected_BBs)
    n_controls = 7

    # Define proxuite problem
    print("==> Define proxuite problem")
    n_CBF = n_selected_BBs*n_obstacle
    cbf_qp = init_proxsuite_qp(n_v=n_controls, n_eq=0, n_in=n_controls+n_CBF)

    # Create shared memory
    print("==> Create shared memory")
    dq_shm_id = "/dq_shm"
    all_J_shm_id = "/all_J_shm"
    all_quat_shm_id = "/all_quat_shm"
    all_R_shm_id = "/all_R_shm"
    all_P_shm_id = "/all_P_shm"
    all_dJdq_shm_id = "/all_dJdq_shm"
    all_h_shm_id = "/all_h_shm"
    all_h_dx_shm_id = "/all_h_dx_shm"
    all_h_dxdx_shm_id = "/all_h_dxdx_shm"
    all_phi1_shm_id = "/all_phi1_shm"
    all_actuation_shm_id = "/all_actuation_shm"
    all_lb_shm_id = "/all_lb_shm"
    all_ub_shm_id = "/all_ub_shm"

    # Clean them up if they exist
    clean_shared_memory(dq_shm_id)
    clean_shared_memory(all_J_shm_id)
    clean_shared_memory(all_quat_shm_id)
    clean_shared_memory(all_R_shm_id)
    clean_shared_memory(all_P_shm_id)
    clean_shared_memory(all_dJdq_shm_id)
    clean_shared_memory(all_h_shm_id)
    clean_shared_memory(all_h_dx_shm_id)
    clean_shared_memory(all_h_dxdx_shm_id)
    clean_shared_memory(all_phi1_shm_id)
    clean_shared_memory(all_actuation_shm_id)
    clean_shared_memory(all_lb_shm_id)
    clean_shared_memory(all_ub_shm_id)

    itemsize = np.dtype(config.np_dtype).itemsize
    dq_shm_mapfile = create_mapfile(dq_shm_id, 7*itemsize)
    all_J_shm_mapfile = create_mapfile(all_J_shm_id, n_selected_BBs*6*7*itemsize)
    all_quat_shm_mapfile = create_mapfile(all_quat_shm_id, n_selected_BBs*4*itemsize)
    all_R_shm_mapfile = create_mapfile(all_R_shm_id, n_selected_BBs*3*3*itemsize)
    all_P_shm_mapfile = create_mapfile(all_P_shm_id, n_selected_BBs*3*itemsize)
    all_dJdq_shm_mapfile = create_mapfile(all_dJdq_shm_id, n_selected_BBs*6*itemsize)
    all_h_shm_mapfile = create_mapfile(all_h_shm_id, n_CBF*itemsize)
    all_h_dx_shm_mapfile = create_mapfile(all_h_dx_shm_id, n_CBF*7*itemsize)
    all_h_dxdx_shm_mapfile = create_mapfile(all_h_dxdx_shm_id, n_CBF*7*7*itemsize)
    all_phi1_shm_mapfile = create_mapfile(all_phi1_shm_id, n_CBF*itemsize)
    all_actuation_shm_mapfile = create_mapfile(all_actuation_shm_id, n_CBF*7*itemsize)
    all_lb_shm_mapfile = create_mapfile(all_lb_shm_id, n_CBF*itemsize)
    all_ub_shm_mapfile = create_mapfile(all_ub_shm_id, n_CBF*itemsize)

    # Create workers
    print("==> Create workers")
    all_solvers = []
    for (i, bb_key) in enumerate(selected_BBs):
        for (j, obs_key) in enumerate(obs_col.face_equations.keys()):
            frame_id = i
            ellipsoid_quadratic_coef = BB_coefs.coefs[bb_key]
            A_obs_np = obs_col.face_equations[obs_key]["A"]
            b_obs_np = obs_col.face_equations[obs_key]["b"]
            obs_kappa = obstacle_kappa
            vertices = obs_col.face_equations[obs_key]["vertices_in_world"]
            np_dtype = config.np_dtype
            cbf_id = i*n_obstacle + j
            solver = SolverNode(ellipsoid_quadratic_coef, A_obs_np, b_obs_np, obs_kappa, vertices, np_dtype,
                                frame_id, n_selected_BBs, dq_shm_id, all_J_shm_id, all_quat_shm_id, all_P_shm_id, all_dJdq_shm_id,
                                alpha0, gamma1, gamma2, compensation, cbf_id, all_h_shm_id,
                                all_h_dx_shm_id, all_h_dxdx_shm_id, all_phi1_shm_id, all_actuation_shm_id, 
                                all_lb_shm_id, all_ub_shm_id)
            all_solvers.append(solver)

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

    # Start a pool of workers
    # NUM_WORKERS = multiprocessing.cpu_count() - 1
    NUM_WORKERS = 1
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        # Forward simulate the system
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
                    all_J_np = np.zeros([n_selected_BBs, 6, 7], dtype=config.np_dtype)
                    all_quat_np = np.zeros([n_selected_BBs, 4], dtype=config.np_dtype)
                    all_R_np = np.zeros([n_selected_BBs, 3, 3], dtype=config.np_dtype)
                    all_dJdq_np = np.zeros([n_selected_BBs, 6], dtype=config.np_dtype)

                    for (ii, bb_key) in enumerate(selected_BBs):
                        all_P_np[ii, :] = info["P_"+bb_key].astype(config.np_dtype)
                        all_J_np[ii, :, :] = info["J_"+bb_key][:,:7].astype(config.np_dtype)
                        all_quat_np[ii, :] = get_quat_from_rot_matrix(info["R_"+bb_key]).astype(config.np_dtype)
                        all_R_np[ii, :, :] = info["R_"+bb_key].astype(config.np_dtype)
                        all_dJdq_np[ii, :] = info["dJdq_"+bb_key][:7].astype(config.np_dtype)
                    
                    # Update shared memory
                    # dq_memmap = np.memmap(dq_shm_mapfile, dtype=config.np_dtype, mode='r+', shape=(7,))
                    # all_P_memmap = np.memmap(all_P_shm_mapfile, dtype=config.np_dtype, mode='r+', shape=(n_selected_BBs,3))
                    # all_J_memmap = np.memmap(all_J_shm_mapfile, dtype=config.np_dtype, mode='r+', shape=(n_selected_BBs,6,7))
                    # all_quat_memmap = np.memmap(all_quat_shm_mapfile, dtype=config.np_dtype, mode='r+', shape=(n_selected_BBs,4))
                    # all_R_memmap = np.memmap(all_R_shm_mapfile, dtype=config.np_dtype, mode='r+', shape=(n_selected_BBs,3,3))
                    # all_dJdq_memmap = np.memmap(all_dJdq_shm_mapfile, dtype=config.np_dtype, mode='r+', shape=(n_selected_BBs,6))
                    dq_memmap = np.ndarray((7,), dtype=config.np_dtype, buffer=dq_shm_mapfile)
                    all_P_memmap = np.ndarray((n_selected_BBs,3), dtype=config.np_dtype, buffer=all_P_shm_mapfile)
                    all_J_memmap = np.ndarray((n_selected_BBs,6,7), dtype=config.np_dtype, buffer=all_J_shm_mapfile)
                    all_quat_memmap = np.ndarray((n_selected_BBs,4), dtype=config.np_dtype, buffer=all_quat_shm_mapfile)
                    all_R_memmap = np.ndarray((n_selected_BBs,3,3), dtype=config.np_dtype, buffer=all_R_shm_mapfile)
                    all_dJdq_memmap = np.ndarray((n_selected_BBs,6), dtype=config.np_dtype, buffer=all_dJdq_shm_mapfile)

                    np.copyto(dq_memmap, dq)
                    np.copyto(all_P_memmap, all_P_np)
                    np.copyto(all_J_memmap, all_J_np)
                    np.copyto(all_quat_memmap, all_quat_np)
                    np.copyto(all_R_memmap, all_R_np)
                    np.copyto(all_dJdq_memmap, all_dJdq_np)
                    
                    time_cvxpy_and_diff_helper_tmp -= time.time()
                    futures = []
                    for (ii, bb_key) in enumerate(selected_BBs):
                        for (jj, obs_key) in enumerate(obs_col.face_equations.keys()):
                            solver = all_solvers[ii*n_obstacle+jj]
                            futures.append(executor.submit(solver.solve))
                    done, _ = concurrent.futures.wait(futures)
                    time_cvxpy_and_diff_helper_tmp += time.time()

                    all_h_memmap = np.ndarray((n_CBF,), dtype=config.np_dtype, buffer=all_h_shm_mapfile)
                    # all_h_dx_memmap = np.ndarray((n_CBF,7), dtype=config.np_dtype, buffer=all_h_dx_shm_mapfile)
                    # all_h_dxdx_memmap = np.ndarray((n_CBF,7,7), dtype=config.np_dtype, buffer=all_h_dxdx_shm_mapfile)
                    all_phi1_memmap = np.ndarray((n_CBF,), dtype=config.np_dtype, buffer=all_phi1_shm_mapfile)
                    all_actuation_memmap = np.ndarray((n_CBF,7), dtype=config.np_dtype, buffer=all_actuation_shm_mapfile)
                    all_lb_memmap = np.ndarray((n_CBF,), dtype=config.np_dtype, buffer=all_lb_shm_mapfile)
                    all_ub_memmap = np.ndarray((n_CBF,), dtype=config.np_dtype, buffer=all_ub_shm_mapfile)

                    CBF_tmp[:] = all_h_memmap[:]
                    phi1_tmp[:] = all_phi1_memmap[:]
                    C[:n_CBF,:] = all_actuation_memmap[:]
                    lb[:n_CBF] = all_lb_memmap[:]
                    ub[:n_CBF] = all_ub_memmap[:]

                    # for kk in range(n_CBF):
                    #     result = futures[kk].result()
                    #     _h, _, _, _phi1, _actuation, _lb, _ub = result
                    #     C[kk,:] = _actuation
                    #     lb[kk] = _lb
                    #     ub[kk] = _ub
                    #     CBF_tmp[kk] = _h
                    #     phi1_tmp[kk] = _phi1

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
        except KeyboardInterrupt:
            print("Caught KeyboardInterrupt, terminating workers")
            executor.shutdown(wait=False)
            posix_ipc.unlink_shared_memory(dq_shm_id)
            posix_ipc.unlink_shared_memory(all_J_shm_id)
            posix_ipc.unlink_shared_memory(all_quat_shm_id)
            posix_ipc.unlink_shared_memory(all_R_shm_id)
            posix_ipc.unlink_shared_memory(all_P_shm_id)
            posix_ipc.unlink_shared_memory(all_dJdq_shm_id)
        else:
            print("All tasks completed successfully")
        finally:
            executor.shutdown(wait=False)
            print("==> All workers have been terminated")
            posix_ipc.unlink_shared_memory(dq_shm_id)
            posix_ipc.unlink_shared_memory(all_J_shm_id)
            posix_ipc.unlink_shared_memory(all_quat_shm_id)
            posix_ipc.unlink_shared_memory(all_R_shm_id)
            posix_ipc.unlink_shared_memory(all_P_shm_id)
            posix_ipc.unlink_shared_memory(all_dJdq_shm_id)

    # Save summary
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

    fig, ax = plt.subplots(figsize=(10,8), dpi=config.dpi, frameon=True)
    plt.plot(times, joint_angles[:,0], linestyle="-", label=r"$q_1$")
    plt.plot(times, joint_angles[:,1], linestyle="-", label=r"$q_2$")
    plt.plot(times, joint_angles[:,2], linestyle="-", label=r"$q_3$")
    plt.plot(times, joint_angles[:,3], linestyle="-", label=r"$q_4$")
    plt.plot(times, joint_angles[:,4], linestyle="-", label=r"$q_5$")
    plt.plot(times, joint_angles[:,5], linestyle="-", label=r"$q_6$")
    plt.plot(times, joint_angles[:,6], linestyle="-", label=r"$q_7$")
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
