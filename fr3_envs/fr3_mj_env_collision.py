import time
from copy import deepcopy

import matplotlib.pyplot as plt
import mujoco
import mujoco.viewer
import numpy as np
import pinocchio as pin
from pinocchio.robot_wrapper import RobotWrapper
from scipy.spatial.transform import Rotation

import os
from pathlib import Path


class FR3MuJocoEnv:
    def __init__(self, render=True, xml_name="fr3", urdf_name="fr3_with_camera_and_bounding_boxes", base_pos=[0,0,0], base_quat=[0,0,0,1],
                 cam_distance=3.0, cam_azimuth=-90, cam_elevation=-45, cam_lookat=[0.0, -0.25, 0.824], dt=1.0/240):
        package_directory = str(Path(__file__).parent.parent)

        self.model = mujoco.MjModel.from_xml_path(package_directory + f"/robots_mj/{xml_name}.xml")

        # Override the simulation timestep.
        self.model.opt.timestep = dt

        self.data = mujoco.MjData(self.model)

        robot_URDF = package_directory + "/robots_mj/{}.urdf".format(urdf_name)
        self.pin_robot = RobotWrapper.BuildFromURDF(robot_URDF, package_directory)

        if render:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.render = True
            self.viewer.cam.distance = cam_distance
            self.viewer.cam.azimuth = cam_azimuth
            self.viewer.cam.elevation = cam_elevation
            self.viewer.cam.lookat[:] = np.array(cam_lookat)
        else:
            self.render = False

        self.model.opt.gravity[2] = -9.81
        self.renderer = None

        # Define base position and orientation
        self.base_p_offset = np.array(base_pos).reshape(-1,1)
        self.base_R_offset = Rotation.from_quat(base_quat).as_matrix()

        # Get frame ID for world
        self.jacobian_frame = pin.ReferenceFrame.LOCAL_WORLD_ALIGNED

        # Get frame ids for bounding boxes
        self.FR3_LINK3_BB_FRAME_ID = self.pin_robot.model.getFrameId("fr3_link3_bounding_box")
        self.FR3_LINK4_BB_FRAME_ID = self.pin_robot.model.getFrameId("fr3_link4_bounding_box")
        self.FR3_LINK5_1_BB_FRAME_ID = self.pin_robot.model.getFrameId("fr3_link5_1_bounding_box")
        self.FR3_LINK5_2_BB_FRAME_ID = self.pin_robot.model.getFrameId("fr3_link5_2_bounding_box")
        self.FR3_LINK6_BB_FRAME_ID = self.pin_robot.model.getFrameId("fr3_link6_bounding_box")
        self.FR3_LINK7_BB_FRAME_ID = self.pin_robot.model.getFrameId("fr3_link7_bounding_box")
        self.FR3_HAND_BB_FRAME_ID = self.pin_robot.model.getFrameId("fr3_hand_bounding_box")

        # Get frame ID for links
        self.FR3_LINK3_FRAME_ID = self.pin_robot.model.getFrameId("fr3_link3")
        self.FR3_LINK4_FRAME_ID = self.pin_robot.model.getFrameId("fr3_link4")
        self.FR3_LINK5_FRAME_ID = self.pin_robot.model.getFrameId("fr3_link5")
        self.FR3_LINK5_FRAME_ID = self.pin_robot.model.getFrameId("fr3_link5")
        self.FR3_LINK6_FRAME_ID = self.pin_robot.model.getFrameId("fr3_link6")
        self.FR3_LINK7_FRAME_ID = self.pin_robot.model.getFrameId("fr3_link7")
        self.FR3_HAND_FRAME_ID = self.pin_robot.model.getFrameId("fr3_hand")
        self.EE_FRAME_ID = self.pin_robot.model.getFrameId("fr3_hand_tcp")
        self.FR3_CAMERA_FRAME_ID = self.pin_robot.model.getFrameId("fr3_camera")

        # Choose the useful frame names with frame ids 
        self.frame_names_and_ids = {
            "LINK3_BB": self.FR3_LINK3_BB_FRAME_ID,
            "LINK4_BB": self.FR3_LINK4_BB_FRAME_ID,
            "LINK5_1_BB": self.FR3_LINK5_1_BB_FRAME_ID,
            "LINK5_2_BB": self.FR3_LINK5_2_BB_FRAME_ID,
            "LINK6_BB": self.FR3_LINK6_BB_FRAME_ID,
            "LINK7_BB": self.FR3_LINK7_BB_FRAME_ID,
            "HAND_BB": self.FR3_HAND_BB_FRAME_ID,
            # "HAND": self.FR3_HAND_FRAME_ID,
            "EE": self.EE_FRAME_ID,
            # "CAMERA": self.FR3_CAMERA_FRAME_ID,
        }
        

    def reset(self, q_nominal):
        for i in range(7):
            self.data.qpos[i] = q_nominal[i]

        self.data.qpos[7] = 0.0
        self.data.qpos[8] = 0.0

        q, dq = self.data.qpos[:9].copy(), self.data.qvel[:9].copy()
        self.update_pinocchio(q, dq)
        info = self.get_info(q, dq)

        return info

    def step(self, tau, finger_pos):
        frc_applied = np.append(tau, finger_pos)

        self.data.ctrl[:] = frc_applied
        mujoco.mj_step(self.model, self.data)
        self.viewer.sync()

        q, dq = self.data.qpos[:9].copy(), self.data.qvel[:9].copy()
        self.update_pinocchio(q, dq)
        info = self.get_info(q, dq)

        return info

    def close(self):
        self.viewer.close()

    def sleep(self, start_time):
        time_until_next_step = self.model.opt.timestep - (time.time() - start_time)

        if time_until_next_step > 0:
            time.sleep(time_until_next_step)

    def update_pinocchio(self, q, dq):
        self.pin_robot.computeJointJacobians(q)
        self.pin_robot.framesForwardKinematics(q)
        self.pin_robot.centroidalMomentum(q, dq)

    def get_info(self, q, dq):
        """
        info contains:
        -------------------------------------
        q: joint position
        dq: joint velocity
        J_{frame_name}: jacobian of frame_name
        P_{frame_name}: position of frame_name
        R_{frame_name}: orientation of frame_name
        """

        # Get Jacobian from grasp target frame
        # preprocessing is done in get_state_update_pinocchio()
        info = {"q": q,
                "dq": dq}
        
        for frame_name, frame_id in self.frame_names_and_ids.items():
            # Frame jacobian
            info[f"J_{frame_name}"] = self.pin_robot.getFrameJacobian(frame_id, self.jacobian_frame)

            # Frame position and orientation
            (   info[f"P_{frame_name}"],
                info[f"R_{frame_name}"]
            ) = self.compute_crude_location(
                self.base_R_offset, self.base_p_offset, frame_id
            )

            # Advanced calculation
            info[f"dJdq_{frame_name}"] = pin.getFrameClassicalAcceleration(
                self.pin_robot.model, self.pin_robot.data, frame_id, self.jacobian_frame
            )

        M, Minv, nle = self.get_dynamics(q, dq)
        info["M"] = M
        info["Minv"] = Minv
        info["nle"] = nle
        info["G"] = self.pin_robot.gravity(q)
        Minv_mj = np.zeros_like(Minv)
        mujoco.mj_solveM(self.model, self.data, Minv_mj, np.eye(Minv_mj.shape[0]))
        info["Minv_mj"] = Minv_mj

        return info

    def get_dynamics(self, q, dq):
        """
        f.shape = (18, 1), g.shape = (18, 9)
        """
        Minv = pin.computeMinverse(self.pin_robot.model, self.pin_robot.data, q)
        M = self.pin_robot.mass(q)
        nle = self.pin_robot.nle(q, dq)

        return M, Minv, nle

    def get_depth_image(self, camera=-1, scene_option=None):
        if self.renderer is None:
            self.renderer = mujoco.Renderer(self.model)
            self.renderer.enable_depth_rendering()

        self.renderer.update_scene(self.data, camera=camera, scene_option=scene_option)
        depth = self.renderer.render()

        # process depth image
        depth -= depth.min()
        depth = np.clip(depth, 0.0, 4.0) / 4.0
        # depth /= 2 * depth[depth <= 1].mean()
        pixels = 255 * np.clip(depth, 0, 1)

        return pixels

    def show_depth_img(self, pixels):
        plt.imshow(pixels.astype(np.uint8))
        plt.show()

    def compute_crude_location(self, base_R_offset, base_p_offset, frame_id):
        # get link orientation and position
        _p = self.pin_robot.data.oMf[frame_id].translation
        _Rot = self.pin_robot.data.oMf[frame_id].rotation

        # compute link transformation matrix
        _T = np.hstack((_Rot, _p[:, np.newaxis]))
        T = np.vstack((_T, np.array([[0.0, 0.0, 0.0, 1.0]])))

        # compute link offset transformation matrix
        _TW = np.hstack((base_R_offset, base_p_offset))
        TW = np.vstack((_TW, np.array([[0.0, 0.0, 0.0, 1.0]])))
        
        # get transformation matrix
        T_mat = TW @ T 

        # compute crude model location
        p = (T_mat @ np.array([[0.0], [0.0], [0.0], [1.0]]))[:3, 0]

        # compute crude model orientation
        Rot = T_mat[:3, :3]

        return p, Rot