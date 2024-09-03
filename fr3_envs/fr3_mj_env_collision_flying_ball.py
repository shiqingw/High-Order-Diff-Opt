import time

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

        self.model = mujoco.MjModel.from_xml_path(package_directory + f"/fr3_mj/{xml_name}.xml")

        # Override the simulation timestep.
        self.model.opt.timestep = dt

        self.data = mujoco.MjData(self.model)

        robot_URDF = package_directory + "/fr3_mj/{}.urdf".format(urdf_name)
        self.pin_robot = RobotWrapper.BuildFromURDF(robot_URDF, package_directory)

        if render:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data, show_left_ui=False, show_right_ui=False)
            self.render = True
            self.viewer.cam.distance = cam_distance
            self.viewer.cam.azimuth = cam_azimuth
            self.viewer.cam.elevation = cam_elevation
            self.viewer.cam.lookat[:] = np.array(cam_lookat)

            self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
            self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
            # self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = True
        else:
            self.render = False

        self.model.opt.gravity[2] = -9.81
        self.renderer = None
        self.ngeom = 0
        self.maxgeom = 1000
        
        self.model.vis.scale.contactwidth = 0.1
        self.model.vis.scale.contactheight = 0.03
        self.model.vis.scale.forcewidth = 0.05
        self.model.vis.map.force = 0.3
        
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
            "CAMERA": self.FR3_CAMERA_FRAME_ID,
        }

        # self.joint_names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7", "finger_joint1", "finger_joint2"]
        

    def reset(self, q_nominal, ball_pos, ball_vel):
        for i in range(9):
            self.data.qpos[i] = q_nominal[i]
            self.data.qvel[i] = 0.0
        for i in range(3):
            self.data.qpos[i+9] = ball_pos[i]
            self.data.qvel[i+9] = ball_vel[i]
        mujoco.mj_step(self.model, self.data)
        if self.render:
            self.viewer.sync()
        q, dq = self.data.qpos[:9].copy(), self.data.qvel[:9].copy()
        self.update_pinocchio(q, dq)
        info = self.get_info(q, dq)

        return info

    def step(self, tau, finger_pos):
        frc_applied = np.append(tau, finger_pos)

        self.data.ctrl[:] = frc_applied
        mujoco.mj_step(self.model, self.data)
        if self.render:
            self.viewer.sync()

        q, dq = self.data.qpos[:9].copy(), self.data.qvel[:9].copy()
        self.update_pinocchio(q, dq)
        info = self.get_info(q, dq)

        return info

    def close(self):
        if self.render:
            self.viewer.close()

    def sleep(self, start_time):
        time_until_next_step = self.model.opt.timestep - (time.time() - start_time)

        if time_until_next_step > 0:
            time.sleep(time_until_next_step)

    def update_pinocchio(self, q, dq):
        self.pin_robot.computeJointJacobians(q)
        self.pin_robot.framesForwardKinematics(q)
        self.pin_robot.centroidalMomentum(q, dq)
        self.pin_robot.forwardKinematics(q, dq, 0*q) # 0*q is the acceleration set to zero
        pin.computeJointJacobiansTimeVariation(self.pin_robot.model, self.pin_robot.data, q, dq)
        pin.updateFramePlacements(self.pin_robot.model, self.pin_robot.data)

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
        # preprocessing is done in update_pinocchio()
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
            # dJ = pin.getFrameJacobianTimeVariation(
            #     self.pin_robot.model, self.pin_robot.data, frame_id, self.jacobian_frame)
            # info[f"dJdq_{frame_name}"] = dJ @ dq 

            # same as above
            info[f"dJdq_{frame_name}"] = pin.getFrameAcceleration(
                self.pin_robot.model, self.pin_robot.data, frame_id, self.jacobian_frame).vector

        M, Minv, nle = self.get_dynamics(q, dq)
        info["M"] = M
        info["Minv"] = Minv
        info["nle"] = nle
        info["G"] = self.pin_robot.gravity(q)
        Minv_mj = np.zeros((15,15))
        mujoco.mj_solveM(self.model, self.data, Minv_mj, np.eye(Minv_mj.shape[0]))
        info["Minv_mj"] = Minv_mj[:9,:9]
        info["nle_mj"] = self.data.qfrc_bias[:9]

        M_mj = np.zeros((15,15))
        mujoco.mj_fullM(self.model, M_mj, self.data.qM)
        info["M_mj"] = M_mj[:9,:9]

        info["qfrc_constraint"] = self.data.qfrc_constraint[:9]
        info["qfrc_smooth"] = self.data.qfrc_smooth[:9]
        info["qfrc_actuator"] = self.data.qfrc_actuator[:9]

        info["ball_pos"] = self.data.qpos[9:12]
        info["ball_vel"] = self.data.qvel[9:12]

        return info

    def ball_future_pos_and_vel(self, t):
        c0 = self.data.qpos[9:12]
        c1 = self.data.qvel[9:12]
        c2 = np.array([0,0,-9.81])
        future_pos = 0.5*t**2*c2 + t*c1 + c0
        future_vel = t*c2 + c1
        return future_pos, future_vel

    def get_dynamics(self, q, dq):
        """
        f.shape = (18, 1), g.shape = (18, 9)
        """
        Minv = pin.computeMinverse(self.pin_robot.model, self.pin_robot.data, q)
        M = self.pin_robot.mass(q)
        nle = self.pin_robot.nle(q, dq)

        return M, Minv, nle

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

    def add_visual_capsule(self, point1, point2, radius, rgba, id_geom_offset=0, limit_num=False):
        """Adds one capsule to an mjvScene."""
        if not self.render:
            return
        scene = self.viewer.user_scn
        if limit_num:
            if self.ngeom >= self.maxgeom:
                id_geom = self.ngeom % self.maxgeom + id_geom_offset
            else:
                scene.ngeom += 1
                id_geom = self.ngeom + id_geom_offset
            self.ngeom += 1
        elif id_geom_offset < scene.ngeom:
            id_geom = id_geom_offset
        else:
            id_geom = scene.ngeom
            scene.ngeom += 1
        # initialise a new capsule, add it to the scene using mjv_makeConnector
        mujoco.mjv_initGeom(scene.geoms[id_geom],
                            mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3),
                            np.zeros(3), np.zeros(9), np.array(rgba).astype(np.float32))
        mujoco.mjv_makeConnector(scene.geoms[id_geom],
                                mujoco.mjtGeom.mjGEOM_CAPSULE, radius,
                                point1[0], point1[1], point1[2],
                                point2[0], point2[1], point2[2])
        self.viewer.sync()
        return 
    
    def add_visual_ellipsoid(self, size, pos, mat, rgba, id_geom_offset=0, limit_num=False):
        """Adds one ellipsoid to an mjvScene."""
        if not self.render:
            return
        scene = self.viewer.user_scn
        if limit_num:
            if self.ngeom >= self.maxgeom:
                id_geom = self.ngeom % self.maxgeom + id_geom_offset
            else:
                scene.ngeom += 1
                id_geom = self.ngeom + id_geom_offset
            self.ngeom += 1
        elif id_geom_offset < scene.ngeom:
            id_geom = id_geom_offset
        else:
            id_geom = scene.ngeom
            scene.ngeom += 1
            
        # initialise a new ellipsoid, add it to the scene
        mujoco.mjv_initGeom(scene.geoms[id_geom],
                            mujoco.mjtGeom.mjGEOM_ELLIPSOID, np.array(size),
                            np.array(pos), np.array(mat).flatten(), np.array(rgba).astype(np.float32))
        self.viewer.sync()
        return
    
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

    def create_renderer(self, height, width):
        self.renderer = mujoco.Renderer(self.model, height, width)

    def get_rgb_image(self, camera, scene_option=None):
        self.renderer.update_scene(self.data, camera=camera, scene_option=scene_option)
        rgb = self.renderer.render()

        return rgb