import time
from copy import deepcopy

import mujoco
import mujoco.viewer
import numpy as np
from scipy.spatial.transform import Rotation

import os
from pathlib import Path


class SkydioMuJocoEnv:
    def __init__(self, render=True, xml_name="scene", cam_distance=3.0, cam_azimuth=-90,
                 cam_elevation=-45, cam_lookat=[0.0, -0.25, 0.824], dt=0.01):
        package_directory = str(Path(__file__).parent.parent)

        self.model = mujoco.MjModel.from_xml_path(package_directory + f"/skydio_x2/{xml_name}.xml")

        # Override the simulation timestep.
        self.model.opt.timestep = dt

        self.data = mujoco.MjData(self.model)

        if render:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data, show_left_ui=False, show_right_ui=False)
            self.render = True
            self.viewer.cam.distance = cam_distance
            self.viewer.cam.azimuth = cam_azimuth
            self.viewer.cam.elevation = cam_elevation
            self.viewer.cam.lookat[:] = np.array(cam_lookat)
        else:
            self.render = False

        self.model.opt.gravity[2] = 0
        self.renderer = None
        self.ngeom = 0
        self.maxgeom = 200

        self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True # contact points
        self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True # contact force
        # self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = True # make dynamic geoms more transparent
        
        self.model.vis.scale.contactwidth = 0.1
        self.model.vis.scale.contactheight = 0.03
        self.model.vis.scale.forcewidth = 0.05
        self.model.vis.map.force = 0.3

    def reset(self, pos, quat, v, omega):
        self.data.ctrl[:] = np.zeros(4)[:]
        self.data.qpos[:3] = pos[:]
        self.data.qpos[3:] = quat[:]
        self.data.qvel[:3] = v[:]
        self.data.qvel[3:] = omega[:]
        mujoco.mj_step(self.model, self.data)
        self.viewer.sync()

    def step(self, pos, quat, v, omega):
        self.data.ctrl[:] = np.zeros(4)[:]
        self.data.qpos[:3] = pos[:]
        self.data.qpos[3:] = quat[:]
        self.data.qvel[:3] = v[:]
        self.data.qvel[3:] = omega[:]
        mujoco.mj_step(self.model, self.data)
        self.viewer.sync()

    def close(self):
        self.viewer.close()

    def add_visual_capsule(self, point1, point2, radius, rgba, id_geom_offset=0, limit_num=False):
        """Adds one capsule to an mjvScene."""
        scene = self.viewer.user_scn
        if limit_num:
            if self.ngeom >= self.maxgeom:
                id_geom = self.ngeom % self.maxgeom + id_geom_offset
            else:
                scene.ngeom += 1
                id_geom = self.ngeom + id_geom_offset
            self.ngeom += 1
        else:
            id_geom = scene.ngeom
            scene.ngeom += 1
        # initialise a new capsule, add it to the scene using mjv_makeConnector
        mujoco.mjv_initGeom(scene.geoms[id_geom],
                            mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3),
                            np.zeros(3), np.eye(3).flatten(), np.array(rgba).astype(np.float32))
        mujoco.mjv_makeConnector(scene.geoms[id_geom],
                                mujoco.mjtGeom.mjGEOM_CAPSULE, radius,
                                point1[0], point1[1], point1[2],
                                point2[0], point2[1], point2[2])
        self.viewer.sync()
        return 
    
    def add_visual_ellipsoid(self, size, pos, mat, rgba, id_geom_offset=0, limit_num=False):
        """Adds one ellipsoid to an mjvScene."""
        scene = self.viewer.user_scn
        if limit_num:
            if self.ngeom >= self.maxgeom:
                id_geom = self.ngeom % self.maxgeom + id_geom_offset
            else:
                scene.ngeom += 1
                id_geom = self.ngeom + id_geom_offset
            self.ngeom += 1
        else:
            id_geom = scene.ngeom
            scene.ngeom += 1
            
        # initialise a new ellipsoid, add it to the scene
        mujoco.mjv_initGeom(scene.geoms[id_geom],
                            mujoco.mjtGeom.mjGEOM_ELLIPSOID, np.array(size),
                            np.array(pos), np.array(mat).flatten(), np.array(rgba).astype(np.float32))
        self.viewer.sync()
        return