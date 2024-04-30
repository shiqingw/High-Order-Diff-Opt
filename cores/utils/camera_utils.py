import mujoco
import numpy as np

def get_camera_intrinsic_matrix(mujoco_model, camera_name, camera_height, camera_width):
    """
    Obtains camera intrinsic matrix.

    Args:
        mujoco_model (mjModel): mujoco model
        camera_name (str): name of camera
        camera_height (int): height of camera images in pixels
        camera_width (int): width of camera images in pixels
    Return:
        K (np.array): 3x3 camera matrix
    """
    camera_id = mujoco.mj_name2id(mujoco_model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
    print(camera_id)
    fovy = mujoco_model.cam_fovy[camera_id]
    f = 0.5 * camera_height / np.tan(fovy * np.pi / 360)
    K = np.array([[f, 0, camera_width / 2], [0, f, camera_height / 2], [0, 0, 1]])
    return K

def project_points_from_world_to_pixel(points_in_world, K, H_world_to_cam, image_width, image_height):
    """
    Helper function to project a batch of points in the world frame
    into camera pixels using the world to camera transformation.

    Args:
        points_in_world (np.array): 3D points in world frame to project onto camera pixel locations, shape (N,3).
        K (np.array): camera intrinsic matrix, shape (3,3)
        H_world_to_ca (np.array): homogenous matrix from world to camera frame, shape (4,4)
        image_width (int): width of camera images in pixels
        image_height (int): height of camera images in pixels

    Return:
        pixels (np.array): projected pixel indices, shape (N, 2)
    """

    if points_in_world.ndim == 1:
        points_in_world = points_in_world[np.newaxis,:]

    ones_pad = np.ones((points_in_world.shape[0],1))
    points_in_world = np.concatenate((points_in_world, ones_pad), axis=-1)  # shape (N, 4)

    points_in_cam = points_in_world @ H_world_to_cam.T # shape (N, 4)
    points_in_cam = points_in_cam[:,:3]  # shape (N, 3)

    pixels = points_in_cam @ K.T  # shape (N, 3)
    pixels = pixels / pixels[:,2] 
    pixels = pixels[:, :2].round().astype(int)  # shape (N, 3)
    pixels[:, 0] = np.clip(pixels[:, 0], 0, image_width - 1)
    pixels[:, 1] = np.clip(pixels[:, 1], 0, image_height - 1)

    return pixels

def project_points_from_pixels_to_world(pixels, depth, K, H_cam_to_world):
    """
    Helper function to project a batch of pixels in the camera frame
    into 3D world points using the camera to world transformation.

    Args:
        pixels (np.array): pixel indices, shape (N, 2)
        depth (np.array): depth values of pixels, shape (N,)
        K (np.array): camera intrinsic matrix, shape (3,3)
        H_cam_to_world (np.array): homogenous matrix from camera to world frame, shape (4,4)

    Return:
        points_in_world (np.array): 3D points in world frame, shape (N, 3)
    """

    if pixels.ndim == 1:
        pixels = pixels[np.newaxis,:]

    ones_pad = np.ones((pixels.shape[0],1))
    pixels = np.concatenate((pixels, ones_pad), axis=-1)  # shape (N, 3)
    pixels = pixels * depth[:, np.newaxis]  # shape (N, 3)

    points_in_cam = pixels @ np.linalg.inv(K).T  # shape (N, 3)
    points_in_cam = np.concatenate((points_in_cam, ones_pad), axis=-1)  # shape (N, 4)

    points_in_world = points_in_cam @ H_cam_to_world.T  # shape (N, 4)
    points_in_world = points_in_world[:,:3]  # shape (N, 3)

    return points_in_world