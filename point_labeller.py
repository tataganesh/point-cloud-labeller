import pykitti
import numpy as np
import cv2
import open3d as o3d
import matplotlib.pyplot as plt
import time
from mask_predictor import MaskPredictor, show_mask

cmap = plt.cm.jet


def scalars_to_colors(scalars):
    N = 256
    bins = np.linspace(scalars.min(), scalars.max(), N + 1)
    dig = np.digitize(scalars, bins) - 1
    dig[dig == N] = N - 1  # map the last half-open interval back
    norm = plt.Normalize(scalars.min(), scalars.max())
    colors = cmap(norm(scalars))
    return colors


def draw_points(image, image_coordinates, color_indices):
    for i, (u, v, z) in enumerate(image_coordinates):
        color = cmap[color_indices[i], :]
        u = int(u)
        v = int(v)
        cv2.circle(image, (u, v), 2, tuple(color), -1)
    return image


basedir = "/Volumes/Expansion/KITTI_datasets/KITTI_Raw/raw_data_downloader"
date = "2011_09_26"
drive = "0005"

# The 'frames' argument is optional - default: None, which loads the whole dataset.
# Calibration, timestamps, and IMU data are read automatically.
# Camera and velodyne data are available via properties that create generators
# when accessed, or through getter methods that provide random access.
data = pykitti.raw(basedir, date, drive)

point_velo = np.array([0, 0, 0, 1])
point_cam0 = data.calib.T_cam2_velo.dot(point_velo)

vis = o3d.visualization.Visualizer()
vis.create_window()
pcd = o3d.geometry.PointCloud()


cmap = plt.cm.get_cmap("hsv", 256)
cmap = np.array([cmap(i) for i in range(256)])[:, :3] * 255

## Init predictor
predictor = MaskPredictor()


for idx in range(0, 100, 10):
    cam2_image_pil, cam3_image = data.get_rgb(idx)
    cam2_image = np.asarray(cam2_image_pil).copy()
    image_height, image_width = cam2_image.shape[:2]

    ## Calculate velo -> cam2-image-coordinates (homogenous)
    p_cam2 = np.zeros((3, 4))
    p_cam2[:3, :3] = data.calib.K_cam2
    T_cam2img_velo = np.matmul(p_cam2, data.calib.T_cam2_velo)
    points = data.get_velo(idx)[:, :3]  ## Points in lidar frame
    image_coordinates_homogenous = np.matmul(
        T_cam2img_velo, np.c_[points, np.ones(points.shape[0])].T
    ).T

    ## Homogenous to cartesian image coordinates (pixels)
    z_camera_frame = image_coordinates_homogenous[:, 2]
    image_coordinates = image_coordinates_homogenous[:, :]
    image_coordinates[:, :2] = (
        image_coordinates_homogenous[:, :2] / z_camera_frame[:, None]
    )

    ## 1. Mask negative image coordinate frame depth values
    ## 2. Mask out-of-bounds image coordinates
    valid_image_indices = np.where(
        (z_camera_frame > 0)
        & np.logical_and(
            image_coordinates[:, 0] > 0, image_coordinates[:, 0] < image_width
        )
        & np.logical_and(
            image_coordinates[:, 1] > 0, image_coordinates[:, 1] < image_height
        )
    )[0]
    image_coordinates = image_coordinates[valid_image_indices]
    color_indices = (
        (image_coordinates[:, 2] - z_camera_frame.min())
        / (z_camera_frame.max() - z_camera_frame.min())
        * 255
    ).astype(int)

    lidar_point_correspondences = points[valid_image_indices]
    image_coordinates = image_coordinates.astype(int)
    # rgb_values = cam2_image[image_coordinates[:, 1], image_coordinates[:, 0]]
    image_to_point_index_map = np.full(
        (image_height, image_width), fill_value=-1, dtype=int
    )
    image_to_point_index_map[image_coordinates[:, 1], image_coordinates[:, 0]] = (
        valid_image_indices
    )

    text = "car."
    mask, id_label_mapping = predictor.inference(cam2_image_pil, text)
    # show_mask(cam2_image_pil, mask)

    label_corresponding_point_idx = image_to_point_index_map[mask == 1]
    label_corresponding_point_idx = label_corresponding_point_idx[
        label_corresponding_point_idx != -1
    ]  # Remove all pixels for which points are NOT available

    # cam2_image = draw_points(cam2_image, image_coordinates, color_indices)
    point_colors = np.full_like(points, (0, 0.1, 0.4))
    point_colors[label_corresponding_point_idx] = (1, 0, 0)
    # point_colors[random_idx] = POINT_IMAGE_COLOR
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(point_colors)
    cv2.imshow("window", cam2_image)
    # o3d.visualization.draw_geometries([pcd])

    # print(parameters.intrinsic)
    ctr = vis.get_view_control()
    parameters = o3d.camera.PinholeCameraParameters()
    parameters.intrinsic = o3d.camera.PinholeCameraIntrinsic(
        image_width, image_height, data.calib.K_cam2
    )  # p_cam2
    parameters.extrinsic = data.calib.T_cam2_velo
    ctr.convert_from_pinhole_camera_parameters(parameters, True)
    # # # ctr.set_up([-0.0694, -0.9768, 0.2024])
    # # ctr.set_zoom(0.5)
    if idx == 0:
        vis.add_geometry(pcd)

    vis.update_geometry(pcd)
    vis.poll_events()
    vis.update_renderer()
    # input()
