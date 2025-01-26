import pykitti
import numpy as np
import cv2
import open3d as o3d
import matplotlib.pyplot as plt
from mask_predictor import MaskPredictor, show_mask

vis = o3d.visualization.Visualizer()
vis.create_window()
pcd = o3d.geometry.PointCloud()


def draw_bboxes(image: np.ndarray, bboxes: np.ndarray) -> np.ndarray:
    for box in bboxes:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 1)
    return image


def get_points_corresponding_to_mask(
    lidar_points, id_label_mapping, image_to_point_index_map, segmentation_mask
):
    point_labels = np.full(lidar_points.shape[0], fill_value=-1)
    object_boxes = list()
    for label_id, _ in id_label_mapping.items():
        label_corresponding_point_idx = image_to_point_index_map[
            segmentation_mask == label_id
        ]
        label_corresponding_point_idx = label_corresponding_point_idx[
            label_corresponding_point_idx != -1
        ]  # Remove indices where pixel coordinates do NOT have a corresponding point

        ## Cluster object points and label points in largest cluster
        object_points = lidar_points[label_corresponding_point_idx]
        object_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(object_points))
        object_pcd, outlier_indices = object_pcd.remove_statistical_outlier(
            nb_neighbors=20, std_ratio=0.5
        )
        outlier_indices = np.array(outlier_indices)
        cluster_ids = object_pcd.cluster_dbscan(eps=1.0, min_points=10)
        cluster_ids = np.asarray(cluster_ids)
        filtered_cluster_ids = cluster_ids[
            cluster_ids != -1
        ]  # Remove points marked as outliers
        if filtered_cluster_ids.size:
            counts = np.bincount(filtered_cluster_ids)
            largest_cluster_ID = np.argmax(counts)
            label_corresponding_point_idx = label_corresponding_point_idx[
                outlier_indices[cluster_ids == largest_cluster_ID]
            ]
            point_labels[label_corresponding_point_idx] = label_id

            ## Create 3D bbox
            object_boxes_3d = (
                o3d.geometry.OrientedBoundingBox.create_from_points_minimal(
                    o3d.utility.Vector3dVector(
                        lidar_points[label_corresponding_point_idx]
                    )
                )
            )
            object_boxes_3d.color = [0, 1, 0]
            object_boxes.append(object_boxes_3d)
            print(np.asarray(object_boxes_3d.get_box_points()))

    return point_labels, object_boxes


## Init predictor
predictor = MaskPredictor()


def label_points(image, lidar_points, text_prompt, T_IMAGE_LIDAR):
    image_width, image_height = image.size
    image_coordinates_homogenous = np.matmul(
        T_IMAGE_LIDAR, np.c_[lidar_points, np.ones(lidar_points.shape[0])].T
    ).T

    ## Homogenous to cartesian image coordinates (pixels)
    z_camera_frame = image_coordinates_homogenous[:, 2]
    image_coordinates = image_coordinates_homogenous[:, :]
    image_coordinates[:, :2] = (
        image_coordinates_homogenous[:, :2] / z_camera_frame[:, None]
    )

    ## 1. Mask points behind image
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
    image_coordinates = image_coordinates[valid_image_indices].astype(int)
    image_to_point_index_map = np.full(
        (image_height, image_width), fill_value=-1, dtype=int
    )
    image_to_point_index_map[image_coordinates[:, 1], image_coordinates[:, 0]] = (
        valid_image_indices
    )

    ## Get segmention mask
    segmentation_mask, id_label_mapping, bounding_boxes = predictor.inference(
        image, text_prompt
    )
    print(id_label_mapping, bounding_boxes)

    ## Get
    point_labels, object_boxes = get_points_corresponding_to_mask(
        lidar_points, id_label_mapping, image_to_point_index_map, segmentation_mask
    )
    point_colors = np.full_like(lidar_points, (0.5, 0.5, 0.5))
    for label_id, _ in id_label_mapping.items():
        point_colors[point_labels == label_id] = (1, 0, 0)
    # lidar_points[:, 2] = 0
    pcd.points = o3d.utility.Vector3dVector(lidar_points)
    pcd.colors = o3d.utility.Vector3dVector(point_colors)
    image_with_bboxes = draw_bboxes(
        cv2.cvtColor(np.asarray(image).copy(), cv2.COLOR_BGR2RGB), bounding_boxes
    )
    return image_with_bboxes, pcd, object_boxes

    # if idx == 0:
    #     vis.add_geometry(pcd)
    # ctr.set_zoom(0.3)
    # vis.update_geometry(pcd)
    # vis.poll_events()
    # vis.update_renderer()


### KITTI Constants

KITTI_RAW_DATA_BASEDIR = (
    "/Volumes/Expansion/KITTI_datasets/KITTI_Raw/raw_data_downloader"
)
DATE = "2011_09_26"
drive = "0015"
pykitti_data = pykitti.raw(KITTI_RAW_DATA_BASEDIR, DATE, drive)


## Calculate transformation matrix from 3D velodyne points to 2D Cam2 image coordinates
p_cam2 = np.zeros((3, 4))
p_cam2[:3, :3] = pykitti_data.calib.K_cam2
T_cam2img_velo = np.matmul(p_cam2, pykitti_data.calib.T_cam2_velo)


## Input KITTI sequence (Preferably an ID)
## Output - Visualized and segmented point cloud.


text_prompt = "person.cars.street pole."
for idx in range(0, 100, 10):
    cam2_image_pil, cam3_image = pykitti_data.get_rgb(idx)
    lidar_points = pykitti_data.get_velo(idx)[:, :3]  ## Points in lidar frame
    print(pykitti_data.get_velo(idx).shape)
    image_with_boxes, point_cloud, object_boxes = label_points(
        cam2_image_pil, lidar_points, text_prompt, T_cam2img_velo
    )
    cv2.imshow("window", image_with_boxes)
    # o3d.visualization.draw_geometries([pcd, *object_boxes])
    vis.create_window()

    # Call only after creating visualizer window.
    vis.get_render_option().background_color = [0, 0, 0]
    # if idx == 0:
    #     vis.add_geometry(pcd)
    vis.add_geometry(pcd)
    for box in object_boxes:
        vis.add_geometry(box)

    ctr = vis.get_view_control()

    parameters = o3d.camera.PinholeCameraParameters()
    parameters.intrinsic = o3d.camera.PinholeCameraIntrinsic(
        image_with_boxes.shape[1], image_with_boxes.shape[0], pykitti_data.calib.K_cam2
    )  # p_cam2
    parameters.extrinsic = pykitti_data.calib.T_cam2_velo
    ctr.convert_from_pinhole_camera_parameters(parameters, True)

    vis.poll_events()
    vis.update_renderer()

    if not vis.poll_events():
        break
    vis.clear_geometries()
