import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Tuple
from parse_synchronized_frames import get_image_pointcloud_pairs

proj_matrix = np.array([
    [958.25209249, 0, 617.35434211, 0],
    [0, 957.39654998, 357.38740741, 0],
    [0, 0, 1, 0]
])
extrinsic = np.array([
    [-1.2582e-05, -0.999986, 0.00523595, 0.297609],
    [-0.0124447, -0.00523539, -0.999909, -0.0185331],
    [0.999923, -7.77404e-05, -0.0124444, -0.000275199],
    [0, 0, 0, 1]
])

def project_points_to_camera(
    points: np.ndarray, proj_matrix: np.ndarray, cam_res: Tuple[int, int], max_dist: float = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Projects 3D points to the camera image plane.
    
    Parameters:
        points (np.ndarray): 4 x N array of homogeneous coordinates.
        proj_matrix (np.ndarray): 3 x 4 projection matrix.
        cam_res (tuple): Resolution of camera as (width, height).
        
    Returns:
        uv_valid (np.ndarray): 2 x M array of pixel coordinates.
        depths_valid (np.ndarray): 1 x M array of depths.
        original_indices (np.ndarray): Indices of the original points that are valid.
    """
    if points.shape[0] == 3:
        points = np.vstack((points, np.ones((1, points.shape[1]))))
    if max_dist is not None:
        in_range = np.linalg.norm(points[:2, :], axis=0) <= max_dist
    else:
        in_range = np.ones(points.shape[1]).astype(bool)
    in_image = points[2, :] > 0  # Filter for points in front of the camera
    depths = points[2, in_image & in_range]
    uvw = np.dot(proj_matrix, points[:, in_image & in_range])
    uv = uvw[:2, :]
    w = uvw[2, :]
    uv[0, :] /= w
    uv[1, :] /= w
    valid = (
        (uv[0, :] >= 0) & (uv[0, :] < cam_res[0]) &
        (uv[1, :] >= 0) & (uv[1, :] < cam_res[1])
    )
    uv_valid = uv[:, valid].astype(int)
    depths_valid = depths[valid]
    original_indices = np.where(in_image & in_range)[0][valid]
    return uv_valid, depths_valid, original_indices

def depths_to_colors(depths: np.ndarray, max_depth: int = 10, cmap: str = "hsv") -> np.ndarray:
    """
    Maps depth values to RGB colors using a specified colormap.
    
    Parameters:
        depths (np.ndarray): Array of depth values.
        max_depth (int): Maximum depth value for normalization.
        cmap (str): Name of the Matplotlib colormap.
        
    Returns:
        np.ndarray: Array of RGB color values.
    """
    depths = np.clip(depths / max_depth, 0, 1)
    to_colormap = plt.get_cmap(cmap)
    rgba_values = to_colormap(depths, bytes=True)
    return rgba_values[:, :3].astype(int)

def main(bag_path: str, max_dist: float, fps: int = 4, save_name: str = 'pc_projection.avi'):
    # Output video filename
    output_video = save_name
    
    # Video settings: resolution (width, height) and frame rate (fps)
    frame_size = (1280, 720)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_writer = cv2.VideoWriter(output_video, fourcc, fps, frame_size)

    generator = get_image_pointcloud_pairs(bag_path=bag_path)
    for image, points in tqdm(generator):
        points = points.T
        points = np.vstack([points, np.ones((1, points.shape[1]))])
        
        points_cam = extrinsic @ points

        uv, depths, _ = project_points_to_camera(points_cam, proj_matrix, frame_size, max_dist)
        rgb_colors = depths_to_colors(depths, max_depth=10)

        for point, color in zip(uv.T, rgb_colors):
            cv2.circle(
                image, 
                tuple(point), 
                radius=2, 
                color=(int(color[0]), int(color[1]), int(color[2])), thickness=cv2.FILLED
            )

        image = cv2.resize(image, frame_size)
        video_writer.write(image)

    video_writer.release()
    print(f'Video saved as: {output_video}')

if __name__ == '__main__':
    main(
        'fixed_rosbag/fixed_rosbag_0.db3',
        save_name='fixed.avi',
        max_dist=50
    )
