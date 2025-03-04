import numpy as np
from typing import Tuple
import open3d as o3d
import cv2
import matplotlib.pyplot as plt

# Projection matrix and extrinsic matrix
proj_matrix = np.array([
    [1182.69, 0, 628.246, 0],
    [0, 1081.13, 354.768, 0],
    [0, 0, 1, 0]
])
extrinsic = np.array([
    [0.0178424, -0.999837, 0.00271008, 0.172524],
    [-0.0212749, -0.00308955, -0.999769, 0.00650253],
    [0.999615, 0.0177807, -0.0213265, -0.0636636],
    [0, 0, 0, 1]
])

img_path = '1740747812728717295/image_1740747812728717295.png'
pc_path = '1740747812728717295/point_cloud_1740747812724153385.pcd'
label_path = '1740747812728717295/image_1740747812728717295.txt'

def project_points_to_camera(
    points: np.ndarray, proj_matrix: np.ndarray, cam_res: Tuple[int, int]
) -> Tuple[np.ndarray, np.ndarray]:
    if points.shape[0] == 3:
        points = np.vstack((points, np.ones((1, points.shape[1]))))
    if len(points.shape) != 2 or points.shape[0] != 4:
        raise ValueError(
            f"Wrong shape of points array: {points.shape}; expected: (4, n), where n - number of points."
        )
    if proj_matrix.shape != (3, 4):
        raise ValueError(f"Wrong proj_matrix shape: {proj_matrix}; expected: (3, 4).")
    in_image = points[2, :] > 0
    depths = points[2, in_image]
    uvw = np.dot(proj_matrix, points[:, in_image])
    uv = uvw[:2, :]
    w = uvw[2, :]
    uv[0, :] /= w
    uv[1, :] /= w
    in_image = (uv[0, :] >= 0) * (uv[0, :] < cam_res[0]) * (uv[1, :] >= 0) * (uv[1, :] < cam_res[1])
    return uv[:, in_image].astype(int), depths[in_image]

def load_yolo_segmentation_labels(label_path: str, image_shape: Tuple[int, int]) -> np.ndarray:
    # Height and width from image_shape (H, W)
    height, width = image_shape
    mask = np.zeros((height, width), dtype=np.uint8)
    
    with open(label_path, 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        parts = line.strip().split()
        if not parts:
            continue
        class_id = int(parts[0])
        polygon = np.array([float(x) for x in parts[1:]]).reshape(-1, 2)
        # Scale normalized coordinates to pixels: (x * width, y * height)
        polygon = (polygon * np.array([width, height])).astype(np.int32)
        cv2.fillPoly(mask, [polygon], color=class_id + 1)  # Offset class_id to avoid 0
    
    return mask

def depths_to_colors(depths: np.ndarray, max_depth: int = 10, cmap: str = "hsv") -> np.ndarray:
    depths /= max_depth
    to_colormap = plt.get_cmap(cmap)
    rgba_values = to_colormap(depths, bytes=True)
    return rgba_values[:, :3].astype(int)

# Load point cloud
pc = o3d.io.read_point_cloud(pc_path)
points = np.asarray(pc.points).T
points = np.vstack([points[0], points[1], points[2], np.ones((1, points[0].shape[0]))])

# Transform points to camera coordinates
points_cam = extrinsic @ points

# Project points to image plane
uv, depths = project_points_to_camera(points_cam, proj_matrix, (1280, 720))

image = cv2.imread(img_path)
rgb_distances = depths_to_colors(depths, max_depth=10)
for point, d in zip(uv.T, rgb_distances):
    c = (int(d[0]), int(d[1]), int(d[2]))
    cv2.circle(image, point, radius=2, color=c, thickness=cv2.FILLED)

plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis(False)
plt.show()

# Load YOLO segmentation labels
segmentation_mask = load_yolo_segmentation_labels(label_path, (720, 1280))

# Map segmentation labels to point cloud
segmented_points = np.zeros(points.shape[1], dtype=np.uint8)
for i, (u, v) in enumerate(uv.T):
    if 0 <= u < 1280 and 0 <= v < 720:
        segmented_points[i] = segmentation_mask[v, u]
        # if segmentation_mask[v, u] != 0:
        #     print(segmentation_mask[v, u])

# Assign colors to points based on segmentation labels
colors = np.zeros((points.shape[1], 3))
for i, label in enumerate(segmented_points):
    if label == 0:
        colors[i] = [0, 0, 0]  # Background (black)
    else:
        colors[i] = plt.cm.tab20(label % 20)[:3]  # Use a colormap for different classes

# Create Open3D point cloud with colors
pc_colored = o3d.geometry.PointCloud()
pc_colored.points = o3d.utility.Vector3dVector(points[:3, :].T)
pc_colored.colors = o3d.utility.Vector3dVector(colors)

o3d.io.write_point_cloud('pc.ply', pc_colored)
# Visualize the segmented point cloud
