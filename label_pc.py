import numpy as np
from typing import Tuple
import open3d as o3d
import cv2
import matplotlib.pyplot as plt

# Projection matrix and extrinsic matrix
proj_matrix = np.array([
    [954.398, 0, 628.246, 0],
    [0, 954.398, 354.768, 0],
    [0, 0, 1, 0],
])
extrinsic = np.array([
    [0.00962721,-0.99995,0.00276538,0.23],
    [-0.0137882,-0.00289799,-0.999901,0.00676965],
    [0.999859,0.0095882,-0.0138154,0.0268142],
    [0,0,0,1]
])

img_path = 'data/report_labelled_samples/1740145651629282926/image_1740145651629282926.png'
pc_path = 'data/report_labelled_samples/1740145651629282926/point_cloud_1740145651663450240.pcd'
label_path = 'data/report_labelled_samples/image_1740145651629282926.txt'



def project_points_to_camera(
    points: np.ndarray, proj_matrix: np.ndarray, cam_res: Tuple[int, int]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:  # Return original valid indices
    if points.shape[0] == 3:
        points = np.vstack((points, np.ones((1, points.shape[1]))))
    in_image = points[2, :] > 0  # Initial filter for points in front of the camera
    depths = points[2, in_image]
    uvw = np.dot(proj_matrix, points[:, in_image])
    uv = uvw[:2, :]
    w = uvw[2, :]
    uv[0, :] /= w
    uv[1, :] /= w
    valid = (uv[0, :] >= 0) & (uv[0, :] < cam_res[0]) & (uv[1, :] >= 0) & (uv[1, :] < cam_res[1])
    uv_valid = uv[:, valid].astype(int)
    depths_valid = depths[valid]
    # Get indices of original points that are valid
    original_indices = np.where(in_image)[0][valid]
    return uv_valid, depths_valid, original_indices

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
uv, depths, valid_indices = project_points_to_camera(points_cam, proj_matrix, (1280, 720))

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
colors = np.zeros((points.shape[1], 3))  # All points start as black

classes_colors = [(1, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0), (1, 1, 0)]
# Assign colors only to valid points using original indices
for i, (u, v) in enumerate(uv.T):
    if 0 <= u < 1280 and 0 <= v < 720:
        label = segmentation_mask[v, u]
        if label != 0:
            # Use class ID to choose a color from a colormap
            colors[valid_indices[i]] = np.array(classes_colors[label])  # Example: tab20 colormap
            #print(label, classes_colors[label])

# Create Open3D point cloud with colors
pc_colored = o3d.geometry.PointCloud()
pc_colored.points = o3d.utility.Vector3dVector(points[:3, :].T)
pc_colored.colors = o3d.utility.Vector3dVector(colors)

o3d.io.write_point_cloud('pc.ply', pc_colored)
# Visualize the segmented point cloud
o3d.visualization.draw_plotly([pc_colored])