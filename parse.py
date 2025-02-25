from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from cv_bridge import CvBridge
import cv2
import sqlite3
import os
import numpy as np
import open3d
import sensor_msgs_py.point_cloud2 as pc2

conn = sqlite3.connect('rosbag2_2025_02_14-16_24_03/rosbag2_2025_02_14-16_24_03_0.db3')
cursor = conn.cursor()

cursor.execute("SELECT timestamp, data from messages where topic_id = 2")
rows = cursor.fetchall()
# print(rows)

bridge = CvBridge()
frames = list(np.random.choice(a=len(rows), size=5))
img_timestamps = []
os.mkdir('data')
for row in [rows[i] for i in frames]:
    timestamp, data = row
    msg_type = get_message('sensor_msgs/msg/CompressedImage')
    msg = deserialize_message(data, msg_type)
    cv_image = bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='bgr8')
    img_timestamps.append(timestamp)
    os.mkdir(f'data/{timestamp}')
    cv2.imwrite(f'data/{timestamp}/image_{timestamp}.png', cv_image)

cursor.execute("SELECT timestamp from messages where topic_id = 1")
rows = cursor.fetchall()
pc_timestamps = np.array([row for row in rows])

def pointcloud2_to_array(cloud_msg):
    # Extract the raw data from the PointCloud2 message
    cloud_data = np.frombuffer(cloud_msg.data, dtype=np.uint8)

    # Get the number of points
    num_points = cloud_msg.width * cloud_msg.height

    # Get the point step (size of each point in bytes)
    point_step = cloud_msg.point_step

    # Create a structured dtype based on the fields in the PointCloud2 message
    dtype_list = []
    for field in cloud_msg.fields:
        dtype_list.append((field.name, np.float32))  # Assuming all fields are float32

    # Create a structured array
    cloud_array = np.zeros(num_points, dtype=dtype_list)

    # Iterate over each point and extract the data
    for i in range(num_points):
        offset = i * point_step
        for field in cloud_msg.fields:
            start = offset + field.offset
            end = start + 4  # Assuming each field is 4 bytes (float32)
            cloud_array[field.name][i] = np.frombuffer(cloud_data[start:end], dtype=np.float32)

    return cloud_array

for img_timestamp in img_timestamps:
    closet_timestamp = pc_timestamps[np.abs(pc_timestamps - img_timestamp).argmin()]
    cursor.execute(f"SELECT data from messages where topic_id = 1 and timestamp = {closet_timestamp[0]}")
    closest_pc = cursor.fetchall()
    msg_type = get_message('sensor_msgs/msg/PointCloud2')
    point_cloud_msg = deserialize_message(closest_pc[0][0], msg_type)
    # pc = pointcloud2_to_array(point_cloud_msg)
    pc = pc2.read_points(point_cloud_msg, field_names=("x", "y", "z"))
    points = np.stack((pc['x'], pc['y'], pc['z']), axis=-1).reshape(-1, 3)
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(points)
    open3d.io.write_point_cloud(f'data/{img_timestamp}/point_cloud_{closet_timestamp[0]}.pcd', pcd)
    print((img_timestamp - closet_timestamp[0]) / 1e9)
    
conn.close()
