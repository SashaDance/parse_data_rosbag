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

for img_timestamp in img_timestamps:
    closet_timestamp = pc_timestamps[np.abs(pc_timestamps - img_timestamp).argmin()]
    cursor.execute(f"SELECT data from messages where topic_id = 1 and timestamp = {closet_timestamp[0]}")
    closest_pc = cursor.fetchall()
    msg_type = get_message('sensor_msgs/msg/PointCloud2')
    point_cloud_msg = deserialize_message(closest_pc[0][0], msg_type)
    pc = pc2.read_points(point_cloud_msg, field_names=("x", "y", "z"))
    points = np.stack((pc['x'], pc['y'], pc['z']), axis=-1).reshape(-1, 3)
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(points)
    open3d.io.write_point_cloud(f'data/{img_timestamp}/point_cloud_{closet_timestamp[0]}.pcd', pcd)
    print((img_timestamp - closet_timestamp[0]) / 1e9)
    
conn.close()
