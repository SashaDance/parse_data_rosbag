import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from rosbag2_py import SequentialReader, SequentialWriter, StorageOptions, ConverterOptions, TopicMetadata
from rclpy.serialization import deserialize_message, serialize_message
from geometry_msgs.msg import TransformStamped
from tf2_msgs.msg import TFMessage
import numpy as np
from scipy.spatial.transform import Rotation
from builtin_interfaces.msg import Time

# --- Your conversion function ---
def convert_pc(point_cloud: np.ndarray, 
               elev_range: list[float], 
               azim_range: list[float]) -> np.ndarray:
    num_azim = 480
    num_elev = point_cloud.shape[0] // num_azim
    assert point_cloud.shape[1] == 3, f'Expected pc shape to be (n, 3), got {point_cloud.shape}'    
    point_cloud = point_cloud.reshape(num_azim, num_elev, 3)
    neg_mask = np.where(point_cloud[:, :, 0] >= 0, 1, -1)
    distance_matrix = (
        np.sqrt(point_cloud[:, :, 0] ** 2 + point_cloud[:, :, 1] ** 2 + point_cloud[:, :, 2] ** 2)
        * neg_mask
    )
    elev_step = (elev_range[1] - elev_range[0]) / num_elev
    azim_step = (azim_range[1] - azim_range[0]) / num_azim
    elev = np.arange(elev_range[0], elev_range[1], elev_step)[::-1]
    azim = np.arange(azim_range[0], azim_range[1], azim_step)[::-1]
    elev_mat, azim_mat = np.meshgrid(np.deg2rad(elev), np.deg2rad(azim))
    x = (distance_matrix * np.cos(elev_mat) * np.cos(azim_mat)).flatten()
    y = (distance_matrix * np.cos(elev_mat) * np.sin(azim_mat)).flatten()
    z = (distance_matrix * np.sin(elev_mat)).flatten()
    
    return np.stack((x, y, z), axis=-1).reshape(-1, 3)

# --- Main processing class ---
class PointCloudFixer:
    def __init__(self, input_bag_path, output_bag_path):
        self.input_bag_path = input_bag_path
        self.output_bag_path = output_bag_path
        self.extrinsic = np.array([
            [-0.00090784,-0.999993,0.00334244,0.30637],
            [-0.0105471,-0.00333269,-0.999939,-0.0190283],
            [0.999944,-0.000943034,-0.010544,0.0432003],
            [0,0,0,1]
        ])
        self.static_written = False

    def create_tf_message(self, timestamp):
        """Create a TF message from the extrinsic matrix"""
        tf_msg = TFMessage()
        transform = TransformStamped()
        
        transform.header.stamp = timestamp
        transform.header.frame_id = 'camera'  # parent frame
        transform.child_frame_id = 'huawei_lidar'  # child frame
        
        transform.transform.translation.x = self.extrinsic[0, 3]
        transform.transform.translation.y = self.extrinsic[1, 3]
        transform.transform.translation.z = self.extrinsic[2, 3]
        
        rotation_matrix = self.extrinsic[:3, :3]
        rot = Rotation.from_matrix(rotation_matrix)
        quat = rot.as_quat()
        
        transform.transform.rotation.x = quat[0]
        transform.transform.rotation.y = quat[1]
        transform.transform.rotation.z = quat[2]
        transform.transform.rotation.w = quat[3]
        
        tf_msg.transforms.append(transform)
        return tf_msg

    def fix_bag(self):
        reader = SequentialReader()
        reader.open(
            StorageOptions(uri=self.input_bag_path, storage_id='sqlite3'),
            ConverterOptions(input_serialization_format='cdr', output_serialization_format='cdr')
        )

        writer = SequentialWriter()
        writer.open(
            StorageOptions(uri=self.output_bag_path, storage_id='sqlite3'),
            ConverterOptions(input_serialization_format='cdr', output_serialization_format='cdr')
        )

        # Get original topics and create output topics
        topics = reader.get_all_topics_and_types()
        for topic in topics:
            writer.create_topic(topic)

        # Ensure static tf topic exists
        tf_topic = TopicMetadata(
            name='/tf_static',
            type='tf2_msgs/msg/TFMessage',
            serialization_format='cdr'
        )
        writer.create_topic(tf_topic)

        # Read and process messages
        while reader.has_next():
            topic, data, t = reader.read_next()
            msg_time = Time(sec=t // 1_000_000_000, nanosec=t % 1_000_000_000)

            # Write static TF once at first message
            if not self.static_written:
                static_tf_msg = self.create_tf_message(msg_time)
                writer.write('/tf_static', serialize_message(static_tf_msg), t)
                self.static_written = True

            # Process point cloud messages
            if topic == '/lidar/points':
                msg = deserialize_message(data, PointCloud2)
                pc_gen = point_cloud2.read_points(
                    msg, field_names=('x', 'y', 'z', 'reflectivity', 'timestamp'), skip_nans=True
                )
                pc_np = np.array(list(pc_gen), dtype=[
                    ('x', np.float32),
                    ('y', np.float32),
                    ('z', np.float32),
                    ('reflectivity', np.uint8),
                    ('timestamp', np.float64)
                ])
                xyz_raw = np.stack((pc_np['x'], pc_np['y'], pc_np['z']), axis=-1)
                fixed_xyz = convert_pc(xyz_raw, [-12.85, 7.85], [-60, 60])
                if fixed_xyz.shape[0] != pc_np.shape[0]:
                    print(f"⚠️ Mismatch: original {pc_np.shape[0]}, fixed {fixed_xyz.shape[0]}")
                    continue
                fixed_struct = np.zeros(fixed_xyz.shape[0], dtype=pc_np.dtype)
                fixed_struct['x'] = fixed_xyz[:, 0]
                fixed_struct['y'] = fixed_xyz[:, 1]
                fixed_struct['z'] = fixed_xyz[:, 2]
                fixed_struct['reflectivity'] = pc_np['reflectivity']
                fixed_struct['timestamp'] = pc_np['timestamp']
                fixed_msg = point_cloud2.create_cloud(msg.header, msg.fields, fixed_struct)
                writer.write('/lidar/points', serialize_message(fixed_msg), t)
            else:
                writer.write(topic, data, t)

        print(f"✅ New bag written to {self.output_bag_path}")


def main():
    import sys
    rclpy.init()
    if len(sys.argv) < 3:
        print("Usage: python3 fix_bag.py <input_bag_path> <output_bag_path>")
        return

    fixer = PointCloudFixer(sys.argv[1], sys.argv[2])
    fixer.fix_bag()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
