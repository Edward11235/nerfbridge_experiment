import rospy
import cv2
import numpy as np
import struct
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped
import tf.transformations as tf_trans
import time
import os

def rotmat2qvec(R):
    # from nerfstudio colmap_parsing.py
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = (
        np.array(
            [
                [Rxx - Ryy - Rzz, 0, 0, 0],
                [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
                [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
                [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz],
            ]
        )
        / 3.0
    )
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec


def packet_subscriber(dataset, framerate):
    """This function receives data from packet and then
    publish the data to ros"""
    rospy.init_node('image_and_pose_publisher', anonymous=True)
    image_pub = rospy.Publisher("/image_raw", Image, queue_size=10000)
    pose_pub = rospy.Publisher("/pose", PoseStamped, queue_size=10000)

    rospy.sleep(3)
    if dataset == 'tum':
        # Path to the image files and pose file
        img_dir = 'rgb'
        pose_file = 'training_set.txt'
    elif dataset == 'replica_qvec':
        # Path to the image files and pose file
        img_dir = 'replica_frame'
        pose_file = 'training_set1.txt'
    elif dataset == 'replica':
        # Path to the image files and pose file
        img_dir = 'office0/results'
        pose_file = 'office0/traj.txt'


    # Ensure the directories and files exist
    assert os.path.exists(img_dir), f"Image directory not found at {img_dir}"
    assert os.path.exists(pose_file), f"Pose file not found at {pose_file}"

    # Read the pose file
    with open(pose_file, 'r') as f:
        poses = f.readlines()

    try:
        index = -1
        for pose_line in poses:
            index += 1
            if rospy.is_shutdown():
                break

            # Read the pose
            pose_data = pose_line.split()

            if dataset == 'tum':
                assert len(pose_data) == 8, "Unexpected pose format"
                timestamp, x, y, z, qx, qy, qz, qw = map(float, pose_data)

                # Construct image path from timestamp with 6 decimal places
                img_path = os.path.join(img_dir, f"{timestamp:.6f}.png")
                assert os.path.exists(img_path), f"No image found for timestamp {timestamp:.6f}"
            
            elif dataset == 'replica_qvec':
                assert len(pose_data) == 8, "Unexpected pose format"
                timestamp, x, y, z, qx, qy, qz, qw = map(float, pose_data)

                # Construct image path from timestamp with 6 decimal places
                img_path = os.path.join(img_dir, f"frame{int(timestamp):06d}.jpg")
                assert os.path.exists(img_path), f"No image found for frame {int(timestamp):06d}"
            elif dataset == 'replica':
                assert len(pose_data) == 16, "Unexpected pose format"
                # 4*4 matrix
                pose = np.array([float(x) for x in pose_data]).reshape(4, 4)

                # 4*4 transformation matrix to point and quaternion
                point = pose[:3, 3]
                qw, qx, qy, qz = rotmat2qvec(pose[:3, :3]).flat
                x, y, z = point.flat

                img_path = os.path.join(img_dir, f"frame{index:06d}.jpg")
                assert os.path.exists(img_path), f"No image found for frame {index:06d}"

            # Read the image file
            img = cv2.imread(img_path)
            height, width, channels = img.shape
            is_color = channels == 3

            # Create Image message
            img_msg = Image()
            img_msg.height = height
            img_msg.width = width
            img_msg.encoding = "bgr8" if is_color else "mono8"
            img_msg.is_bigendian = False
            img_msg.step = channels * width  # Full row length in bytes
            img_msg.data = np.array(img).tostring()

            # Create PoseStamped message
            pose_msg = PoseStamped()

            pose_msg.pose.position.x = x
            pose_msg.pose.position.y = y
            pose_msg.pose.position.z = z

            pose_msg.pose.orientation.x = qx
            pose_msg.pose.orientation.y = qy
            pose_msg.pose.orientation.z = qz
            pose_msg.pose.orientation.w = qw

            # Assign timestamps and publish
            time_now = rospy.Time.now()
            img_msg.header.stamp = time_now
            pose_msg.header.stamp = time_now

            image_pub.publish(img_msg)
            pose_pub.publish(pose_msg)
            
            print(img_path)

            # You may want to control the rate of publishing
            # For that, uncomment and adjust the line below
            rospy.sleep(1/framerate)
    except KeyboardInterrupt:
        rospy.signal_shutdown('User requested shutdown')

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='tum', help='tum or replica')
    parser.add_argument('--framerate', type=int, default=1, help='framerate of the dataset')
    dataset = parser.parse_args().dataset
    framerate = parser.parse_args().framerate
    if dataset not in ['tum', 'replica', 'replica_qvec']:
        raise ValueError('Invalid dataset method')
    if framerate < 0:
        raise ValueError('Framerate must be positive')
    packet_subscriber(dataset=dataset, framerate=framerate)
