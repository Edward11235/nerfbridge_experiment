import rospy
import cv2
import numpy as np
import struct
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped
import tf.transformations as tf_trans
import time
import os

def packet_subscriber():
    """This function receives data from packet and then
    publish the data to ros"""
    rospy.init_node('image_and_pose_publisher', anonymous=True)
    image_pub = rospy.Publisher("/image_raw", Image, queue_size=10000)
    pose_pub = rospy.Publisher("/pose", PoseStamped, queue_size=10000)

    # Path to the image files and pose file
    img_dir = 'rgb'
    pose_file = 'ORB_SLAM2_output_trajectory.txt'

    # Ensure the directories and files exist
    assert os.path.exists(img_dir), f"Image directory not found at {img_dir}"
    assert os.path.exists(pose_file), f"Pose file not found at {pose_file}"

    # Read the pose file
    with open(pose_file, 'r') as f:
        poses = f.readlines()

    try:
        for pose_line in poses:
            if rospy.is_shutdown():
                break

            # Read the pose
            pose_data = pose_line.split()
            assert len(pose_data) == 8, "Unexpected pose format"

            timestamp, x, y, z, qx, qy, qz, qw = map(float, pose_data)

            # Construct image path from timestamp with 6 decimal places
            img_path = os.path.join(img_dir, f"{timestamp:.6f}.png")
            assert os.path.exists(img_path), f"No image found for timestamp {timestamp:.6f}"

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

            print(f"{timestamp:.6f}.png")

            # You may want to control the rate of publishing
            # For that, uncomment and adjust the line below
            rospy.sleep(10)
    except KeyboardInterrupt:
        rospy.signal_shutdown('User requested shutdown')

if __name__ == "__main__":
    packet_subscriber()
