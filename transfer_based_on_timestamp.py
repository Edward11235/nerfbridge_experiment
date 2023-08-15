import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped
import os
import matplotlib.pyplot as plt

def plot_timestamps(timestamps, title, output_file):
    """Plot and save timestamps to a file."""
    fig, ax = plt.subplots()
    ax.plot(timestamps, [0] * len(timestamps), 'o')
    ax.set_yticks([])  # Remove y-axis ticks
    ax.set_title(title)
    ax.set_xlabel("Timestamps")
    plt.savefig(output_file)
    plt.show()  # Optionally display the plot

def packet_subscriber():
    rospy.init_node('image_and_pose_publisher', anonymous=True)
    image_pub = rospy.Publisher("/image_raw", Image, queue_size=10000)
    pose_pub = rospy.Publisher("/pose", PoseStamped, queue_size=10000)

    # Paths
    img_dir = 'rgb'
    pose_file = '/home/edward/Desktop/nerfbridge_experiment/ORB_SLAM2_output_trajectory.txt'

    assert os.path.exists(img_dir), f"Image directory not found at {img_dir}"
    assert os.path.exists(pose_file), f"Pose file not found at {pose_file}"

    with open(pose_file, 'r') as f:
        poses = f.readlines()

    pose_timestamps = []  # List to hold pose transmission timestamps

    prev_timestamp = None

    try:
        for index, pose_line in enumerate(poses):
            if rospy.is_shutdown():
                break

            loop_start_time = rospy.Time.now().to_sec()
            pose_data = pose_line.split()
            assert len(pose_data) == 8, "Unexpected pose format"
            timestamp, x, y, z, qx, qy, qz, qw = map(float, pose_data)

            img_path = os.path.join(img_dir, f"{timestamp:.6f}.png")
            assert os.path.exists(img_path), f"No image found for timestamp {timestamp:.6f}"
            img = cv2.imread(img_path)
            height, width, channels = img.shape
            is_color = channels == 3

            # Create and publish Image message
            img_msg = Image()
            img_msg.height = height
            img_msg.width = width
            img_msg.encoding = "bgr8" if is_color else "mono8"
            img_msg.is_bigendian = False
            img_msg.step = channels * width
            img_msg.data = np.array(img).tostring()
            time_now = rospy.Time.now()
            img_msg.header.stamp = time_now
            image_pub.publish(img_msg)

            # Create and publish PoseStamped message
            pose_msg = PoseStamped()
            pose_msg.pose.position.x = x
            pose_msg.pose.position.y = y
            pose_msg.pose.position.z = z
            pose_msg.pose.orientation.x = qx
            pose_msg.pose.orientation.y = qy
            pose_msg.pose.orientation.z = qz
            pose_msg.pose.orientation.w = qw
            pose_msg.header.stamp = time_now
            pose_pub.publish(pose_msg)
            pose_timestamps.append(timestamp)  # Record pose transmission timestamp

            if prev_timestamp is not None and index < len(poses) - 1:
                next_timestamp = float(poses[index + 1].split()[0])
                time_diff = next_timestamp - timestamp
                elapsed_time = rospy.Time.now().to_sec() - loop_start_time
                sleep_duration = max(time_diff - elapsed_time, 0)
                rospy.sleep(sleep_duration)

            prev_timestamp = timestamp

    except KeyboardInterrupt:
        rospy.signal_shutdown('User requested shutdown')
    # finally:
        # Plot the transmission timestamps for pose
        # plot_timestamps(pose_timestamps, "Pose Transmission Timestamps", "pose_transmission_timestamp.png")

if __name__ == "__main__":
    packet_subscriber()
