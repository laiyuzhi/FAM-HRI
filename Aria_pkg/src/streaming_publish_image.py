#!/usr/bin/env python
import rospy
from std_msgs.msg import String, Header
from sensor_msgs.msg import Image as Image_ros
from sensor_msgs.msg import Imu
from cv_bridge import CvBridge

from typing import Sequence
import argparse
import sys
import time
import aria.sdk as aria
import redis
import cv2
import numpy as np
from common import quit_keypress, update_iptables
import csv
from projectaria_tools.core import calibration
from projectaria_tools.core.sensor_data import (
    ImageDataRecord,
    MotionData,
)
import os
from PIL import Image
from projectaria_eyetracking.projectaria_eyetracking import real_time_inference
import matplotlib.pyplot as plt 
import shutil

saving_state = {"saving": 0}
save_folder = "/home/ylai/aria_data/save/mav0/"
camera_id_map = {0: "cam0", 1: "cam1", 2: "rgbcam", 3: "eyetrack"}
imu_id_map = {0:"imu0", 1:"imu1"}

eyetrack_folder = "/home/ylai/aria_data/save/mav0/eyetrack"

if os.path.exists(eyetrack_folder):
    os.remove(eyetrack_folder)

if not os.path.exists(eyetrack_folder):
    os.makedirs(eyetrack_folder)


#write camera header
for camera_index in camera_id_map.keys():

    save_path = os.path.join(save_folder, camera_id_map[camera_index])

    print(save_path)
    if os.path.exists(save_path):  
        shutil.rmtree(save_path)

    if not os.path.exists(save_path):  
        os.makedirs(save_path)
)

eye_track_file_path = os.path.join(eyetrack_folder, "general_eye_gaze.csv")
# Initialize the CSV file with headers (if it doesn't already exist)
with open(eye_track_file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    # Write the header row
    writer.writerow([
        "tracking_timestamp_ns", "yaw_rads_cpf", "pitch_rads_cpf", "depth_m_str",
        "yaw_low_rads_cpf", "pitch_low_rads_cpf", "yaw_high_rads_cpf", "pitch_high_rads_cpf"
    ])

#write imu header
for imu_idx in imu_id_map.keys():

    save_path = os.path.join(save_folder, imu_id_map[imu_idx])

    if os.path.exists(save_path):  
        shutil.rmtree(save_path)

    if not os.path.exists(save_path):  
        os.makedirs(save_path)
    csv_file_path = os.path.join(save_path,"data.csv")

    csv_header = ['#timestamp [ns]', 'w_RS_S_x [rad s^-1]', 'w_RS_S_y [rad s^-1]', 'w_RS_S_z [rad s^-1]', 'a_RS_S_x [m s^-2]', 'a_RS_S_y [m s^-2]', 'a_RS_S_z [m s^-2]']
    if not os.path.exists(csv_file_path):
        with open(csv_file_path, mode="w", newline="") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(csv_header) 

def save_eye_gaze_result_to_csv(result, file_path):
    with open(file_path, mode='a', newline='') as file:  # Open the file in append mode
        #print(result)
        writer = csv.writer(file)
        writer.writerow(result) 

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--update_iptables",
        default=False,
        action="store_true",
        help="Update iptables to enable receiving the data stream, only for Linux.",
    )
    return parser.parse_args()

def control_command_callback(msg):
    global saving_state
    if msg.data == "start" and saving_state["saving"] == 0:
        saving_state["saving"] = 1
    if msg.data == "finish" and saving_state["saving"] == 1:
        saving_state["saving"] = 2
    else:
        rospy.loginfo("Subscriber is running (non-blocking).")
def main():
    global saving_state
    # slam camera calibration
    # ros publisher init
    rospy.init_node('aria_publisher', anonymous=True)
    

    inference_model, device_calibration, rgb_camera_calibration, rgb_stream_label, rgb_linear_camera_calibration, slam_camera_calibration, slam_linear_camera_calibration = real_time_inference.eyetracking_initialization()
    save = 0
    args = parse_args()
    if args.update_iptables and sys.platform.startswith("linux"):
        update_iptables()

    #  Optional: Set SDK's log level to Trace or Debug for more verbose logs. Defaults to Info
    aria.set_log_level(aria.Level.Info)

    # 1. Create StreamingClient instance
    streaming_client = aria.StreamingClient()
    device_client = aria.DeviceClient()
    client_config = aria.DeviceClientConfig()
    device_client.set_client_config(client_config)

    #  2. Configure subscription to listen to Aria's RGB and SLAM streams.
    # @see StreamingDataType for the other data types
    config = streaming_client.subscription_config
    config.subscriber_data_type = (
        aria.StreamingDataType.Rgb | aria.StreamingDataType.Slam | aria.StreamingDataType.EyeTrack | aria.StreamingDataType.Imu
    )

    # A shorter queue size may be useful if the processing callback is always slow and you wish to process more recent data
    # For visualizing the images, we only need the most recent frame so set the queue size to 1
    config.message_queue_size[aria.StreamingDataType.Rgb] = 1
    config.message_queue_size[aria.StreamingDataType.Slam] = 1
    config.message_queue_size[aria.StreamingDataType.EyeTrack] = 1
    config.message_queue_size[aria.StreamingDataType.Imu] = 1
   
    
    # Set the security options
    # @note we need to specify the use of ephemeral certs as this sample app assumes
    # aria-cli was started using the --use-ephemeral-certs flag
    options = aria.StreamingSecurityOptions()
    options.use_ephemeral_certs = True
    config.security_options = options
    streaming_client.subscription_config = config

    # 3. Create and attach observer
    class StreamingClientObserver:
        def __init__(self, slam_camera_calibration, slam_linear_camera_calibration):
            self.images = {}
            self.timestamp = 0
            self.timestamps = {}
            self.save_flag = False
            self.sample = []
            self.pub = rospy.Publisher('/imu0', Imu, queue_size=1)
            self.pub_img = rospy.Publisher('/cam0/image_raw', Image_ros, queue_size=1)
            self.imu_data = Imu()
            self.header = Header()
            self.header_img = Header()
            self.slam_data = Image_ros()
            self.bridge = CvBridge()
            self.slam_camera_calibration = slam_camera_calibration
            self.slam_linear_camera_calibration = slam_linear_camera_calibration

        def on_imu_received(self, samples: Sequence[MotionData], imu_idx: int) -> None:
            self.sample = samples
            for c in range(len(samples)):  
               
                if imu_idx == 1:
                    secs = samples[c].capture_timestamp_ns // 1000000000
                    nsecs = samples[c].capture_timestamp_ns % 1000000000
                    ros_time = rospy.Time(secs, nsecs)
                    self.header.stamp = ros_time
                    self.header.frame_id = "imu"
                    self.imu_data.header = self.header
                    self.imu_data.angular_velocity.x = samples[c].gyro_radsec[0]
                    self.imu_data.angular_velocity.y = samples[c].gyro_radsec[1]
                    self.imu_data.angular_velocity.z = samples[c].gyro_radsec[2]
                    self.imu_data.linear_acceleration.x = samples[c].accel_msec2[0]
                    self.imu_data.linear_acceleration.y = samples[c].accel_msec2[1]
                    self.imu_data.linear_acceleration.z = samples[c].accel_msec2[2]
                    self.pub.publish(self.imu_data)

        def on_image_received(self, image: np.array, record: ImageDataRecord):
            self.images[record.camera_id] = image
            self.timestamp = record.capture_timestamp_ns
            save_path = os.path.join(save_folder, camera_id_map[record.camera_id],'data')
            if self.save_flag:
                if not os.path.exists(save_path):  
                    os.makedirs(save_path)
                csv_file_path = os.path.join(save_folder, camera_id_map[record.camera_id],"data.csv")
                #filename_png = f"{self.timestamp}.npy"
                csv_header = ["#timestamp [ns]", "filename"]
                if not os.path.exists(csv_file_path):
                    with open(csv_file_path, mode="w", newline="") as csv_file:
                        writer = csv.writer(csv_file)
                        writer.writerow(csv_header) 

                
                with open(csv_file_path, mode="a", newline="") as csv_file:
                    writer = csv.writer(csv_file)
                    writer.writerow([self.timestamp, f"{self.timestamp}.npy"])
                filename = os.path.join(save_path, f"{self.timestamp}.npy")
                np.save(filename, image)
            if record.camera_id == 0:
                secs = self.timestamp // 1000000000
                nsecs = self.timestamp % 1000000000
                ros_time = rospy.Time(secs, nsecs)
                self.header_img.stamp = ros_time
                self.header_img.frame_id = "camera"
                undistort_image = calibration.distort_by_calibration(
                    image,
                    self.slam_linear_camera_calibration,
                    self.slam_camera_calibration,
                )
                ros_image = self.bridge.cv2_to_imgmsg(undistort_image, encoding="mono8")
                ros_image.header = self.header_img
                self.pub_img.publish(ros_image)
                    
                    #rospy.loginfo("Published image with size: {}x{} at {}".format(ros_image.width, ros_image.height, ros_image.header.seq))

    observer = StreamingClientObserver(slam_camera_calibration, slam_linear_camera_calibration)
    streaming_client.set_streaming_client_observer(observer) 
    rospy.Subscriber("/command", String, control_command_callback)
    # 4. Start listening
    print("Start listening to image data")
    streaming_client.subscribe()

    plt.ion()  # start interaction model
    fig, ax = plt.subplots()
    ax.set_title("Real-Time Eye Tracking")
    image_np = np.zeros((1408, 1408, 3), dtype=np.uint8)
    image_init = Image.fromarray(image_np)
    im = ax.imshow(image_init)

    while not rospy.is_shutdown():  
        value_mapping, eye_gaze_inference_result = real_time_inference.real_time_eyetracking(inference_model, observer.images, aria.CameraId.EyeTrack, observer.timestamp)
        image_with_gaze = real_time_inference.eye_tracking_visuliazation(device_calibration, rgb_camera_calibration, rgb_stream_label, aria.CameraId.Rgb, observer.images, value_mapping)
        if saving_state["saving"] == 1: # 1
            observer.save_flag=True # True
            if image_with_gaze is not None:
                rotated_image = np.rot90(image_with_gaze, k=-1)
                im.set_data(rotated_image)
                fig.canvas.draw()
                fig.canvas.flush_events()
                save_eye_gaze_result_to_csv(eye_gaze_inference_result, eye_track_file_path)
        
        if saving_state["saving"] == 2:
            print("Stop listening to image data")
            streaming_client.unsubscribe()
            break
    observer.save_flag=False    
    streaming_client.unsubscribe()
    print("finish")   
    
if __name__ == "__main__":
   
    main()
