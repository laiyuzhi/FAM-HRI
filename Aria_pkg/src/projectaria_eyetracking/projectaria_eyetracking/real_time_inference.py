import argparse
import csv
import os
import copy
import torch
import numpy as np
try:
   from inference import infer  # Try local imports first
except ImportError:
   from projectaria_eyetracking.projectaria_eyetracking.inference import infer
from projectaria_tools.core import data_provider, calibration
from projectaria_tools.core.mps import EyeGaze, get_eyegaze_point_at_depth
from projectaria_tools.core.mps.utils import get_gaze_vector_reprojection
from projectaria_tools.core.sensor_data import SensorDataType, TimeDomain
from projectaria_tools.core.stream_id import StreamId
from projectaria_tools.utils.rerun_helpers import AriaGlassesOutline
from PIL import Image, ImageDraw


import matplotlib.pyplot as plt
import numpy as np
# python model_inference_demo.py --vrs /home/ylai/aria_data/test/3.vrs
def display_with_matplotlib(img):
    
    plt.ion()  
    fig, ax = plt.subplots()
    
  
    img = np.zeros((1408, 1408, 3), dtype=np.uint8)
    im = ax.imshow(img)

   
    im.set_data(img)  
    fig.canvas.draw() 
    fig.canvas.flush_events()  

    plt.ioff()  
    plt.show()  


def eyetracking_initialization(device='cuda'):
    provider = data_provider.create_vrs_data_provider("/home/ylai/aria_data/test/3.vrs")  # calibration data, to get the transformmatrix. CAD inframaation are saved in vrs
    eye_stream_id = StreamId("211-1")
    rgb_stream_id = StreamId("214-1")
    slam_stream_id = StreamId("1201-1")
    eye_stream_label = provider.get_label_from_stream_id(eye_stream_id)
    rgb_stream_label = provider.get_label_from_stream_id(rgb_stream_id)
    slam_stream_label = provider.get_label_from_stream_id(slam_stream_id)
    device_calibration = provider.get_device_calibration()
    T_device_CPF = device_calibration.get_transform_device_cpf()
    rgb_camera_calibration = device_calibration.get_camera_calib(rgb_stream_label)
    slam_camera_calibration = device_calibration.get_camera_calib(slam_stream_label)
    model_checkpoint_path = f"{os.path.dirname(__file__)}/inference/model/pretrained_weights/social_eyes_uncertainty_v1/weights.pth"
    model_config_path = f"{os.path.dirname(__file__)}/inference/model/pretrained_weights/social_eyes_uncertainty_v1/config.yaml"
    inference_model = infer.EyeGazeInference(
        model_checkpoint_path, model_config_path, device
    )
    eye_gaze_inference_results = []
    eye_gaze_inference_results.append(
        [
            "tracking_timestamp_ns",
            "yaw_rads_cpf",
            "pitch_rads_cpf",
            "depth_m",
            "yaw_low_rads_cpf",
            "pitch_low_rads_cpf",
            "yaw_high_rads_cpf",
            "pitch_high_rads_cpf",
        ]
    )

    rgb_linear_camera_calibration = calibration.get_linear_camera_calibration(
        int(rgb_camera_calibration.get_image_size()[0]),
        int(rgb_camera_calibration.get_image_size()[1]),
        rgb_camera_calibration.get_focal_lengths()[0],
        "fisheye",
        rgb_camera_calibration.get_transform_device_camera(),
    )

    slam_linear_camera_calibration = calibration.get_linear_camera_calibration(
        int(slam_camera_calibration.get_image_size()[0]),
        int(slam_camera_calibration.get_image_size()[1]),
        slam_camera_calibration.get_focal_lengths()[0],
        "fisheye",
        slam_camera_calibration.get_transform_device_camera(),
    )
    return inference_model, device_calibration, rgb_camera_calibration, rgb_stream_label, rgb_linear_camera_calibration, slam_camera_calibration, slam_linear_camera_calibration

def draw_eye_tracking(image, gaze_point):

    draw = ImageDraw.Draw(image)

    # 
    
    x, y = gaze_point
    radius = 20 
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill="red")

    return image



def real_time_eyetracking(inference_model, images_observer, eye_tracking_camera_id, timestamp_observer, device='cuda'):
    """
    put eyetracking points on rgb images
    :param images: images[record.camera_id] from live stream
    :param rgb_camera_id: rgb id
    :param eye_tracking_camera_id: eye tracking id
    :param timestamp: eye_tracking_camera timestamp
    :param depth: depth
    :return gaze_point: gaze_point on rgb image
    """
    
    eye_gaze_inference_results = []
    eye_gaze_inference_results.append(
        [
            "tracking_timestamp_ns",
            "yaw_rads_cpf",
            "pitch_rads_cpf",
            "depth_m",
            "yaw_low_rads_cpf",
            "pitch_low_rads_cpf",
            "yaw_high_rads_cpf",
            "pitch_high_rads_cpf",
        ]
    )

    images = copy.deepcopy(images_observer)
    timestamp = timestamp_observer
    depth_m = 0.5 
    if eye_tracking_camera_id in images:
        img = torch.tensor(
                images[eye_tracking_camera_id], device=device
            )
        preds, lower, upper = inference_model.predict(img)
        preds = preds.detach().cpu().numpy()
        lower = lower.detach().cpu().numpy()
        upper = upper.detach().cpu().numpy()

        value_mapping = {
            "yaw": preds[0][0],
            "pitch": preds[0][1],
            "yaw_lower": lower[0][0],
            "pitch_lower": lower[0][1],
            "yaw_upper": upper[0][0],
            "pitch_upper": upper[0][1],
        }

        depth_m_str = ""
        eye_gaze_inference_result = [
            int(timestamp),  # ns
            value_mapping["yaw"],
            value_mapping["pitch"],
            depth_m_str,
            value_mapping["yaw_lower"],
            value_mapping["pitch_lower"],
            value_mapping["yaw_upper"],
            value_mapping["pitch_upper"],
        ]

        
        return value_mapping, eye_gaze_inference_result
    else:
        return None, None
    
def eye_tracking_visuliazation(device_calibration, rgb_camera_calibration, rgb_stream_label, rgb_camera_id, images_observer, value_mapping):    

    depth_m = 0.5     
    images = copy.deepcopy(images_observer)    
    if rgb_camera_id in images:
        if len(value_mapping) > 0:
            eye_gaze = EyeGaze
            eye_gaze.yaw = value_mapping["yaw"]
            eye_gaze.pitch = value_mapping["pitch"]
            # Compute eye_gaze vector at depth_m reprojection in the image
            gaze_projection = get_gaze_vector_reprojection(
                eye_gaze,
                rgb_stream_label,
                device_calibration,
                rgb_camera_calibration,
                depth_m,
            )
            image_np = images[rgb_camera_id]
            image = Image.fromarray(image_np)
            image_with_gaze = draw_eye_tracking(image, gaze_projection)
            
            
            return image_with_gaze
        else:
            return None
    else:
        return None