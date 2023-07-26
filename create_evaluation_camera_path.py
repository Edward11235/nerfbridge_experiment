import json
from converter import pose_correction, matrices_to_camera_path
import torch
import numpy as np

with open('/home/edward/Desktop/nerfbridge_all_data/transforms.json', 'r') as f:
    data = json.load(f)

# print(data['frames'][0])

for i in range(20):
    index = 8 + 10*i
    print(data['frames'][index]['transform_matrix'])
    evaluation_pose = np.array(data['frames'][index]['transform_matrix'])
    print(evaluation_pose)
    corrected_frame_1_pose = pose_correction(evaluation_pose)
    print(corrected_frame_1_pose)