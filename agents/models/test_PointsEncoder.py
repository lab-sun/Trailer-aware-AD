import torch
import torch.nn as nn
import torch.nn.functional as F
from time import time
import numpy as np

from agents.models.points.points import PointsEncoder

voxel_size = [0.16, 0.16, 4]
point_cloud_range = [0, -40, -3, 70, 40, 1]
device = 'cuda'
lr = 0.001
weight_decay = 0.0
checkpoint_dir = 'agents/models/points/checkpoints'
max_num_points = 32
max_voxels = (16000, 40000)

pointencoder = PointsEncoder(voxel_size=voxel_size, point_cloud_range=point_cloud_range, device=device, lr=lr, weight_decay=weight_decay, checkpoint_dir=checkpoint_dir, max_num_points=max_num_points, max_voxels=max_voxels)

checkpoint = torch.load("PointPillars/pretrained/epoch_160.pth", map_location="cuda")
# print("Available keys in checkpoint:", checkpoint.keys())  # Inspect available keys
dict_pretrained = checkpoint.get("model_state_dict", checkpoint)  # Use "state_dict" if available, otherwise use the entire checkpoint
load_dict = {k: v for k, v in dict_pretrained.items() if k in pointencoder.state_dict()}

dict_pointsencoder = pointencoder.state_dict().copy()

pointsencoder_list = list (pointencoder.state_dict().keys() )
trained_list = list (dict_pretrained.keys()  )
print("new_state_dict size: {}  trained state_dict size: {}".format(len(pointsencoder_list),len(trained_list)) )
print("New state_dict first 14th parameters names")
print(pointsencoder_list[:14])
print("trained state_dict first 15th parameters names")
print(trained_list[:15])
print("Keys in load_dict that will be loaded into PointsEncoder:")
print(list(load_dict.keys()).__len__())

# print(type(dict_pointsencoder))
# print(type(dict_pretrained))
# print(checkpoint.keys())