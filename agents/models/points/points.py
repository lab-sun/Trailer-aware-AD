import torch
import torch.nn as nn
import numpy as np
from PointPillars.pointpillars.model.pointpillars import PillarEncoder, PillarLayer

class PointsEncoder(nn.Module):
    def __init__(self, voxel_size, point_cloud_range, device, lr, weight_decay, checkpoint_dir,max_num_points, max_voxels = (16000, 40000)):
        super(PointsEncoder, self).__init__()
        
        self.device = device
        self.pretrained_checkpoint_file = "PointPillars/pretrained/epoch_160.pth"
        self.save_checkpoint_file = f"{checkpoint_dir}/weights/points_encoder.pt"
        self.save_checkpoint_optimizer_file = f"{checkpoint_dir}/weights/optimizers/points_encoder.pt"
        
        # 初始化 PillarLayer
        self.pillar_layer = PillarLayer(voxel_size = voxel_size,
                                        point_cloud_range = point_cloud_range,
                                        max_num_points = max_num_points,
                                        max_voxels = max_voxels)
        # 初始化 PillarEncoder
        self.pillar_encoder = PillarEncoder(voxel_size=voxel_size,
                                            point_cloud_range=point_cloud_range,
                                            in_channel=9,
                                            out_channel=64)
        
        # 将3D特征图转换为1D特征向量 输入 （batch_size，C，H，W） 64 496 432
        self.block = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), #（1, 128, 248, 216）
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1), #（1, 256, 124, 108）
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1), #（1, 256, 62, 54）
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1), #（1, 256, 31, 27）
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),  # (1, 256, 1, 1)
            nn.Flatten()                    # (1, 256)
        )

        self.load_pretrianed_checkpoints_and_freeze()
        self.to(self.device)
        self.optimizer = torch.optim.Adam(lr=lr, weight_decay=weight_decay, params=self.parameters())


    def forward(self, batched_pts):
        pillars, coors_batch, npoints_per_pillar = self.pillar_layer(batched_pts)
        print(f"coors_batch_shape: {coors_batch.shape}, coors_batch: {coors_batch}")
        pillar_features = self.pillar_encoder(pillars, coors_batch, npoints_per_pillar)
        # 将特征图转换为1D特征向量
        points_feature = self.block(pillar_features)

        return points_feature
    
    def load_pretrianed_checkpoints_and_freeze(self):
        
        pretrained_checkpoint = torch.load(self.pretrained_checkpoint_file, map_location=self.device)
        pretrained_checkpoint_ = pretrained_checkpoint.get("model_state_dict", pretrained_checkpoint)  # Use "state_dict" if available, otherwise use the entire checkpoint
        load_dict = {k: v for k, v in pretrained_checkpoint_.items() if k in self.state_dict()}
        self.load_state_dict(load_dict, strict=False)
        self.to(self.device)
        # 冻结pillarencoder参数
        for name, param in self.named_parameters():
            if 'pillar' in name:
                param.requires_grad = False

        

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.save_checkpoint_file)
        torch.save(self.optimizer.state_dict(), self.save_checkpoint_optimizer_file)
        print(f"Model checkpoint saved to {self.save_checkpoint_file}")

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.save_checkpoint_file, map_location=self.device))
        self.optimizer.load_state_dict(torch.load(self.save_checkpoint_optimizer_file, map_location=self.device))
        self.optimizer.param_groups[0]['capturable'] = True