import torch
import torch.nn as nn
from pointnet2_utils import PointNetSetAbstraction, PointNetSetAbstractionMsg, PointNetFeaturePropagation


# Modified version of:
# https://github.com/yanx27/Pointnet_Pointnet2_pytorch/blob/master/models/pointnet2_sem_seg_msg.py
class Model(nn.Module):
    def __init__(self, device='cpu'):
        super(Model, self).__init__()
        
        self.device = device

        self.sa1 = PointNetSetAbstractionMsg(256, [0.1, 0.2, 0.4], [32, 64, 128], 3, [[32, 32, 64], [64, 64, 128], [64, 96, 128]])
        self.sa2 = PointNetSetAbstractionMsg(64, [0.4,0.8], [64, 128], 128+128+64, [[128, 128, 256], [128, 192, 256]])
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=512 + 3, mlp=[256, 512, 1024], group_all=True)
        self.fp3 = PointNetFeaturePropagation(in_channel=1024+256+256, mlp=[256, 256])
        self.fp2 = PointNetFeaturePropagation(in_channel=256+128+128+64, mlp=[256, 128])
        self.fp1 = PointNetFeaturePropagation(in_channel=128+4, mlp=[128, 128, 128])
        
        self.reduction = nn.Sequential(
                                nn.Conv1d(512, 512, 1),
                                nn.BatchNorm1d(512),
                                nn.ReLU(),
                                nn.Conv1d(512, 256, 1),
                                nn.BatchNorm1d(256),
                                nn.ReLU(),
                                nn.Conv1d(256, 128, 1),
                                nn.BatchNorm1d(128),
                                nn.ReLU(),
                                nn.Conv1d(128, 64, 1)).to(device)

    def forward(self, xyz, parts_count):
        xyz = xyz.permute(0, 2, 1)

        l0_points = xyz
        l0_xyz = xyz[:,:3,:]

        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)

        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        # parts count embedding
        parts_embedding = parts_count.float().unsqueeze(1).unsqueeze(1).repeat(1, 1, l0_points.shape[2])
        l0_points = self.fp1(l0_xyz, l1_xyz, torch.cat((l0_points, parts_embedding), dim=1), l1_points)
        
        x = l0_points
        x = torch.einsum('bci,bcj->bij', x, x) # inner product; get pointwise unnormalized similarity (-inf, inf)
        x = torch.sigmoid(x) # map inner product (unnormalized similarity) to probability (0, 1)
        sim = x.clone() # original similarity matrix

        x = self.reduction(x)
        mask = torch.zeros((x.shape[0], 101)).to(self.device)
        mask[(torch.arange(x.shape[0]), parts_count)] = 1
        mask = (1 - mask.cumsum(dim=1)[:, :x.shape[1]]).repeat(x.shape[2], 1, 1).permute(1, 2, 0)
        x = (torch.exp(x) * mask).permute(0, 2, 1) # masking parts
        x = x / (x.sum(dim=-1, keepdim=True) + 1e-5) # reducted similarity matrix, same as label probability
        return sim, x


if __name__ == '__main__':
    import torch
    model = Model()
    xyz = torch.rand(2, 1024, 3)
    print(model(xyz, torch.randint(1, 12, (2,))))