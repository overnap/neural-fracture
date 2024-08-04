import torch
import torch.nn as nn
from pointnet2_utils import PointNetSetAbstraction, PointNetSetAbstractionMsg, PointNetFeaturePropagation


class MLPBlock(nn.Module):
    def __init__(self, in_channel, out_channel, activation=nn.SiLU()) -> None:
        super().__init__()

        self.conv = nn.Conv1d(in_channel, out_channel, 1)
        self.bn = nn.BatchNorm1d(out_channel)
        self.act = activation
        self.residual = (in_channel == out_channel)
    
    def forward(self, input):
        x = self.conv(input)
        x = self.bn(x)
        x = self.act(x)

        if self.residual:
            return x + input
        else:
            return x


class AttentionBlock(nn.Module):
    def __init__(self, channel, head=1):
        super().__init__()

        self.head = head
        self.norm = nn.BatchNorm1d(channel)
        self.q = nn.Conv1d(channel, channel * head, 1)
        self.k = nn.Conv1d(channel, channel * head, 1)
        self.v = nn.Conv1d(channel, channel * head, 1)
        self.out = nn.Conv1d(channel * head, channel, 1)

    def forward(self, input):
        x = self.norm(input)

        b, c, p = x.shape

        q = self.q(x).reshape(b, c, self.head, p)
        k = self.k(x).reshape(b, c, self.head, p)
        v = self.v(x).reshape(b, c, self.head, p)

        weight = torch.einsum("bchi,bchj->bhij", q, k)
        weight *= int(c) ** (-0.5)
        weight = nn.functional.softmax(weight, dim=3)

        x = torch.einsum("bhij,bchj->bchi", weight, v)
        x = x.reshape(b, c * self.head, p)
        x = self.out(x)

        return x + input


class Model(nn.Module):
    def __init__(self, device='cpu'):
        super(Model, self).__init__()
        
        self.device = device

        self.first = torch.nn.Sequential(
            MLPBlock(3+1, 32),
            AttentionBlock(32, 12),
            MLPBlock(32, 32),
            AttentionBlock(32, 12),
        )
        
        self.second = torch.nn.Sequential(
            MLPBlock(32, 64),
            AttentionBlock(64, 8),
            MLPBlock(64, 64),
            AttentionBlock(64, 8),
            MLPBlock(64, 64),
            AttentionBlock(64, 8),
        )

        self.last = torch.nn.Sequential(
            MLPBlock(64, 128),
            AttentionBlock(128, 4),
            MLPBlock(128, 128),
            AttentionBlock(128, 4),
            MLPBlock(128, 128),
            AttentionBlock(128, 4),
        )
        
        self.reduction = nn.Sequential(
                                MLPBlock(512+1, 256),
                                AttentionBlock(256),
                                MLPBlock(256, 128),
                                AttentionBlock(128, 4),
                                MLPBlock(128, 12),
                                AttentionBlock(12, 8))

    def forward(self, input, parts_count):
        x = input.permute(0, 2, 1)

        # parts count embedding (on data dimension)
        embed1 = torch.normal(torch.ones((x.shape[0], 1, x.shape[2])).to(self.device)
                              * parts_count.float().unsqueeze(1).unsqueeze(1))
        x = torch.cat((x, embed1), dim=1)

        # get similarity matrix
        x = self.first(x)
        x = self.second(x)
        x = self.last(x)

        x = torch.einsum('bci,bcj->bij', x, x)
        x = torch.sigmoid(x)

        # save similarity matrix
        sim = x.clone()

        # parts count embedding (on matrix dimension)
        embed2 = torch.normal(torch.ones((x.shape[0], 1, x.shape[2])).to(self.device)
                              * parts_count.float().unsqueeze(1).unsqueeze(1))
        x = torch.cat((sim, embed2), dim=1)

        # matrix reduction to group
        x = self.reduction(x)
        x = torch.softmax(x, dim=1).permute(0, 2, 1)

        return sim, x


if __name__ == '__main__':
    import torch
    model = Model(device='cuda').to('cuda')
    xyz = torch.rand(256, 512, 3).to('cuda')
    print(model(xyz, torch.randint(1, 12, (256,)).to('cuda')))