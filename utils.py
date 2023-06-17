import torch
import trimesh


# Modified version of:
# https://github.com/yanx27/Pointnet_Pointnet2_pytorch/blob/master/models/pointnet2_utils.py
def pc_normalize(pc):
    centroid = torch.mean(pc, dim=0).detach()
    pc = pc - centroid
    m = (pc ** 2).sum(dim=1).sqrt().max().detach()
    pc = pc / m
    return pc, centroid, m

# load the mesh and point cloud from it
def mesh_to_pcd(mesh_path, sample_grid_size=0.025):
    mesh = trimesh.load(mesh_path)
    points = mesh.bounding_box.sample_grid(step=sample_grid_size)
    
    # only the points within the mesh are filtered
    points = points[mesh.contains(points)]

    return points


# Modified version of:
# https://github.com/POSTECH-CVLab/point-transformer/blob/10d43ab5210fc93ffa15886f2a4c6460cc308780/util/transform.py#L49

class BatchTransform:
    class Compose(object):
        def __init__(self, transforms):
            self.transforms = transforms

        def __call__(self, x):
            for t in self.transforms:
                x = t(x)
            return x

    class RandomJitter(object):
        def __init__(self, sigma=0.01, clip=0.05):
            self.sigma = sigma
            self.clip = clip

        def __call__(self, x):
            assert (self.clip > 0)
            jitter = torch.clip(self.sigma * torch.randn_like(x), -1 * self.clip, self.clip).to(x.device)
            return x + jitter

    class RandomRotate(object):
        def __init__(self, angle=[0, 0, 1]):
            self.angle = angle

        def __call__(self, x):
            angle_x = (torch.rand(1) * self.angle[0] * 2 - self.angle[0]) * torch.pi
            angle_y = (torch.rand(1) * self.angle[1] * 2 - self.angle[1]) * torch.pi
            angle_z = (torch.rand(1) * self.angle[2] * 2 - self.angle[2]) * torch.pi
            cos_x, sin_x = torch.cos(angle_x), torch.sin(angle_x)
            cos_y, sin_y = torch.cos(angle_y), torch.sin(angle_y)
            cos_z, sin_z = torch.cos(angle_z), torch.sin(angle_z)
            R_x = torch.tensor([[1, 0, 0], [0, cos_x, -sin_x], [0, sin_x, cos_x]])
            R_y = torch.tensor([[cos_y, 0, sin_y], [0, 1, 0], [-sin_y, 0, cos_y]])
            R_z = torch.tensor([[cos_z, -sin_z, 0], [sin_z, cos_z, 0], [0, 0, 1]])
            R = (R_z @ R_y @ R_x).to(x.device)
            return torch.einsum('bij,jk->bik', x, R.transpose(0, 1))

    class RandomShift(object):
        def __init__(self, shift=[0.2, 0.2, 0]):
            self.shift = shift

        def __call__(self, x):
            shift_x = torch.rand(1) * self.shift[0]*2 - self.shift[0]
            shift_y = torch.rand(1) * self.shift[1]*2 - self.shift[1]
            shift_z = torch.rand(1) * self.shift[2]*2 - self.shift[2]
            return x + torch.tensor([shift_x, shift_y, shift_z]).to(x.device)
        
    class RandomScale(object):
        def __init__(self, scale=[0.9, 1.1], anisotropic=False):
            self.scale = scale
            self.anisotropic = anisotropic

        def __call__(self, x):
            scale = (torch.rand((x.shape[0], 1, 3 if self.anisotropic else 1))
                    * (self.scale[1] - self.scale[0]) + self.scale[0]).to(x.device)
            return x * scale