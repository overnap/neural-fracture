from tqdm import tqdm
from utils import pc_normalize, BatchTransform
import numpy as np
import open3d as o3d
import torch


def train(model, optimizer, scheduler, dataloader):
    model.train()
    device = model.device

    # transforms for training
    transforms = BatchTransform.Compose([
        BatchTransform.RandomRotate([0.1, 0.1, 0.1]),
        BatchTransform.RandomScale([0.9, 1.1]),
        BatchTransform.RandomShift([0.05, 0.05, 0.05]),
        BatchTransform.RandomJitter(0.002, 0.005)
    ])
    
    loss_acc = 0.

    for x, y in tqdm(dataloader, leave=False, desc='train set'):
        x = x.to(device)
        y = y.to(device)
        parts_count = y.max(dim=1)[0] + 1

        x, _, _ = pc_normalize(x)
        x = transforms(x)
        sim, label = model(x, parts_count)
        
        # sim[y == -1, :] *= 0 # mask belonging nowhere
        dist = torch.cdist(sim, sim)
        same = y.unsqueeze(1) == y.unsqueeze(2)

        loss = torch.scalar_tensor(0.).to(device)
        # low-rank loss
        loss += (same.float() - torch.einsum('bir,bjr->bij', label, label)).square().mean()
        # similarity loss
        loss += ((dist * same) + torch.clip((~same).float() * 20. - (dist * ~same), 0.)).mean()

        loss_acc += float(loss)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

    return loss_acc / len(dataloader)


def eval(model, dataloader):
    model.eval()
    device = model.device

    loss_acc = 0.
    
    for x, y in tqdm(dataloader, leave=False, desc='eval set'):
        x = x.to(device)
        y = y.to(device)
        parts_count = y.max(dim=1)[0] + 1

        x, _, _ = pc_normalize(x)
        sim, label = model(x, parts_count)

        # sim[y == -1, :] *= 0 # mask belonging nowhere
        dist = torch.cdist(sim, sim)
        same = y.unsqueeze(1) == y.unsqueeze(2)

        loss = torch.scalar_tensor(0.).to(device)
        # low-rank loss
        loss += (same.float() - torch.einsum('bir,bjr->bij', label, label)).square().mean()
        # similarity loss
        loss += ((dist * same) + torch.clip((~same).float() * 20. - (dist * ~same), 0.)).mean()

        loss_acc += float(loss)

    return loss_acc / len(dataloader)


# draw first predicted pcd from dataloader
def draw(model, dataloader):
    model.eval()
    device = model.device

    # get first batch
    x, y = next(iter(dataloader))

    x = x.to(device)
    y = y.to(device)
    parts_count = y.max(dim=1)[0] + 1

    x, x_mean, x_std = pc_normalize(x)
    sim, label = model(x, parts_count)

    same = y.unsqueeze(1) == y.unsqueeze(2)
    print(parts_count[0])
    print(same[0])

    # get arbitrarily sample and apply inverse of normalize
    x = (x[0] * x_std + x_mean).cpu()
    y = y[0]
    y[y == -1] = y.max() + 1
    y = y.cpu()
    sim = sim[0].cpu()
    label = label[0].cpu()

    print(sim)
    print(torch.einsum('ir,jr->ij', label, label))
    
    # # make adjacency matrix "hard"
    # sim[sim < 1] = 0
    # sim[sim > 0] = 1
    # sim = sim.long()

    # # assign class number for arbitrarily order
    # pred = torch.ones(dataloader.dataset.npoints) * -1
    # for i in range(len(pred)):
    #     if pred[i] == -1:
    #         label = pred.max() + 1
    #         for j in range(len(pred)):
    #             if sim[i, j] == 1:
    #                 pred[j] = label
    
    # # interpolate Red to Blue
    # color = (pred/pred.max()).unsqueeze(1)
    # color = (1-color) * torch.tensor([237, 28, 36]) + color * torch.tensor([23, 23, 244])

    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(x.numpy())
    # pcd.colors = o3d.utility.Vector3dVector(color.numpy())

    # o3d.visualization.draw_geometries([pcd])


if __name__ == '__main__':
    from model import Model
    model = Model('cpu')
    x = torch.rand(6, 512, 3)
    y = torch.randint(16, size=(6, 512))
    sim, sim_reduct = model(x, torch.arange(6))

    sim[y == -1, :] *= 0
    dist = torch.cdist(sim, sim)
    same = y.unsqueeze(1) == y.unsqueeze(2)

    loss_reduction = ((sim - sim_reduct) ** 2).sum()
    loss_similarity = (dist * same).sum() - torch.min(dist * ~same, torch.ones(dist.shape) * 1.).sum()
    loss = loss_reduction + loss_similarity

    print(loss)