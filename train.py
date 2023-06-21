from tqdm import tqdm
from utils import normalize, BatchTransform
import numpy as np
import torch


def train(model, optimizer, scheduler, dataloader):
    model.train()
    device = model.device

    # transforms for training
    transforms = BatchTransform.Compose([
        BatchTransform.RandomJitter(0.003, 0.01),
        normalize,
        BatchTransform.RandomRotate([1, 1, 1]),
        BatchTransform.RandomScale([0.9, 1.1]),
    ])
    
    loss_acc = 0.

    for x, y in tqdm(dataloader, leave=False, desc='train set'):
        x = x.to(device)
        y = y.to(device)
        parts_count = y.max(dim=1)[0] + 1

        x = transforms(x)
        sim, label = model(x, parts_count)

        dist = torch.cdist(sim, sim)
        rdist = torch.cdist(label, label)
        same = y.unsqueeze(1) == y.unsqueeze(2)

        loss = torch.scalar_tensor(0.).to(device)
        # low-rank loss
        loss += (same.float() - torch.einsum('bir,bjr->bij', label, label)).square().mean()
        # similarity loss (32 = sqrt(1024) - it's near-maximum value of the distance of sim vectors)
        loss += ((dist * same) + torch.clip((~same).float() * 20. - (dist * ~same), min=0.)).mean()
        # similarity loss for reducted similarity
        loss += ((rdist * same) + torch.clip((~same).float() - (rdist * ~same), min=0.)).mean()

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

        x = normalize(x)
        sim, label = model(x, parts_count)

        dist = torch.cdist(sim, sim)
        rdist = torch.cdist(label, label)
        same = y.unsqueeze(1) == y.unsqueeze(2)

        loss = torch.scalar_tensor(0.).to(device)
        # low-rank loss
        loss += (same.float() - torch.einsum('bir,bjr->bij', label, label)).square().mean()
        # similarity loss (32 = sqrt(1024) - it's near-maximum value of the distance of sim vectors)
        loss += ((dist * same) + torch.clip((~same).float() * 20. - (dist * ~same), min=0.)).mean()
        # similarity loss for reducted similarity
        loss += ((rdist * same) + torch.clip((~same).float() - (rdist * ~same), min=0.)).mean()

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

    x = normalize(x)
    sim, label = model(x, parts_count)

    same = y.unsqueeze(1) == y.unsqueeze(2)

    print(parts_count[0])
    print(same[0])
    print(sim[0])
    print(torch.einsum('ir,jr->ij', label[0], label[0]))
    print(label[0])


if __name__ == '__main__':
    from model import Model
    model = Model('cpu')
    x = torch.rand(6, 1024, 3)
    y = torch.randint(16, size=(6, 1024))
    sim, sim_reduct = model(x, torch.arange(6))
    print(sim)
    print(sim_reduct)