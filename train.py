from tqdm import tqdm
from utils import normalize, BatchTransform
import numpy as np
import torch


def train(model, optimizer, scheduler, dataloader, device="cuda"):
    model.train()

    # transforms for training
    transforms = BatchTransform.Compose(
        [
            normalize,
            BatchTransform.RandomJitter(0.01, 0.05),
            BatchTransform.RandomRotate([1, 1, 1]),
            BatchTransform.RandomScale([0.9, 1.1]),
        ]
    )

    loss_acc = 0.0

    for x, y in tqdm(dataloader, leave=False, desc="train set"):
        x = x.to(device)
        y = y.to(device)
        parts_count = y.max(dim=1)[0] + 1

        x = transforms(x)
        output = model(x, parts_count, part_noise=True)
        loss = model.loss(x, output, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        loss_acc += float(loss)

    return loss_acc / len(dataloader)


def eval(model, dataloader, device="cuda"):
    model.eval()

    loss_acc = 0.0

    for x, y in tqdm(dataloader, leave=False, desc="eval set"):
        x = x.to(device)
        y = y.to(device)
        parts_count = y.max(dim=1)[0] + 1

        x = normalize(x)
        output = model(x, parts_count)
        loss = model.loss(x, output, y)

        loss_acc += float(loss)

    return loss_acc / len(dataloader)


# draw first predicted pcd from dataloader
def draw(model, dataloader, device="cuda"):
    model.eval()

    # get first batch
    x, y = next(iter(dataloader))

    x = x.to(device)
    y = y.to(device)
    parts_count = y.max(dim=1)[0] + 1

    x = normalize(x)
    output = model(x, parts_count)

    print(parts_count[0])
    print(y[0])
    print(output[0].argmax(1))


if __name__ == "__main__":
    from model import Model

    model = Model()
    x = torch.rand(6, 512, 3)
    y = torch.randint(16, size=(6, 512))
    output = model(x, torch.arange(6))
    print(output)
