from tqdm import tqdm
from dataloader import FracturedDataset
from model import Model
from train import train, eval, draw
from torch.utils.tensorboard import SummaryWriter
import torch


POINT_COUNT = 256
EPOCH = 200
LAST_EPOCH = -1  # set -1 if you want from scratch
SHOW = True  # for visualization


def main():
    model = Model(point_count=POINT_COUNT, alpha=0.3).cuda()
    # model = torch.nn.DataParallel(model).cuda()

    loader_train = torch.utils.data.DataLoader(
        dataset=FracturedDataset(npoints=POINT_COUNT),
        batch_size=512,
        num_workers=8,
        drop_last=False,
        shuffle=True,
        pin_memory=True,
    )
    loader_eval = torch.utils.data.DataLoader(
        dataset=FracturedDataset(npoints=POINT_COUNT, split="test"),
        batch_size=512,
        num_workers=8,
        drop_last=False,
        shuffle=False,
        pin_memory=True,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, EPOCH // 4 + 1
    )

    if LAST_EPOCH != -1:
        model.load_state_dict(torch.load(f"./params2/model_{LAST_EPOCH}.pt"))
        optimizer.load_state_dict(torch.load(f"./params2/optimizer_{LAST_EPOCH}.pt"))
        scheduler.load_state_dict(torch.load(f"./params2/scheduler_{LAST_EPOCH}.pt"))

    # for tensorboard
    writer = SummaryWriter()

    # main training loop
    for epoch in tqdm(range(LAST_EPOCH + 1, EPOCH), desc="Training..."):
        loss_train = train(model, optimizer, scheduler, loader_train)
        writer.add_scalar("loss/train", loss_train, epoch)

        if epoch % 10 == 9:
            torch.save(model.state_dict(), f"./params2/model_{epoch}.pt")
            torch.save(optimizer.state_dict(), f"./params2/optimizer_{epoch}.pt")
            torch.save(scheduler.state_dict(), f"./params2/scheduler_{epoch}.pt")

        with torch.no_grad():
            loss_eval = eval(model, loader_eval)
            writer.add_scalar("loss/eval", loss_eval, epoch)

            if SHOW:
                draw(model, loader_eval)

    writer.close()


if __name__ == "__main__":
    main()
