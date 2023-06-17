
from tqdm import tqdm
from utils import pc_normalize
from dataloader import FracturedDataset
from model import Model
from train import train, eval, draw
from torch.utils.tensorboard import SummaryWriter
import torch
import open3d as o3d


EPOCH = 100
LAST_EPOCH = -1 # set -1 if you want from scratch

def main():
    # torch.autograd.set_detect_anomaly(True)
    model = Model('cuda').cuda()
    # model = torch.nn.DataParallel(model).cuda()
    # model.device = 'cuda'

    loader_train = torch.utils.data.DataLoader(
        dataset=FracturedDataset(),
        batch_size=64,
        num_workers=8,
        drop_last=True,
        shuffle=True,
        pin_memory=True)
    loader_eval = torch.utils.data.DataLoader(
        dataset=FracturedDataset(split='test'),
        batch_size=64,
        num_workers=8,
        drop_last=True,
        shuffle=True,
        pin_memory=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCH)

    if LAST_EPOCH != -1:
        model.load_state_dict(torch.load(f'./params/model_{LAST_EPOCH}.pt'))
        optimizer.load_state_dict(torch.load(f'./params/optimizer_{LAST_EPOCH}.pt'))
        scheduler.load_state_dict(torch.load(f'./params/scheduler_{LAST_EPOCH}.pt'))

    # for tensorboard
    writer = SummaryWriter()

    # for visualize
    SHOW = True

    # main training loop
    for epoch in tqdm(range(LAST_EPOCH + 1, EPOCH), desc="Training..."):
        loss_train = train(model, optimizer, scheduler, loader_train)
        writer.add_scalar('loss/train', loss_train, epoch)

        torch.save(model.state_dict(), f'./params/model_{epoch}.pt')
        torch.save(optimizer.state_dict(), f'./params/optimizer_{epoch}.pt')
        torch.save(scheduler.state_dict(), f'./params/scheduler_{epoch}.pt')

        with torch.no_grad():
            loss_eval = eval(model, loader_eval)
            writer.add_scalar('loss/eval', loss_eval, epoch)

            if SHOW:
                draw(model, loader_eval)

    writer.close()


if __name__ == '__main__':
    main()