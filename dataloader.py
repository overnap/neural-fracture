import os
import torch
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm
from glob import glob


class FracturedDataset(Dataset):
    def __init__(self,root = './dataset', npoints=512, split='train'):
        self.npoints = npoints
        self.root = root

        self.x = []
        self.y = []

        for folder in tqdm(glob(root + '/*/'), desc='Loading Dataset'):
            # load whole point cloud
            self.x.append(torch.load(folder + '/input.pt'))

            # skip too small objects
            if len(self.x[-1]) == 0:
                self.x.pop()
                continue

            # load labels
            output = []
            for out in glob(folder + '/output_*.pt'):
                output.append(torch.load(out))
                if self.x[-1].shape[0] != output[-1].shape[0]:
                    print(out)
            self.y.append(output)
        
        # split train and test sample if split info have not made
        if not os.path.exists(root + '/train.pt'):
            perm = torch.randperm(len(self.x))
            torch.save(perm[:len(perm)*9//10], root + '/train.pt')
            torch.save(perm[len(perm)*9//10:], root + '/test.pt')

        # get split data
        if split == 'train':
            idx = torch.load(root + '/train.pt')
        else:
            idx = torch.load(root + '/test.pt')

        self.x = [x for i, x in enumerate(self.x) if i in idx]
        self.y = [x for i, x in enumerate(self.y) if i in idx]

        self.idxmap = []
        for i in range(len(self.x)):
            for j in range(len(self.y[i])):
                self.idxmap.append([i, j])
                assert((self.y[i][j] != -1).float().sum() > 0)
                assert(not torch.isnan(self.y[i][j].max(dim=0)[0] + 1).any())
        self.idxmap = torch.tensor(self.idxmap)


    def __getitem__(self, index):
        index, yindex = self.idxmap[index]
        sample = np.arange(len(self.x[index]))[self.y[index][yindex] != -1]
        sample = np.random.choice(sample, self.npoints, replace=True)
        return self.x[index][sample, :].float(), \
                self.y[index][yindex][sample].long()

    def __len__(self):
        return self.idxmap.shape[0]



if __name__ == '__main__':
    loader_train = FracturedDataset()
    x, y = next(iter(loader_train))
    loader_train = FracturedDataset(split='eval')
    x, y = next(iter(loader_train))
    print(x.shape, y.shape)