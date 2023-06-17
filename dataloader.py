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
            self.y.append(output)
        
        self.per_pcd = len(self.y[0])

        # compensate for data with fewer pieces than `per_pcd`
        for i, x in enumerate(self.y):
            if len(x) < self.per_pcd:
                sample = np.random.choice(len(x), self.per_pcd - len(x), replace=True)
                for j in sample:
                    self.y[i].append(x[j])
            elif len(x) > self.per_pcd:
                self.y[i] = self.y[i][:self.per_pcd]
                

        self.x = self.x[:len(self.x)]
        self.y = self.y[:len(self.y)]
        
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


    def __getitem__(self, index):
        sample = np.arange(len(self.x[index//self.per_pcd]))[self.y[index//self.per_pcd][index%self.per_pcd] != -1]
        sample = np.random.choice(sample, self.npoints, replace=True)
        return self.x[index//self.per_pcd][sample, :].float(), \
                self.y[index//self.per_pcd][index%self.per_pcd][sample].long()

    def __len__(self):
        return len(self.x) * self.per_pcd



if __name__ == '__main__':
    loader_train = FracturedDataset()
    x, y = next(iter(loader_train))
    print(x.shape, y.shape)