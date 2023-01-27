import random
import torch
import torchvision
from ffcv.fields import IntField, RGBImageField
from ffcv.writer import DatasetWriter

class NoisyDataset:
    def __init__(self, dset, p):
        self.dset = dset
        n_cls = max(dset.targets)+1
        xx = torch.rand(len(dset))
        mask = (xx < xx.quantile(p))
        self.targets = []
        for i in range(len(dset.targets)):
            this_label = dset.targets[i]
            other_labels = [i for i in range(n_cls) if i != this_label]
            t = random.choice(other_labels) if mask[i] else this_label
            self.targets.append(t)
    def __len__(self):
        return len(self.dset)
    def __getitem__(self, i):
        return self.dset[i][0], self.targets[i]

from PIL import Image
import matplotlib.pyplot
import numpy as np
colors = matplotlib.pyplot.rcParams['axes.prop_cycle'].by_key()['color']
class SpuriousDataset:
    def __init__(self, dset, p=0): 
        self.dset = dset
        self.p = p
        
    def __len__(self):
        return len(self.dset)
    
    def get_bg(self, k):
        if k == -1:
            c = '000000'
        else:
            c = colors[k][1:]
        vec = [c[0:2], c[2:4], c[4:6]]
        vec = [int(x, 16) for x in vec]
        img_a = torch.empty(40, 40, 3, dtype=torch.uint8)
        img_a[:] = torch.tensor(vec)[None, None]
        return img_a
        
    def __getitem__(self, i):
        img, lab = self.dset[i]
        ci = lab if (torch.rand([]) < self.p) else -1
        img_a = self.get_bg(ci)
        img_a[4:-4, 4:-4, :] = torch.tensor(np.array(img))
        return Image.fromarray(img_a.numpy()), lab

train_dset = torchvision.datasets.CIFAR10('/tmp', train=True, download=True)
test_dset = torchvision.datasets.CIFAR10('/tmp', train=False, download=True)
datasets = {
    'train': train_dset,
    'train10k': torch.utils.data.Subset(train_dset, range(10000)),
    #'train_p01': NoisyDataset(train_dset, p=0.1),
    #'train_p02': NoisyDataset(train_dset, p=0.2),
    #'train_p025': NoisyDataset(train_dset, p=0.25),
    #'train_p03': NoisyDataset(train_dset, p=0.3),
    #'train_p04': NoisyDataset(train_dset, p=0.4),
    'train_s0': SpuriousDataset(train_dset, p=0.),
    'train_s05': SpuriousDataset(train_dset, p=0.5),
    'train_s1': SpuriousDataset(train_dset, p=1.),
    'test': test_dset,
    'test_s0': SpuriousDataset(test_dset, p=0.),
    'test_s1': SpuriousDataset(test_dset, p=1.),
}

for (name, ds) in datasets.items():
    writer = DatasetWriter(f'/tmp/cifar_{name}.beton', {
        'image': RGBImageField(),
        'label': IntField()
    })
    writer.from_indexed_dataset(ds)

