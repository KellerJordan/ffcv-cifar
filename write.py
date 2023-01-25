import torch
import torchvision
from ffcv.fields import IntField, RGBImageField
from ffcv.writer import DatasetWriter

train_dset = torchvision.datasets.CIFAR10('/tmp', train=True, download=True)
datasets = {
    'train': train_dset,
    'train10k': torch.utils.data.Subset(train_dset, range(10000)),
    'test': torchvision.datasets.CIFAR10('/tmp', train=False, download=True)
}

for (name, ds) in datasets.items():
    writer = DatasetWriter(f'/tmp/cifar_{name}.beton', {
        'image': RGBImageField(),
        'label': IntField()
    })
    writer.from_indexed_dataset(ds)

