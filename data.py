import torch
import torchvision.transforms as T

from ffcv.fields import IntField, RGBImageField
from ffcv.fields.decoders import IntDecoder, SimpleRGBImageDecoder
from ffcv.loader import Loader, OrderOption
from ffcv.transforms import RandomHorizontalFlip, Cutout, \
    RandomTranslate, Convert, ToDevice, ToTensor, ToTorchImage
from ffcv.transforms.common import Squeeze

CIFAR_MEAN = [125.307, 122.961, 113.8575]
CIFAR_STD = [51.5865, 50.847, 51.255]

def create_loader(batch_size, train, gpu=0):

    device = 'cuda:%d' % gpu

    label_pipeline = [IntDecoder(), ToTensor(), ToDevice(device), Squeeze()]
    image_pipeline = [SimpleRGBImageDecoder()]

    if train:
        image_pipeline.extend([
            RandomHorizontalFlip(),
            RandomTranslate(padding=2),
            Cutout(12, tuple(map(int, CIFAR_MEAN))),
        ])
    image_pipeline.extend([
        ToTensor(),
        ToDevice(device, non_blocking=True),
        ToTorchImage(),
        Convert(torch.float16),
        T.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])

    # Create loaders
    order_opt = OrderOption.RANDOM if train else OrderOption.SEQUENTIAL
    name = 'train' if train else 'test'
    loader = Loader(f'/tmp/cifar_{name}.beton',
                    batch_size=batch_size,
                    num_workers=8,
                    order=order_opt,
                    drop_last=train,
                    pipelines={'image': image_pipeline,
                               'label': label_pipeline})
    return loader

