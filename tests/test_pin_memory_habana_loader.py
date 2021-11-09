import torch
import random
import os
import pytest
from torch.utils.data import _utils, TensorDataset, DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np

from habana_frameworks.torch.utils.library_loader import load_habana_module
load_habana_module()

def test_hpu_pin_memory():
    traindir = os.path.join('/software/lfs/data/pytorch/imagenet/ILSVRC2012', 'train')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
    train_dataset = datasets.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))
    import habana_dataloader
    data_loader = habana_dataloader.HabanaDataLoader(train_dataset, pin_memory=True, pin_memory_device='hpu')
    for i, (images, target) in enumerate(data_loader):
        #print("test Is pinned memory images", images.is_pinned(device='hpu'))
        #print("test Is pinned memory target", target.is_pinned(device='hpu'))
        target_hpu = target.to('hpu')
        images_hpu = images.to('hpu')
        target_out = target_hpu.to('cpu')
        images_out = images_hpu.to('cpu')
        assert np.allclose(images_out, images, atol=0.001, rtol=1.e-3), f"Data mismatch"
        assert np.allclose(target_out, target, atol=0.001, rtol=1.e-3), f"Data mismatch"
        if (i == 10):
           return
    print("Done....")
