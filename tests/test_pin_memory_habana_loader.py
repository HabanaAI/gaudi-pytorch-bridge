import glob
import os

import numpy as np
import pytest
import torchvision.datasets as datasets
import torchvision.transforms as transforms


@pytest.mark.skip(reason="slow test")
def test_hpu_pin_memory():
    traindir = os.path.join("/software/lfs/data/pytorch/imagenet/ILSVRC2012", "train")
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        ),
    )
    import habana_dataloader

    data_loader = habana_dataloader.HabanaDataLoader(train_dataset, pin_memory=True, pin_memory_device="hpu")
    for i, (images, target) in enumerate(data_loader):
        # print("test Is pinned memory images", images.is_pinned(device='hpu'))
        # print("test Is pinned memory target", target.is_pinned(device='hpu'))
        target_hpu = target.to("hpu")
        images_hpu = images.to("hpu")
        target_out = target_hpu.to("cpu")
        images_out = images_hpu.to("cpu")
        assert np.allclose(images_out, images, atol=0.001, rtol=1.0e-3), "Data mismatch"
        assert np.allclose(target_out, target, atol=0.001, rtol=1.0e-3), "Data mismatch"
        if i == 10:
            return


@pytest.mark.skip(reason="slow test")
def test_hpu_habana_unet_dataloader():
    import habana_frameworks.torch.utils.experimental as htexp
    import torch

    DeviceType = htexp._get_device_type()
    device = torch.device("hpu")

    train_dir = "/software/data/pytorch/unet/01_3d/01_3d"
    x_list = list(np.array(sorted(glob.glob(train_dir + "/{}".format("*_x.npy")))))
    y_list = list(np.array(sorted(glob.glob(train_dir + "/{}".format("*_y.npy")))))
    x_in = x_list[: len(x_list) // 3]
    y_in = y_list[: len(y_list) // 3]

    kwargs = {
        "num_workers": 1,
        "benchmark": False,
        "train_batches": 0,
        "test_batches": 150,
        "dim": 3,
        "num_device": 1,
        "seed": 123,
        "meta": [],
        "patch_size": [128, 128, 128],
        "oversampling": 0.33,
        "nvol": 1,
        "augment": True,
        "set_aug_seed": 123,
        "test_pytorch_iterator": True,
    }
    import habana_dataloader

    data_loader = habana_dataloader.fetch_habana_unet_loader(x_in, y_in, 2, "train", **kwargs)
    for out_dict in data_loader:
        label = out_dict["label"].to("cpu")
        images = out_dict["image"].to("cpu")
        assert np.allclose(images, images, atol=0.001, rtol=1.0e-3), "Data mismatch"
        assert np.allclose(label, label, atol=0.001, rtol=1.0e-3), "Data mismatch"
