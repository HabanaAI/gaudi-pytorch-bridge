# Copyright (c) 2021, Habana Labs Ltd.  All rights reserved.

from torchvision import transforms


class HabanaAeonTransforms:
    def __init__(self, torch_transforms, is_train=True):
        if not isinstance(torch_transforms, transforms.Compose):
            raise ValueError("torch_transforms should be of type torchvision.transforms")
        self.transforms = torch_transforms.transforms
        self.transforms_config = {}
        self.is_train = False
        self.is_val = False

    def _parse_transforms(self):
        for t in self.transforms:
            if (
                isinstance(t, transforms.RandomResizedCrop)
                or isinstance(t, transforms.CenterCrop)
                or isinstance(t, transforms.Resize)
            ):
                self._handle_resize_crop(t)
            elif isinstance(t, transforms.RandomHorizontalFlip):
                self._handle_random_horizontal_flip(t)
            elif isinstance(t, transforms.ToTensor):
                self._handle_to_tensor(t)
            elif isinstance(t, transforms.Normalize):
                self._handle_normalize(t)
            else:
                raise ValueError("Unsupported transform: " + str(type(t)))

    def _handle_resize_crop(self, t):
        if (
            not isinstance(t, transforms.RandomResizedCrop)
            and not isinstance(t, transforms.CenterCrop)
            and not isinstance(t, transforms.Resize)
        ):
            raise ValueError("not a Crop/Resize transform")
        if isinstance(t, transforms.CenterCrop) or isinstance(t, transforms.Resize):
            self.is_val = True
        if isinstance(t, transforms.RandomResizedCrop):
            self.is_train = True

        size = t.size
        if isinstance(t, transforms.Resize):
            h = size
            w = size
        else:
            h = size[0]
            w = size[1]
        self.transforms_config["height"] = h
        self.transforms_config["width"] = w

    def _handle_random_horizontal_flip(self, t):
        if not isinstance(t, transforms.RandomHorizontalFlip):
            raise ValueError("not a RandomHorizontalFlip transform")
        if isinstance(t, transforms.RandomHorizontalFlip):
            self.is_train = True
        if t.p != 0.5:
            raise ValueError("aeon RandomHorizontalFlip supports only probability of 0.5")
        self.transforms_config["flip_enable"] = True

    def _handle_to_tensor(self, t):
        if not isinstance(t, transforms.ToTensor):
            raise ValueError("not a ToTensor transform")

    def _handle_normalize(self, t):
        if not isinstance(t, transforms.Normalize):
            raise ValueError("not a Normalize transform")
        mean = t.mean
        std = t.std

        # AEON currently only support constans values
        if mean != [0.485, 0.456, 0.406]:
            raise ValueError("aeon Normalize supports only mean of [0.485, 0.456, 0.406]")
        if std != [0.229, 0.224, 0.225]:
            raise ValueError("aeon Normalize supports only std of [0.229, 0.224, 0.225]")

        # caffe_mode is aeon for normalize image
        self.transforms_config["caffe_mode"] = True

    def get_aeon_transforms(self):
        self._parse_transforms()
        if (self.is_train and self.is_val) or (not self.is_train and not self.is_val):
            raise ValueError("Could not determine if running on train or validation mode")
        return self.transforms_config, self.is_train
