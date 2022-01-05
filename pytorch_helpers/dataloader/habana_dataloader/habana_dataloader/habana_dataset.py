# Copyright (c) 2021, Habana Labs Ltd.  All rights reserved.

from pathlib import Path
import json
import inspect
import copy

import torch.utils.data
import torchvision.datasets
from enum import Enum
import torch.hpu

class hpuDeviceType(Enum):
    synDeviceGaudi = 2
    synDeviceGaudi2 = 4

class HabanaDataLoader(torch.utils.data.DataLoader):
    def __init__(self, *args, **kwargs):
        keyword_args = copy.deepcopy(kwargs)
        keyword_args.update(dict(zip(inspect.getfullargspec(super(HabanaDataLoader, self).__init__).args[1:], args)))

        DeviceType = torch.hpu.get_device_type()

        self.fallback_activated = False
        try:
            self.eDeviceType = hpuDeviceType(DeviceType)
            print("HabanaDataLoader device type ", self.eDeviceType)

            if self.eDeviceType == hpuDeviceType.synDeviceGaudi:
                from .aeon_config import get_aeon_config
                from .aeon_transformers import HabanaAeonTransforms
                from .aeon_manifest import generate_aeon_manifest
                import habana_dataloader.habana_dl_app
                self._aeon_dl_handle_vars(keyword_args)
                if not isinstance(self.dataset, torchvision.datasets.ImageFolder):
                    raise ValueError("HabanaDataLoader supports only ImageFolder as dataset")
                torch_transforms = self.dataset.transform
                aeon_data_dir = self.dataset.root

                ht = HabanaAeonTransforms(torch_transforms)
                aeon_transform_config, is_train = ht.get_aeon_transforms()
                manifest_filename = generate_aeon_manifest(self.dataset.imgs)
                aeon_config_json = get_aeon_config(aeon_data_dir, manifest_filename, aeon_transform_config, self.batch_size, self.num_workers, is_train)
                self.aeon = habana_dataloader.habana_dl_app.HabanaAcceleratedPytorchDL(aeon_config_json,
                                                                        True, # pin_memory
                                                                        True, # use_prefetch
                                                                        False, # channels-last
                                                                        self.drop_last
                                                                        )
                print("Running with Habana aeon DataLoader")

            elif self.eDeviceType == hpuDeviceType.synDeviceGaudi2:

                from torchmedialoader.media_dataloader_mediapipe import HPUMediaPipe

                self._media_dl_handle_vars(keyword_args)
                if not isinstance(self.dataset, torchvision.datasets.ImageFolder):
                    raise ValueError(
                        "MediaDataLoader supports only ImageFolder as dataset")
                root = self.dataset.root
                torch_transforms = self.dataset.transform
                pipeline = HPUMediaPipe(a_torch_transforms=torch_transforms, a_root=root, a_batch_size=self.batch_size,
                                        a_shuffle=self.shuffle, a_drop_last=self.drop_last, a_prefetch_count=self.prefetch_factor)

                from mediapipe.plugins.pytorch.iterators import HPUGenericIterator
                self.iterator = HPUGenericIterator(
                    mediapipe=pipeline, device_id=0)

                print("Running with Habana media DataLoader")
            else:
                raise ValueError("Unsupported device")

        except (ValueError, ImportError) as e:
            print(f"Failed to initialize Habana Dataloader, error: {str(e)}\nRunning with PyTorch Dataloader")
            self.fallback_activated = True
            super(HabanaDataLoader, self).__init__(*args, **kwargs)

    def __len__(self):
        if self.fallback_activated:
            return super().__len__()
        elif self.eDeviceType == hpuDeviceType.synDeviceGaudi:
            return len(self.aeon)
        elif self.eDeviceType == hpuDeviceType.synDeviceGaudi2:
            return len(self.iterator)
        else:
            assert False, "Invalid device type"

    def __iter__(self):
        if self.fallback_activated:
            return super().__iter__()
        elif self.eDeviceType == hpuDeviceType.synDeviceGaudi:
            return iter(self.aeon)
        elif self.eDeviceType == hpuDeviceType.synDeviceGaudi2:
            return iter(self.iterator)
        else:
            assert False, "Invalid device type"

    def _aeon_dl_handle_vars(self, kwargs):
        if not kwargs.get('dataset'):
            raise ValueError("'dataset' can not be None")
        self.dataset = kwargs.get('dataset')
        self.batch_size = kwargs.get('batch_size', 1)
        self._enforce_value_for_arg(kwargs, 'shuffle', False)  # TODO: support
        self.sampler = kwargs.get('sampler', None)
        self._enforce_value_for_arg(kwargs, 'batch_sampler', None)
        self._enforce_value_for_arg(kwargs, 'num_workers', 8)  # TODO: support
        self.num_workers = kwargs.get('num_workers', 8)
        self._enforce_value_for_arg(kwargs, 'collate_fn', None)
        self._enforce_value_for_arg(kwargs, 'pin_memory', True, False)  # TODO: support
        self.drop_last = kwargs.get('drop_last', False)
        self._enforce_value_for_arg(kwargs, 'timeout', 0)
        self._enforce_value_for_arg(kwargs, 'worker_init_fn', None)
        self._enforce_value_for_arg(kwargs, 'multiprocessing_context', None)
        self._enforce_value_for_arg(kwargs, 'generator', None)
        self._enforce_value_for_arg(kwargs, 'prefetch_factor', 2)  # TODO: support
        self._enforce_value_for_arg(kwargs, 'persistent_workers', False)

    def _media_dl_handle_vars(self, kwargs):
        if not kwargs.get('dataset'):
            raise ValueError("'dataset' can not be None")
        self.dataset = kwargs.get('dataset')
        self.batch_size = kwargs.get('batch_size', 1)

        if 'shuffle' in kwargs:
            self.shuffle = kwargs.get('shuffle')
            is_shuffle_default = False
        else:
            self.shuffle = False
            is_shuffle_default = True

        sampler = kwargs.get('sampler', None)
        if sampler != None:
            print(
                "Warning: sampler is not supported by MediaDataLoader, ignoring sampler: ", sampler)
        if (is_shuffle_default == True) and isinstance(sampler, torch.utils.data.RandomSampler):
            self.shuffle = True
            print("Warning: Updated shuffle to True as sampler is RandomSampler")

        self._enforce_value_for_arg(kwargs, 'batch_sampler', None)

        num_workers = kwargs.get('num_workers', 0)
        if num_workers != 0:
            print(
                "Warning: num_workers is not supported by MediaDataLoader, ignoring num_workers: ", num_workers)

        self._enforce_value_for_arg(kwargs, 'collate_fn', None)

        # ignored pin_memory

        if 'drop_last' in kwargs:
            self.drop_last = kwargs.get('drop_last')
            if self.drop_last != True:
                print(
                    "Warning: drop_last = False is not supported by MediaDataLoader, using drop_last: True")
        else:
            print("Warning: MediaDataLoader using drop_last: True")
        self.drop_last = True

        self._enforce_value_for_arg(kwargs, 'timeout', 0)
        self._enforce_value_for_arg(kwargs, 'worker_init_fn', None)
        self._enforce_value_for_arg(kwargs, 'multiprocessing_context', None)
        self._enforce_value_for_arg(kwargs, 'generator', None)

        if 'prefetch_factor' in kwargs:
            self.prefetch_factor = kwargs.get('prefetch_factor')
            if self.prefetch_factor < 1:
                print(
                    "Warning: prefetch_factor < 1 is not supported by MediaDataLoader, updating to 1")
                self.prefetch_factor = 1
            elif self.prefetch_factor > 3:
                print("Warning: prefetch_factor updated from ",
                      self.prefetch_factor, " to 3")
                self.prefetch_factor = 3
            else:
                print("MediaDataLoader got prefetch_factor ",
                      self.prefetch_factor)
        else:
            self.prefetch_factor = 2
            print("MediaDataLoader using prefetch_factor 2")

        self._enforce_value_for_arg(kwargs, 'persistent_workers', False)

    def _enforce_value_for_arg(self, kwargs, var_name, expected_value, allow_default=True):
        if not allow_default and kwargs.get(var_name) is None:
            raise ValueError(f"'{var_name}' is supported only as {expected_value}")
        # In case the value was not sent, it will be 'None'
        if kwargs.get(var_name) is not None and kwargs.get(var_name) != expected_value:
            raise ValueError(f"'{var_name}' is supported only as {expected_value}")
