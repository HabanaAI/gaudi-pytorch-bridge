# Copyright (c) 2021, Habana Labs Ltd.  All rights reserved.

from pathlib import Path
import json
import inspect
import copy

import torch.utils.data
import torchvision.datasets

import habana_dataloader.habana_dl_app
from .aeon_config import get_aeon_config
from .aeon_transformers import HabanaAeonTransforms
from .aeon_manifest import generate_aeon_manifest


class HabanaDataLoader(torch.utils.data.DataLoader):
    def __init__(self, *args, **kwargs):
        keyword_args = copy.deepcopy(kwargs)
        keyword_args.update(dict(zip(inspect.getfullargspec(super(HabanaDataLoader, self).__init__).args[1:], args)))

        self.fallback_activated = False

        try:
            self._handle_vars(keyword_args)
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

        except ValueError as e:
            print(f"Failed to initialize Habana Dataloader, error: {str(e)}\nRunning with PyTorch Dataloader")
            self.fallback_activated = True
            super(HabanaDataLoader, self).__init__(*args, **kwargs)

    def __len__(self):
        if self.fallback_activated:
            return super().__len__()
        return len(self.aeon)

    def __iter__(self):
        if self.fallback_activated:
            return super().__iter__()
        return iter(self.aeon)



    def _handle_vars(self, kwargs):
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


    def _enforce_value_for_arg(self, kwargs, var_name, expected_value, allow_default=True):
        if not allow_default and kwargs.get(var_name) is None:
            raise ValueError(f"'{var_name}' is supported only as {expected_value}")
        # In case the value was not sent, it will be 'None'
        if kwargs.get(var_name) is not None and kwargs.get(var_name) != expected_value:
            raise ValueError(f"'{var_name}' is supported only as {expected_value}")
