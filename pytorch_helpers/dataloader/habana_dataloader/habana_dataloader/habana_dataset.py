# Copyright (c) 2021, Habana Labs Ltd.  All rights reserved.

from pathlib import Path
import json
import inspect
import copy
import os

import torch.utils.data
import torchvision.datasets
from enum import Enum
import habana_frameworks.torch.utils.experimental as htexp

from .aeon_config import get_aeon_config
from .aeon_ssd_configurator import AeonSSDConfigurator
from .aeon_manifest import generate_aeon_manifest
import torch.distributed as dist

def _is_distributed():
    return dist.is_available() and dist.is_initialized()

def _get_world_size():
    if _is_distributed():
        return dist.get_world_size()
    else:
        return 1

def _get_rank():
    if _is_distributed():
        return dist.get_rank()
    else:
        return 0

def isGaudi(device):
    return (device == htexp.synDeviceType.synDeviceGaudi) or (device == htexp.synDeviceType.synDeviceGaudiM)

def isGaudi2(device):
    return device == htexp.synDeviceType.synDeviceGaudi2

def isGreco(device):
    return device == htexp.synDeviceType.synDeviceGreco

def deviceStr(device):
    if isGaudi(device):
        return "gaudi"
    elif isGaudi2(device):
        return "gaudi2"
    elif isGreco(device):
        return "greco"
    else:
        raise ValueError("Unsupported device")

class SSDDataLoader(torch.utils.data.DataLoader):
    def __init__(self, *args, **kwargs):
        import habana_dataloader.habana_dl_app
        dataset = kwargs.get('dataset', args[0])
        self.batch_size = kwargs.get('batch_size')
        num_workers = kwargs.get('num_workers')
        shuffle = kwargs.get('shuffle')
        manifest = kwargs.get('manifest', "manifest.cfg")
        drop_last = kwargs.get('drop_last', False)
        self.encoder = None
        distributed = kwargs.get('sampler', None) != None
        channels_last = kwargs.get('channels_last', False)

        self.configurator = AeonSSDConfigurator(dataset, self.batch_size, num_workers, shuffle, channels_last, manifest, distributed=distributed)
        aeon_config = self.configurator.get_config()

        self.aeon = habana_dataloader.habana_dl_app.HabanaAcceleratedPytorchDL.create(aeon_config,
                                                                                      True, # pin_memory
                                                                                      True, # use_prefetch
                                                                                      channels_last, # channels-last
                                                                                      drop_last
                                                                                      )

    def __iter__(self):
        self.iter = iter(self.aeon)
        return self
    def __len__(self):
        return len(self.aeon)
    def __next__(self):
        img, img_id, img_size, bbox, label = next(self.iter)
        if not self.configurator.is_train():
            img_size = torch.split(img_size, 1, dim=1)
            img_size = (img_size[1].squeeze(dim=1), img_size[0].squeeze(dim=1))

        if self.encoder:
            bbox_out = torch.empty((self.batch_size, 8732, 4), dtype = bbox.dtype)
            label_out = torch.empty((self.batch_size, 8732), dtype = label.dtype)
            for i, (b,l) in enumerate(zip(bbox, label)):
                indexes = l.nonzero()
                if indexes.nelement() == 0:
                    #WA for empty label
                    l = torch.zeros((1), dtype = label.dtype)
                    b = torch.zeros((1,4), dtype = bbox.dtype)
                    b[:, 2:] = 1
                else:
                    l = l[indexes].squeeze(dim=1)
                    b = b[indexes].squeeze(dim=1)
                b, l = self.encoder.encode(b, l)
                bbox_out[i] = b
                label_out[i] = l
        else: #aeon encoder
            bbox_out = bbox
            label_out = label

        return img.contiguous(), img_id, img_size, bbox_out, label_out

class SSDMediaDataLoader(torch.utils.data.DataLoader):
    def __init__(self, *args, **kwargs):
        dataset = kwargs.get('dataset', args[0])
        transform = dataset.transform
        self.is_train =  not transform.val
        self._media_ssd_dl_handle_vars(kwargs)
        root = dataset.img_folder
        annotate_file = dataset.annotate_file
        num_instances = _get_world_size()
        instance_id = _get_rank()

        DeviceType = htexp._get_device_type()
        if isGaudi2(DeviceType):
            media_device_type = "gaudi2"
        elif isGreco(DeviceType):
            media_device_type = "greco"
        else:
            raise ValueError("Unsupported device")

        from habana_frameworks.medialoaders.torch.media_dataloader_mediapipe import HPUMediaPipe
        pipeline = HPUMediaPipe(a_torch_transforms=transform, a_root=root, a_annotation_file=annotate_file, a_batch_size=self.batch_size,
                                a_shuffle=self.shuffle, a_drop_last=self.drop_last, a_prefetch_count=self.prefetch_factor,
                                a_num_instances=num_instances, a_instance_id=instance_id, a_model_ssd=True, a_device=media_device_type)

        from habana_frameworks.mediapipe.plugins.iterator_pytorch import HPUSsdPytorchIterator
        self.iterator = HPUSsdPytorchIterator(mediapipe=pipeline)
        print(f"Running with Habana media DataLoader with num_instances = {num_instances}, instance_id = {instance_id}.")


    def __iter__(self):
        return iter(self.iterator)

    def __len__(self):
        return len(self.iterator)

    def _media_ssd_dl_handle_vars(self, kwargs):

        self.batch_size = kwargs.get('batch_size')
        self.shuffle = kwargs.get('shuffle')

        sampler = kwargs.get('sampler', None)
        if self.shuffle == False:
            if isinstance(sampler, torch.utils.data.distributed.DistributedSampler) and (sampler.shuffle == True):
                self.shuffle = True
                print("Warning: Updated shuffle to True as sampler is DistributedSampler with shuffle True")
        if sampler != None:
            print("Warning: sampler is not supported by MediaDataLoader, ignoring sampler: ", sampler)

        self._enforce_value_for_arg(kwargs, 'batch_sampler', None)

        num_workers = kwargs.get('num_workers', 0)
        if num_workers != 0:
            print("Warning: num_workers is not supported by MediaDataLoader, ignoring num_workers: ", num_workers)

        self._enforce_value_for_arg(kwargs, 'collate_fn', None)

        # ignored pin_memory

        if 'drop_last' in kwargs:
            self.drop_last = kwargs.get('drop_last')
            if (self.drop_last == False) and (self.is_train == True):
                print("Warning: MediaDataLoader got drop_last: False, round up of last batch will be done for train")
            else:
                print("MediaDataLoader got drop_last: ", self.drop_last)
        else:
            if self.is_train == True:
                print("Warning: MediaDataLoader using drop_last: False, round up of last batch will be done for train")
            else:
                print("MediaDataLoader using drop_last: False")
            self.drop_last = False

        self._enforce_value_for_arg(kwargs, 'timeout', 0)
        self._enforce_value_for_arg(kwargs, 'worker_init_fn', None)
        self._enforce_value_for_arg(kwargs, 'multiprocessing_context', None)
        self._enforce_value_for_arg(kwargs, 'generator', None)

        if 'prefetch_factor' in kwargs:
            self.prefetch_factor = kwargs.get('prefetch_factor')
            if self.prefetch_factor < 1:
                print("Warning: prefetch_factor < 1 is not supported by MediaDataLoader, updating to 1")
                self.prefetch_factor = 1
            elif self.prefetch_factor > 3:
                print("Warning: prefetch_factor updated from ", self.prefetch_factor, " to 3")
                self.prefetch_factor = 3
            else:
                print("MediaDataLoader got prefetch_factor ", self.prefetch_factor)
        else:
            self.prefetch_factor = 3
            print("Warning: MediaDataLoader using prefetch_factor 3")

        self._enforce_value_for_arg(kwargs, 'persistent_workers', False)

    def _enforce_value_for_arg(self, kwargs, var_name, expected_value, allow_default=True):
        if not allow_default and kwargs.get(var_name) is None:
            raise ValueError(f"'{var_name}' is supported only as {expected_value}")
        # In case the value was not sent, it will be 'None'
        if kwargs.get(var_name) is not None and kwargs.get(var_name) != expected_value:
            raise ValueError(f"'{var_name}' is supported only as {expected_value}")

class ResnetDataLoader(torch.utils.data.DataLoader):
    def __init__(self, *args, **kwargs):
        keyword_args = copy.deepcopy(kwargs)
        keyword_args.update(dict(zip(inspect.getfullargspec(super(ResnetDataLoader, self).__init__).args[1:], args)))
        channels_last = keyword_args.get('channels_last', False)

        self.DeviceType = htexp._get_device_type()

        self.fallback_activated = False
        try:
            print("HabanaDataLoader device type ", self.DeviceType)

            self.aeon_fallback_activated = False
            if 'PT_HPU_MEDIA_PIPE' in os.environ:
                self.aeon_fallback_activated = os.getenv('PT_HPU_MEDIA_PIPE').lower() in ('false', '0', 'f')

            # Try aeon when HPUMediaPipe is not available
            if (not self.aeon_fallback_activated) and (isGaudi2(self.DeviceType) or isGreco(self.DeviceType)):
                try:
                    from habana_frameworks.medialoaders.torch.media_dataloader_mediapipe import HPUMediaPipe

                except (ImportError) as e:
                    print(f"Failed to initialize Habana media Dataloader, error: {str(e)}\nFallback to aeon dataloader")
                    self.aeon_fallback_activated = True

            if isGaudi(self.DeviceType) or (self.aeon_fallback_activated):
                from .aeon_config import get_aeon_config
                from .aeon_transformers import HabanaAeonTransforms
                from .aeon_manifest import generate_aeon_manifest
                import habana_dataloader.habana_dl_app

                self._aeon_dl_handle_vars(keyword_args)
                torch_transforms = self.dataset.transform
                aeon_data_dir = self.dataset.root

                ht = HabanaAeonTransforms(torch_transforms)
                aeon_transform_config, is_train = ht.get_aeon_transforms()
                manifest_filename = generate_aeon_manifest(self.dataset.imgs)
                aeon_config_json = get_aeon_config(aeon_data_dir, manifest_filename, aeon_transform_config, self.batch_size, self.num_workers, channels_last, is_train)
                self.aeon = habana_dataloader.habana_dl_app.HabanaAcceleratedPytorchDL(aeon_config_json,
                                                                        True, # pin_memory
                                                                        True, # use_prefetch
                                                                        channels_last,
                                                                        self.drop_last
                                                                        )
                print("Running with Habana aeon DataLoader")

            elif isGaudi2(self.DeviceType) or isGreco(self.DeviceType):

                self._media_dl_handle_vars(keyword_args)
                root = self.dataset.root
                torch_transforms = self.dataset.transform
                num_instances=_get_world_size()
                instance_id=_get_rank()
                pipeline = HPUMediaPipe(a_torch_transforms=torch_transforms, a_root=root, a_batch_size=self.batch_size,
                                        a_shuffle=self.shuffle, a_drop_last=self.drop_last, a_prefetch_count=self.prefetch_factor,
                                        a_num_instances=num_instances, a_instance_id=instance_id, a_device=deviceStr(self.DeviceType))

                from habana_frameworks.mediapipe.plugins.iterator_pytorch import HPUResnetPytorchIterator
                self.iterator = HPUResnetPytorchIterator(mediapipe=pipeline)

                print(f"Running with Habana media DataLoader with num_instances = {num_instances}, instance_id = {instance_id}.")
            else:
                raise ValueError("Unsupported device")

        except (ValueError, ImportError) as e:
            print(f"Failed to initialize Habana Dataloader, error: {str(e)}\nRunning with PyTorch Dataloader")
            self.fallback_activated = True
            super(ResnetDataLoader, self).__init__(*args, **kwargs)

    def __len__(self):
        if self.fallback_activated:
            return super().__len__()
        elif isGaudi(self.DeviceType) or (self.aeon_fallback_activated == True):
            return len(self.aeon)
        elif isGaudi2(self.DeviceType):
            return len(self.iterator)
        elif isGreco(self.DeviceType):
            return len(self.iterator)
        else:
            assert False, "Invalid device type"

    def __iter__(self):
        if self.fallback_activated:
            return super().__iter__()
        elif isGaudi(self.DeviceType) or (self.aeon_fallback_activated == True):
            return iter(self.aeon)
        elif isGaudi2(self.DeviceType):
            return iter(self.iterator)
        elif isGreco(self.DeviceType):
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
        if is_shuffle_default == True:
            if isinstance(sampler, torch.utils.data.RandomSampler):
                self.shuffle = True
                print("Warning: Updated shuffle to True as sampler is RandomSampler")
            elif isinstance(sampler, torch.utils.data.distributed.DistributedSampler) and (sampler.shuffle == True):
                self.shuffle = True
                print(
                    "Warning: Updated shuffle to True as sampler is DistributedSampler with shuffle True")
        if sampler != None:
            print(
                "Warning: sampler is not supported by MediaDataLoader, ignoring sampler: ", sampler)

        self._enforce_value_for_arg(kwargs, 'batch_sampler', None)

        num_workers = kwargs.get('num_workers', 0)
        if num_workers != 0:
            print(
                "Warning: num_workers is not supported by MediaDataLoader, ignoring num_workers: ", num_workers)

        self._enforce_value_for_arg(kwargs, 'collate_fn', None)

        # ignored pin_memory

        if 'drop_last' in kwargs:
            self.drop_last = kwargs.get('drop_last')
            if self.drop_last == False:
                print(
                    "Warning: MediaDataLoader got drop_last: False, round up of last batch will be done")
            else:
                print("MediaDataLoader got drop_last: ", self.drop_last)
        else:
            print(
                "Warning: MediaDataLoader using drop_last: False, round up of last batch will be done")
            self.drop_last = False

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
            self.prefetch_factor = 3
            print("Warning: MediaDataLoader using prefetch_factor 3")

        self._enforce_value_for_arg(kwargs, 'persistent_workers', False)

    def _enforce_value_for_arg(self, kwargs, var_name, expected_value, allow_default=True):
        if not allow_default and kwargs.get(var_name) is None:
            raise ValueError(f"'{var_name}' is supported only as {expected_value}")
        # In case the value was not sent, it will be 'None'
        if kwargs.get(var_name) is not None and kwargs.get(var_name) != expected_value:
            raise ValueError(f"'{var_name}' is supported only as {expected_value}")

def _is_coco_dataset(dataset):
        try:
            if "COCO 2017 Dataset" in dataset.data["info"]["description"]:
                return True
        except:
            return False
        return False

def _is_hpumediapipe_available():
    try:
        from habana_frameworks.medialoaders.torch.media_dataloader_mediapipe import HPUMediaPipe
        return True
    except (ImportError) as e:
        print(f"import HPUMediaPipe error: {str(e)}")
        return False

class HabanaDataLoader:
    def __init__(self, *args, **kwargs):
        dataset = kwargs.get("dataset", args[0])
        dataloader_type = None
        if isinstance(dataset, torchvision.datasets.ImageFolder):
            dataloader_type = ResnetDataLoader
        elif _is_coco_dataset(dataset):
            self.DeviceType = htexp._get_device_type()
            print("HabanaDataLoader device type ", self.DeviceType)

            self.aeon_fallback_activated = False
            if 'PT_HPU_MEDIA_PIPE' in os.environ:
                self.aeon_fallback_activated = os.getenv('PT_HPU_MEDIA_PIPE').lower() in ('false', '0', 'f')

            media_multi = False
            if (self.aeon_fallback_activated == False) and ('PT_HPU_ENABLE_MEDIA_PIPE_SSD_MULTI_CARD' in os.environ):
                    media_multi = os.getenv('PT_HPU_ENABLE_MEDIA_PIPE_SSD_MULTI_CARD').lower() in ('true', '1', 't')

            # Try aeon when HPUMediaPipe is not available
            if (not self.aeon_fallback_activated) and (isGaudi2(self.DeviceType) or isGreco(self.DeviceType)):
                num_instances = _get_world_size()
                if _is_hpumediapipe_available() == False:
                    print("Fallback to aeon dataloader")
                    self.aeon_fallback_activated = True
                elif (media_multi == False) and (num_instances > 1):
                    print("Fallback to aeon dataloader as world_size is ", num_instances)
                    self.aeon_fallback_activated = True

            if isGaudi(self.DeviceType) or (self.aeon_fallback_activated):
                dataloader_type = SSDDataLoader
            elif isGaudi2(self.DeviceType):
                dataloader_type = SSDMediaDataLoader
            elif isGreco(self.DeviceType):
                dataloader_type = SSDMediaDataLoader

        try:
            self.dataloader = dataloader_type(*args, **kwargs)

        except Exception as e:
            fallback_enabled = os.getenv('DATALOADER_FALLBACK_EN', True)
            if fallback_enabled:
                #Fallback to PT Dataloader
                print(f"Failed to initialize Habana Dataloader, error: {str(e)}\nRunning with PyTorch Dataloader")
                self.dataloader = torch.utils.data.DataLoader(*args, **kwargs)
            else:
                print(f"Habana dataloader configuration failed: {e}")
                raise

    def __iter__(self):
        self.iter = iter(self.dataloader)
        return self
    def __next__(self):
        return next(self.iter)
    def __len__(self):
        return len(self.dataloader)