# Copyright (c) 2021, Habana Labs Ltd.  All rights reserved.

import json
import torch.distributed as dist

# All constant values in aeon config are picked up for resnet50,
# for optimal aeon performance.

def _is_distributed():
    return dist.is_available() and dist.is_initialized()

def _get_world_size():
    if _is_distributed():
        return dist.get_world_size()
    else:
        return 1

def _get_rank():
    if not _is_distributed():
        return dist.get_rank()
    else:
        return 0

def _get_image_config(height, width):
    image_config = {
        "type": "image",
        "height": height,
        "width": width,
        "channel_major": False,
        "output_type": "float"
    }
    return image_config

def _get_label_config():
    label_config = {
        "type": "label",
        "binary": False
    }
    return label_config

def _get_augmentation():
    augmentation_config = {
          "caffe_mode": True,
          "center": False,
          "crop_enable": True,
          "do_area_scale": True,
          "flip_enable": True,
          "horizontal_distortion": [
              0.75,
              1.33333337306976
          ],
          "scale": [
              0.08,
              1.0
          ],
          "type": "image"
    }
    return augmentation_config

def get_aeon_config(aeon_data_dir, manifest_filename, batch_size, workers, height, width, is_train=True):
    image_config = _get_image_config(height, width)
    label_config = _get_label_config()
    augmentation_config = _get_augmentation()
    instance_id = _get_rank()
    num_instances = _get_world_size()
    aeon_config = {
        "manifest_filename": manifest_filename,
        "manifest_root": aeon_data_dir,
        "etl": (image_config, label_config),
        "augmentation": [augmentation_config],
        "batch_size": batch_size,
        "decode_thread_count": workers,
        "fread_thread_count": 4,
        "instance_id": instance_id,
        "num_instances": num_instances,
        "file_shuffle_seed": 5,
        "shuffle_manifest": True,
        "iteration_mode": "ONCE"
    }
    return aeon_config