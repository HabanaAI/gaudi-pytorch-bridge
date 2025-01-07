###############################################################################
#
#  Copyright (c) 2021-2024 Intel Corporation
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
###############################################################################


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
    if _is_distributed():
        return dist.get_rank()
    else:
        return 0


def _get_image_config(transforms, channels_last):
    image_config = {
        "type": "image",
        "height": transforms["height"],
        "width": transforms["width"],
        "channel_major": not channels_last,
        "output_type": "float",
    }
    return image_config


def _get_label_config():
    label_config = {"type": "label", "binary": False}
    return label_config


def _get_augmentation(transforms, is_train):
    augmentation_config = {"type": "image"}
    if is_train:
        augmentation_config["caffe_mode"] = transforms.get("caffe_mode", False)
        augmentation_config["center"] = False
        augmentation_config["crop_enable"] = True
        augmentation_config["do_area_scale"] = True
        augmentation_config["flip_enable"] = transforms.get("flip_enable", False)
        augmentation_config["horizontal_distortion"] = [0.75, 1.33333337306976]
        augmentation_config["scale"] = [0.08, 1.0]
    else:
        augmentation_config["validation_mode"] = True
    return augmentation_config


def get_aeon_config(aeon_data_dir, manifest_filename, transforms, batch_size, workers, channels_last, is_train=True):
    image_config = _get_image_config(transforms, channels_last)
    label_config = _get_label_config()
    augmentation_config = _get_augmentation(transforms, is_train)
    instance_id = _get_rank()
    num_instances = _get_world_size()
    aeon_config = {
        "manifest_filename": manifest_filename,
        "manifest_root": aeon_data_dir,
        "etl": (image_config, label_config),
        "augmentation": [augmentation_config],
        "batch_size": batch_size,
        "file_shuffle_seed": 5,
        "iteration_mode": "ONCE",
        "instance_id": instance_id,
        "num_instances": num_instances,
    }
    if is_train:
        aeon_config["decode_thread_count"] = workers
        aeon_config["fread_thread_count"] = 4
        aeon_config["shuffle_manifest"] = True
    else:
        aeon_config["decode_thread_count"] = 1
        aeon_config["fread_thread_count"] = 1
        aeon_config["shuffle_manifest"] = False
    return aeon_config
