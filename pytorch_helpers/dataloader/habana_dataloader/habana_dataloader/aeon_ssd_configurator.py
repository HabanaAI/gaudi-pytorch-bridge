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
import os
import string
from dataclasses import dataclass
from random import shuffle
from sys import stdout

import torch.distributed as dist
import torch.utils.data as data
from torchvision import transforms


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


@dataclass
class AeonSSDConfigurator:
    dataset: data.Dataset
    batch_size: int
    num_workers: int
    shuffle: bool
    channel_last: bool
    manifest: string
    distributed: bool = True

    def __post_init__(self):
        transform = self.dataset.transform
        self.train = not transform.val
        self.transforms = transform.img_trans.transforms if self.train else transform.trans_val.transforms
        # TODO in SSD some of the transforms are not part of a compose object, they are called by name by the script.
        # Need to think what to do about it.
        additional = [] if not self.train else [transform.crop, transform.hflip, transform.normalize]

        self.transforms_config = {}
        self._make_transforms_config(additional)
        default_out_folder = os.environ.get("HABANA_MANIFEST_PATH", os.path.join(os.path.curdir, "habana_manifest"))
        self.out_folder = os.path.join(default_out_folder, "train" if self.train else "val")
        self._get_or_create_aeon_manifest()

    def _get_or_create_aeon_manifest(self):
        if not self.train:
            self.manifest = f"val_{self.manifest}"
        manifest_file = os.path.join(self.out_folder, self.manifest)
        if not os.path.exists(manifest_file):
            if _get_rank() == 0:
                os.makedirs(self.out_folder, exist_ok=True)
                self._generate_aeon_manifest()
        if self.distributed and _is_distributed():
            dist.barrier()

        with open(manifest_file) as f:
            l = len(f.readlines()) - 1
        print(f"Taking aeon config with {l} entries from <{self.out_folder}>")

    @staticmethod
    def make_object(x, y, x_len, y_len, name):
        bbox = {"xmin": x, "ymin": y, "xmax": x + x_len, "ymax": y + y_len}
        object = {"bndbox": bbox, "difficult": False, "name": name}
        return object

    @staticmethod
    def show_progress(j, size, message, file=stdout):
        prefix = "Generating aeon manifest: "
        bar_size = 50
        x = int(j * bar_size / size)
        file.write("%s[%s%s] %i/%i\r" % (prefix, "#" * x, "." * (bar_size - x), j, size))
        file.flush()

    def _generate_aeon_manifest(self):
        update = 100  # update progress of generation every # images
        restrict_images = 0  # image num restriction disabled

        manifest = ["@FILE\tFILE\tASCII_INT\n"]
        dataset_size = restrict_images if restrict_images != 0 else len(self.dataset.images)
        print(f"Generating aeon manifest for {dataset_size} images:")

        for count, (id, (image, size, annotations)) in enumerate(self.dataset.images.items()):
            objects = []
            json_out = os.path.splitext(image)[0] + ".json"

            for bbox, category in annotations:
                if bbox[0] + bbox[2] > size[1]:
                    continue
                if bbox[1] + bbox[3] > size[0]:
                    print("bad bbox dims")
                    continue
                o = self.make_object(*bbox, self.dataset.label_info[category])
                objects.append(o)
            dict_size = {"height": size[0], "width": size[1], "depth": 3}
            dict_config = {"size": dict_size, "object": objects}
            with open(os.path.join(self.out_folder, json_out), "w") as f:
                json.dump(dict_config, f)
            manifest.append(f"{json_out}\t{self.dataset.img_folder}/{image}\t{id}\n")

            if count % update == 0:
                self.show_progress(count, dataset_size, "Generating Aeon manifest: ")

            if count == dataset_size:
                break

        print("Done!")

        with open(os.path.join(self.out_folder, self.manifest), "w") as m:
            m.writelines(manifest)

    def _make_transforms_config(self, additional=[]):
        for t in self.transforms + additional:
            if t is None:
                continue
            if isinstance(t, transforms.Resize):
                self._handle_resize_crop(t)
            elif isinstance(t, transforms.ToTensor):
                self._handle_to_tensor(t)
            elif isinstance(t, transforms.Normalize):
                self._handle_normalize(t)
            elif isinstance(t, transforms.ColorJitter):
                self._handle_color_jitter(t)
            # WA for SSD transforms that are privetly defined in model garden
            elif "utils.SSDCropping" in str(type(t)):
                self._handle_ssd_cropping(t)
            elif "utils.RandomHorizontalFlip" in str(type(t)):
                self._handle_random_horizontal_flip(t)
            else:
                raise ValueError("Unsupported transform: " + str(type(t)))

    def _handle_resize_crop(self, t):
        self.transforms_config["height"] = t.size[0]
        self.transforms_config["width"] = t.size[1]
        self.transforms_config["pillow_mode"] = True

    def _handle_color_jitter(self, t):
        self.transforms_config["brightness"] = t.brightness
        self.transforms_config["saturation"] = t.saturation
        self.transforms_config["contrast"] = t.contrast
        self.transforms_config["hue"] = [180 * e for e in t.hue]  # transform to degrees

    def _handle_ssd_cropping(self, t):
        self.transforms_config["crop_enabled"] = False  # crop is handled by batch_samplers
        self.transforms_config["center"] = False
        self.transforms_config["emit_constraint_type"] = "center"  # emit bboxes whos center not within crop
        self.transforms_config["batch_samplers"] = [
            {
                "max_trials": 1,
                "max_sample": 1,
                "sample_constraint": {"emit_no_center": True, "cover_all_objects": True},
                "sampler": {"aspect_ratio": [0.5, 2.0], "scale": [0.3, 1.0]},
            },
            {
                "max_trials": 1,
                "max_sample": 1,
                "sample_constraint": {"min_jaccard_overlap": 0.1, "emit_no_center": True, "cover_all_objects": True},
                "sampler": {"aspect_ratio": [0.5, 2.0], "scale": [0.3, 1.0]},
            },
            {
                "max_trials": 1,
                "max_sample": 1,
                "sample_constraint": {"min_jaccard_overlap": 0.3, "emit_no_center": True, "cover_all_objects": True},
                "sampler": {
                    "aspect_ratio": [0.5, 2.0],
                    "scale": [0.3, 1.0],
                },
            },
            {
                "max_trials": 1,
                "max_sample": 1,
                "sample_constraint": {"min_jaccard_overlap": 0.5, "emit_no_center": True, "cover_all_objects": True},
                "sampler": {
                    "aspect_ratio": [0.5, 2.0],
                    "scale": [0.3, 1.0],
                },
            },
            {
                "max_trials": 1,
                "max_sample": 1,
                "sample_constraint": {"min_jaccard_overlap": 0.7, "emit_no_center": True, "cover_all_objects": True},
                "sampler": {
                    "aspect_ratio": [0.5, 2.0],
                    "scale": [0.3, 1.0],
                },
            },
            {
                "max_trials": 1,
                "max_sample": 1,
                "sample_constraint": {"min_jaccard_overlap": 0.9, "emit_no_center": True, "cover_all_objects": True},
                "sampler": {
                    "aspect_ratio": [0.5, 2.0],
                    "scale": [0.3, 1.0],
                },
            },
        ]

    def _handle_random_horizontal_flip(self, t):
        if t.p != 0.5:
            raise ValueError("aeon RandomHorizontalFlip supports only probability of 0.5")
        self.transforms_config["flip_enable"] = True

    def _handle_to_tensor(self, t):
        pass

    def _handle_normalize(self, t):
        mean = t.mean
        std = t.std

        # AEON currently only support constans values
        if mean != [0.485, 0.456, 0.406]:
            raise ValueError("aeon Normalize supports only mean of [0.485, 0.456, 0.406]")
        if std != [0.229, 0.224, 0.225]:
            raise ValueError("aeon Normalize supports only std of [0.229, 0.224, 0.225]")

        self.transforms_config["caffe_mode"] = True

    def _get_image_config(self):
        image_config = {
            "type": "image",
            "height": self.transforms_config["height"],
            "width": self.transforms_config["width"],
            "channel_major": not self.channel_last,
            "output_type": "float",
        }
        return image_config

    def _get_ssd_config(self):
        ssd_config = {
            "type": "localization_ssd",
            "pt_mode": True,
            "val": not self.train,
            "max_gt_boxes": 8732 if self.train else 200,
            "class_names": [f"{v}" for v in self.dataset.label_info.values()],
            "height": self.transforms_config["height"],
            "width": self.transforms_config["width"],
        }
        return ssd_config

    def _get_blob_config(self):
        blob_config = {"type": "blob", "output_count": 1, "output_type": "int32_t"}
        return blob_config

    def _get_augmentation_config(self):
        augmentation_config = {"type": "image"}

        augmentation_config["caffe_mode"] = self.transforms_config.get("caffe_mode", False)
        augmentation_config["pillow_mode"] = self.transforms_config.get("pillow_mode", False)
        augmentation_config["crop_enable"] = self.transforms_config.get("crop_enabled", False)
        augmentation_config["center"] = self.transforms_config.get("center", False)
        augmentation_config["pt_mode"] = True

        if self.train:
            augmentation_config["flip_enable"] = self.transforms_config.get("flip_enable", False)
            augmentation_config["contrast"] = self.transforms_config.get("contrast", [1, 1])
            augmentation_config["brightness"] = self.transforms_config.get("brightness", [1, 1])
            augmentation_config["saturation"] = self.transforms_config.get("saturation", [1, 1])
            augmentation_config["hue"] = self.transforms_config.get("hue", [0, 0])
            augmentation_config["batch_samplers"] = self.transforms_config.get("batch_samplers", [])
            augmentation_config["emit_constraint_type"] = self.transforms_config.get("emit_constraint_type", "")
            augmentation_config["emit_constraint_type"] = "center"
        return augmentation_config

    def get_config(self):
        image_config = self._get_image_config()
        ssd_config = self._get_ssd_config()
        blob_config = self._get_blob_config()
        augmentation_config = self._get_augmentation_config()
        instance_id = _get_rank() if self.distributed else 0
        num_instances = _get_world_size() if self.distributed else 1
        aeon_config = {
            "manifest_filename": os.path.join(self.out_folder, self.manifest),
            "manifest_root": self.out_folder + "/",
            "etl": (ssd_config, image_config, blob_config),
            "augmentation": [augmentation_config],
            "batch_size": self.batch_size,
            "file_shuffle_seed": 5,
            "iteration_mode": "ONCE",
            "instance_id": instance_id,
            "num_instances": num_instances,
        }
        if self.train:
            aeon_config["decode_thread_count"] = self.num_workers
            aeon_config["fread_thread_count"] = 4
            aeon_config["shuffle_manifest"] = True
        else:
            aeon_config["decode_thread_count"] = 1
            aeon_config["fread_thread_count"] = 1
            aeon_config["shuffle_manifest"] = False
        return aeon_config

    def is_train(self):
        return self.train
