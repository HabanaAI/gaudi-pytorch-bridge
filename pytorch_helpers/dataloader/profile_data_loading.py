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


# help:
#  mpirun -n 8 --bind-to core --map-by slot:PE=7 --report-bindings \
#       --allow-run-as-root python -u \
#       pytorch-integration/pytorch_helpers/dataloader/profile_data_loading.py  \
#       --dl-type='MP' --dl-workers=5 --profile 2>&1 | tee 8-worker-dl.log
#

import argparse
import cProfile
import json
import os
import pathlib
import pstats
import sys
import time

import torch
import torch.utils.data as torch_data
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# DATA_LOADER_AEON_LIB_PATH='/home/janand/trees/npu-stack/tf_aeon/lib_python/aeon.so'
DATA_LOADER_AEON_LIB_PATH = "/home/janand/trees/npu-stack/dev/data_loader/build/lib/aeon.so"
sys.path.append(os.path.dirname(os.environ["DATA_LOADER_AEON_LIB_PATH"]))
# from aeon_config import *
from aeon import DataLoader
from mpi4py import MPI

global mpi_comm
mpi_comm = MPI.COMM_WORLD

BATCH_SIZE = 1
IMG_HEIGHT = 224
IMG_WIDTH = 224
AEON_DATA_DIR = "/software/data/pytorch/imagenet/ILSVRC2012/"
# AEON_MANIFEST = '/home/janand/trees/npu-stack/tf_aeon/manifest.txt'
AEON_MANIFEST = "/home/janand/trees/npu-stack/data_loader/python/manifest.txt"


def getAeonConfig(instance_id, num_instances):
    return {
        "augmentation": [
            {
                "caffe_mode": False,
                "center": False,
                "crop_enable": True,
                "do_area_scale": True,
                "flip_enable": True,
                "horizontal_distortion": [0.75, 1.33333337306976],
                "scale": [0.08, 1.0],
                "type": "image",
            }
        ],
        "batch_size": BATCH_SIZE,
        "decode_thread_count": 8,
        "fread_thread_count": 4,
        "instance_id": instance_id,
        "num_instances": num_instances,
        "file_shuffle_seed": 5,
        "shuffle_manifest": True,
        "etl": [
            {"type": "image", "channel_major": False, "height": IMG_HEIGHT, "width": IMG_WIDTH, "output_type": "float"},
            {"binary": False, "type": "label"},
        ],
        "iteration_mode": "ONCE",
        "manifest_filename": AEON_MANIFEST,
        "manifest_root": AEON_DATA_DIR,
    }


dl_args = {}


class ImageRandomDataLoader:
    def __init__(self, batch_size, num_steps, train=True, drop_last=False):

        def get_val(env_var, default_val):
            val = default_val
            val_str = os.environ.get(env_var)
            if val_str is not None:
                val = int(val_str)
            return val

        self.batch_size = batch_size
        self.channels = C = get_val("CDL_IMG_NUM_CHANNELS", 3)
        self.height = H = get_val("CDL_IMG_HEIGHT", 224)
        self.width = W = get_val("CDL_IMG_WIDTH", 224)
        self.num_classes = get_val("CDL_NUM_CLASSES", 1000)
        if train:
            phase = "train"
            self.dataset_size = get_val(
                "CDL_TRAIN_DATASET_SIZE", batch_size * num_steps
            )  # num images; default 10 iterations

        self.train = train
        self.num_batches = self.dataset_size // self.batch_size
        self.curr_batch_no = 0
        self.partial_batch_size = 0
        if not drop_last:
            self.partial_batch_size = self.dataset_size % self.batch_size

        if self.partial_batch_size:
            self.num_batches = self.num_batches + 1
        print(
            "ImageRandomDataLoader: ",
            "phase=",
            phase,
            " N=",
            batch_size,
            " C=",
            C,
            " H=",
            H,
            " W=",
            W,
            " num classes=",
            self.num_classes,
            " dataset size=",
            self.dataset_size,
            "num batches(iterations) per epoch=",
            self.num_batches,
            " drop_last =",
            drop_last,
        )

    def __iter__(self):
        return self

    def __len__(self):
        return self.num_batches

    def __next__(self):
        self.curr_batch_no = self.curr_batch_no + 1
        if self.curr_batch_no <= self.num_batches:
            batch_size = self.batch_size
            if self.curr_batch_no == self.num_batches and self.partial_batch_size:
                batch_size = self.partial_batch_size

            target = torch.randint(0, self.num_classes, (batch_size,))
            image = torch.randn(batch_size, self.channels, self.height, self.width)
            return image, target
        else:
            self.curr_batch_no = 0
            raise StopIteration


def init_data_loader():
    wsize = mpi_comm.Get_size()
    rank = mpi_comm.Get_rank()
    print(" world_size :: ", wsize)
    print(" local rank :: ", rank)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(224, interpolation=2),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )

    # torch.multiprocessing.set_start_method('forkserver')
    # torch.multiprocessing.set_start_method('spawn')
    # train_dir = pathlib.Path('/root/data/pytorch/imagenet/ILSVRC2012/')
    train_dir = pathlib.Path("/software/data/pytorch/imagenet/ILSVRC2012/")
    # train_dir = pathlib.Path('/tmp/ramdisk/imagenet/ILSVRC2012/')
    dataset = datasets.ImageFolder(train_dir, transform)
    bs = BATCH_SIZE
    total_images = 1280000
    num_steps = total_images / bs / wsize
    workers = dl_args["dl-workers"]
    print("number of DL workers :: ", workers)
    print("number of steps :: ", num_steps)
    print("batch size :: ", bs)

    dl_type = dl_args["dl-type"]
    if dl_type == "SYN":
        print("synthetic dataset selected")
        dataloader = ImageRandomDataLoader(batch_size=bs, num_steps=num_steps, train=True)
    elif dl_type == "MT":
        try:
            import habana_torch_dataloader
        except ImportError:
            assert False, "Could Not import habana_torch_dataloader"
        print("Multi-Threading DL with imagenet dataset selected")
        dataloader = habana_torch_dataloader.DataLoader(dataset, batch_size=bs, num_workers=workers, shuffle=True)
    elif dl_type == "AEON":
        print("AEON dataloader selected")
        INST_ID = rank
        NUM_INSTANCES = wsize
        dataloader = DataLoader(getAeonConfig(INST_ID, NUM_INSTANCES))
    else:
        print("Multi-process DL with imagenet dataset selected")
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=bs, num_workers=workers, shuffle=True)
        # dataloader = torch.utils.data.DataLoader(dataset, batch_size=bs, num_workers=workers, shuffle=True, prefetch_factor=4)

    return dataloader, num_steps, bs, rank


def test_pytorch_data_loader_for_resnet(dataloader, num_steps, bs, rank):
    t_sum = 0
    t_ips = 0
    t_imgs = 0
    epoch = 0
    total_epoch = 1

    for epoch in range(total_epoch):
        start_time = time.time()
        temp_time = start_time
        for i, data in enumerate(dataloader):
            images, lables = data
            # image_temp = images.to('cpu', non_blocking=False)
            # if rank == 0:
            # print(image_temp[0][1])
            #   print("Image dimensions :: ",images[0].size())
            #   print("current processed images :: ", len(images))
            t_imgs += len(images)
            dl_time = time.time()
            t_diff = dl_time - start_time
            t_sum += t_diff
            start_time = dl_time
            ips = bs / t_diff
            t_ips += ips

            if rank == 0:
                print("iteration : ", i, " ips : ", ips)

            if i >= num_steps:
                break

        end_time = time.time()
        diff = end_time - temp_time
        print(" Epoch :: ", epoch)
        print("Relative ips per card  = ", t_imgs / diff)
        print("Avg iteration ips per card = ", t_ips / num_steps)
        print("Total time take = ", t_sum)
        t_sum = 0
        t_ips = 0
        t_imgs = 0


def handle_args():
    parser = argparse.ArgumentParser(description="""Run DataLoader test""")

    parser.add_argument("--dl-type", type=str, default="MP", help="supported types ::, 'MT','MP' (default),'SYN'")
    parser.add_argument("--dl-workers", type=int, default="0", help="number of DL read workers, default:5")
    parser.add_argument("--profile", action="store_true", help="enable cprofile for the parent function")
    args = parser.parse_args()
    if args.dl_type:
        dl_args["dl-type"] = args.dl_type
    if args.dl_workers is not None:
        dl_args["dl-workers"] = args.dl_workers
    if args.profile:
        dl_args["profile"] = True
    else:
        dl_args["profile"] = False


if __name__ == "__main__":
    handle_args()
    can_profile = dl_args["profile"]
    print("profiling :: ", can_profile)

    dataloader, num_steps, bs, rank = init_data_loader()

    if can_profile:
        pr = cProfile.Profile()
        pr.enable()
        test_pytorch_data_loader_for_resnet(dataloader, num_steps, bs, rank)
        pr.disable()
        stats = pstats.Stats(pr).sort_stats("tottime")
        if mpi_comm.Get_rank() == 0:
            stats.print_stats()
    else:
        test_pytorch_data_loader_for_resnet(dataloader, num_steps, bs, rank)
