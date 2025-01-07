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


import os
import pathlib
import time

import habana_torch_dataloader
import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.utils.data as torch_data
import torchvision.datasets as datasets
import torchvision.transforms as transforms

X = []
Y = []


def save_points(x, y):
    X.append(x)
    Y.append(y)
    print(x, y)


def profile_dataloader(s, dataloader, dl_type, workers, num_iterations):

    t_sum = 0
    last_time = time.time()

    for i, data in enumerate(dataloader):
        t = time.time()
        t_diff = t - last_time
        t_sum += t_diff
        ##save_points(t_diff, t_sum)
        save_points(i, t_diff)
        last_time = t

        if i >= num_iterations:
            break

    print(
        f"Total time taken for {s} dataloader({dl_type}) with {workers} \
           workers for {num_iterations} iterations is = {t_sum} sec"
    )


def profile_dataloader_for_resnet(dataset_path, workers, num_iterations, batch_size, dl_worker_type):
    matplotlib.use("tkagg")

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = transforms.Compose(
        [transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize]
    )

    train_dir = pathlib.Path(dataset_path)
    train = datasets.ImageFolder(train_dir, transform)

    torch_dataloader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=workers)
    habana_dataloader = habana_torch_dataloader.DataLoader(
        train, batch_size=batch_size, shuffle=True, num_workers=workers
    )

    if dl_worker_type == "MP" or dl_worker_type == "ALL":
        print("Running torch dataloader")
        profile_dataloader("torch", torch_dataloader, "MP", workers, num_iterations)
    if dl_worker_type == "MT" or dl_worker_type == "ALL":
        print("Running habana dataloader")
        profile_dataloader("habana", habana_dataloader, "MT", workers, num_iterations)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="DataLoader Profiling")
    parser.add_argument("--data-path", default="/software/data/pytorch/imagenet/ILSVRC2012/", help="dataset path")
    parser.add_argument(
        "--dl-worker-type",
        default="MT",
        type=lambda x: x.upper(),
        choices=["MT", "MP", "ALL"],
        help="select multithreading or multiprocessing or both",
    )
    parser.add_argument(
        "--num_iterations", default=50, type=int, metavar="N", help="number of iterations to run dataloader"
    )
    parser.add_argument("--batch_size", default=256, type=int, metavar="N", help="batch size per iteration")
    parser.add_argument("-j", "--workers", default=16, type=int, metavar="N", help="number of data loading workers")
    args = parser.parse_args()
    profile_dataloader_for_resnet(
        args.data_path, args.workers, args.num_iterations, args.batch_size, args.dl_worker_type
    )
    # plt.plot(X,Y)
    # plt.show()
