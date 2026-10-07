###############################################################################
# Copyright (c) 2021-2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
###############################################################################


from collections.abc import Callable
from typing import Any

import torch.utils.data
import torchvision.datasets


class ImageFolderWithManifest(torchvision.datasets.DatasetFolder):
    def __init__(
        self,
        root: str,
        manifest: dict,
        loader: Callable[[str], Any] = torchvision.datasets.folder.default_loader,
        extensions: tuple[str, ...] | None = torchvision.datasets.folder.IMG_EXTENSIONS,
        transform: Callable | None = None,
        target_transform: Callable | None = None,
        is_valid_file: Callable[[str], bool] | None = None,
    ) -> None:
        self.root = root
        self.manifest = manifest
        self.loader = loader
        self.extensions = extensions

        has_separate_transform = transform is not None or target_transform is not None

        # for backwards-compatibility
        self.transform = transform
        self.target_transform = target_transform

        transforms = None
        if has_separate_transform:
            transforms = torchvision.datasets.vision.StandardTransform(transform, target_transform)
        self.transforms = transforms

        self.is_valid_file = is_valid_file

        self._cashed_samples = None
        self._cashed_classes = None
        self._cashed_class_to_idx = None
        self._cashed_targets = None
        self._cashed_imgs = None

    @property
    def samples(self):
        if not self._cashed_samples:
            self._cashed_samples = self.make_dataset(self.root, self.class_to_idx, self.extensions, self.is_valid_file)
        return self._cashed_samples

    @property
    def classes(self):
        if not self._cashed_classes:
            self._cashed_classes, self._cashed_class_to_idx = self.find_classes(self.root)
        return self._cashed_classes

    @property
    def class_to_idx(self):
        if not self._cashed_class_to_idx:
            self._cashed_classes, self._cashed_class_to_idx = self.find_classes(self.root)
        return self._cashed_class_to_idx

    @property
    def targets(self):
        if not self._cashed_targets:
            self._cashed_targets = [s[1] for s in self.samples]
        return self._cashed_targets

    @property
    def imgs(self):
        if not self._cashed_imgs:
            self._cashed_imgs = self.samples
        return self._cashed_imgs

    def __len__(self) -> int:
        file_list = self.manifest.get("file_list", None)
        if file_list:
            return len(file_list)
        else:
            return len(self.samples)


class HabanaDataLoader(torch.utils.data.DataLoader):
    pass
