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
import random
import tempfile


def _is_valid_image(image):
    return image.lower().endswith(".jpeg") or image.lower().endswith(".jpg")


def generate_aeon_manifest(imgs):
    """Generates aeon manifest file from ImageFolder dataset
    Args:
        imgs: a list of Tuples: (file_path, label_num)
    """
    manifest = tempfile.NamedTemporaryFile(mode="w", delete=False)
    manifest.write("@FILE\tSTRING\n")

    for file_path, label_num in imgs:
        if not _is_valid_image(file_path):
            raise ValueError(
                "HabanaDataLoader supports only jpg/jpeg files, found unsupported file: {}.".format(file_path)
            )
        manifest.write(str(file_path) + "\t" + str(label_num) + "\n")
    manifest.close()
    return manifest.name
