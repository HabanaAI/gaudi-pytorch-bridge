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

import setuptools

root = os.path.join(os.environ["PYTORCH_MODULES_ROOT_PATH"])


def get_version():
    try:
        import re
        import subprocess

        describe = (
            subprocess.check_output(["git", "-C", root, "describe", "--abbrev=7", "--tags", "--dirty"])
            .decode("ascii")
            .strip()
        )
        version = re.search(r"\d+(\.\d+)*", describe).group(0)
        sha = re.search(r"g([a-z0-9\-]+)", describe).group(1)
        return version + "+" + sha
    except Exception as e:
        print("Error getting version: {}".format(e), file=sys.stderr)
        return "0.0.0+unknown"


setuptools.setup(
    name="habana_torch_dataloader",
    version=get_version(),
    author="Habana Labs, Ltd. an Intel Company",
    author_email="support@habana.ai",
    license="See LICENSE.txt",
    description="Package to create custom multithreaded dataloader for PyTorch",
    url="https://habana.ai/",
    install_requires=[],
    python_requires=">=3.6",
    packages=setuptools.find_packages(
        exclude=(
            "build",
            "dist",
            "habana_torch_dataloader.egg-info",
        )
    ),
    classifiers=[  #'License :: Approved ::  License',
        "Programming Language :: Python :: 3",
    ],
)
