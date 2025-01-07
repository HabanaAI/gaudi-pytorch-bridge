#!/usr/bin/env python
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
import shutil

from setup_utils import InstallCMakeLibs, PrebuiltPtExtension, get_version
from setuptools import find_namespace_packages, setup
from setuptools.command.build_ext import build_ext

modules_build_dir_var = "PYTORCH_MODULES_BUILD"
modules_build_dir = os.getenv(modules_build_dir_var)


if modules_build_dir is None:
    raise EnvironmentError(f"{modules_build_dir_var} not set")
build_dir = os.path.join(modules_build_dir, "python_packages")
if os.path.exists(build_dir):
    shutil.rmtree(build_dir)
os.makedirs(build_dir)

wheel_build_dir_var = "PYTORCH_MODULES_WHL_BUILD_DIR"
wheel_build_dir = os.getenv(wheel_build_dir_var)
if wheel_build_dir is None:
    raise EnvironmentError(f"{wheel_build_dir_var} not set")

wheel_pt_vers_var = "PT_WHEEL_VERS"
wheel_pt_vers = os.getenv(wheel_pt_vers_var)
if wheel_pt_vers is None:
    raise EnvironmentError(f"{wheel_pt_vers_var} not set")


class InstallHeaders(build_ext):
    def run(self):
        # copying exposed header files into package
        pytorch_modules_root_var = "PYTORCH_MODULES_ROOT_PATH"
        pytorch_modules_root = os.getenv(pytorch_modules_root_var)
        if pytorch_modules_root is None:
            raise EnvironmentError(f"{pytorch_modules_root_var} not set")
        src_path = os.path.join(pytorch_modules_root, "include", "habanalabs")
        dst_path = os.path.join(self.build_lib, "habana_frameworks", "torch", "include")
        shutil.copytree(src_path, dst_path)


def get_installed_symengine():
    import symengine

    return "symengine==" + symengine.__version__


setup(
    name="habana-torch-plugin",
    description="This package provides PyTorch bridge interfaces and DL training support modules "
    "like optimizers, mixed precision configuration, fused kernels etc on Habana® Gaudi®",
    url="https://habana.ai/",
    license="See LICENSE.txt",
    license_files=("LICENSE.txt",),
    author="Habana Labs Ltd., an Intel Company",
    author_email="support@habana.ai",
    version=get_version(),
    zip_safe=False,
    packages=find_namespace_packages(include=["habana_frameworks.*", "habana_frameworks", "torch_hpu"]),
    package_data={"habana_frameworks.torch": ["*.txt"]},
    ext_modules=[PrebuiltPtExtension("habana_frameworks.torch", modules_build_dir)],
    cmdclass={
        "build_ext": InstallHeaders,
        "install_lib": InstallCMakeLibs(
            module_namespace=os.path.join("habana_frameworks", "torch"),
            wheel_name="habana_torch_plugin",
            wheel_pt_vers=wheel_pt_vers,
            wheel_build_dir=wheel_build_dir,
            ignore_func=shutil.ignore_patterns("*.debug", "__pycache__"),
        ),
    },
    entry_points={
        "torch.backends": [
            "device_backend = habana_frameworks.autoload:__autoload",
        ],
    },
    options={
        "egg_info": {"egg_base": build_dir},
        "build": {"build_base": build_dir + "/build"},
        "bdist_wheel": {"dist_dir": build_dir + "/dist"},
        "sdist": {"dist_dir": build_dir + "/dist"},
    },
    install_requires=[get_installed_symengine()],
)
