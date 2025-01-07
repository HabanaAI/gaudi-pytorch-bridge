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

from setup_utils import InstallCMakeLibs, PrebuiltPtExtension, SkipBuildExt, get_version
from setuptools import setup

release_build_dir_var = "PYTORCH_MODULES_RELEASE_BUILD"
release_build_dir = os.getenv(release_build_dir_var)
if release_build_dir is None:
    raise EnvironmentError(f"{release_build_dir_var} not set")
build_dir = os.path.join(release_build_dir, "pytorch_helpers/dataloader/habana_dataloader")
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

setup(
    name="habana-torch-dataloader",
    version=get_version(),
    description="Habana's Pytorch-specific optimized software dataloader",
    url="https://habana.ai/",
    license="See LICENSE.txt",
    license_files=("LICENSE.txt",),
    author="Habana Labs Ltd., an Intel Company",
    author_email="support@habana.ai",
    zip_safe=False,
    packages=["habana_dataloader"],
    ext_modules=[PrebuiltPtExtension("habana_dataloader.habana_dl_app", release_build_dir)],
    cmdclass={
        "build_ext": SkipBuildExt,
        "install_lib": InstallCMakeLibs(
            module_namespace=os.path.join("habana_dataloader"),
            wheel_name="habana_torch_dataloader",
            wheel_pt_vers=wheel_pt_vers,
            wheel_build_dir=wheel_build_dir,
            ignore_func=shutil.ignore_patterns("*.debug", "__pycache__"),
        ),
    },
    options={
        "egg_info": {"egg_base": build_dir},
        "build": {"build_base": os.path.join(build_dir, "build")},
        "bdist_wheel": {"dist_dir": os.path.join(build_dir, "dist")},
        "sdist": {"dist_dir": os.path.join(build_dir, "dist")},
    },
)
