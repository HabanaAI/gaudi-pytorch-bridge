#!/usr/bin/env python
###############################################################################
# Copyright (C) 2021-2023 Habana Labs, Ltd. an Intel Company
# All Rights Reserved.
#
# Unauthorized copying of this file or any element(s) within it, via any medium
# is strictly prohibited.
# This file contains Habana Labs, Ltd. proprietary and confidential information
# and is subject to the confidentiality and license agreements under which it
# was provided.
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
