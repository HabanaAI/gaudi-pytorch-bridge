#!/usr/bin/env python
# ##############################################################################
# Copyright (C) 2021-2023 Habana Labs, Ltd. an Intel Company
# All Rights Reserved.
#
# Unauthorized copying of this file or any element(s) within it, via any medium
# is strictly prohibited.
# This file contains Habana Labs, Ltd. proprietary and confidential information
# and is subject to the confidentiality and license agreements under which it
# was provided.
#
# ##############################################################################

from setuptools import setup, find_namespace_packages
from distutils.file_util import copy_file
from torch.utils import cpp_extension

import copy
import glob
import os
import torch


def _check_env_flag(name, default=""):
    return os.getenv(name, default).upper() in ["ON", "1", "YES", "TRUE", "Y"]


root = os.environ["PYTORCH_MODULES_ROOT_PATH"]

DEBUG = _check_env_flag("DEBUG")
PT_VER = '.'.join(torch.version.__version__.split('.')[:2])

include_dirs = [
    root,
    os.path.join(root, "pytorch_helpers"),
    os.path.join(root, "pt_ver", PT_VER),
    os.path.join(os.environ["SYNAPSE_ROOT"], "include"),
    os.path.join(os.environ["HCL_ROOT"], "include"),
    os.path.join(os.environ["THIRD_PARTIES_ROOT"], "abseil-cpp"),
    os.path.join(os.environ["PYTORCH_FORK_ROOT"], "third_party", "pybind11", "include"),
    os.path.join(os.environ["THIRD_PARTIES_ROOT"], "json", "single_include"),
    os.path.join(os.environ["THIRD_PARTIES_ROOT"], "fmt-9.1.0", "include"),
    os.environ["SPECS_EXT_ROOT"],
] + os.environ["HL_LOGGER_INCLUDE_DIRS"].split(';')

extra_compile_args = [
    "-std=c++17",
    "-DMAX_DEVICES_PER_BOX=8",
    "-fopenmp",
    "-fpermissive",
    "-DFMT_HEADER_ONLY",
]
extra_link_args = []

if DEBUG:
    extra_compile_args += ["-O0", "-g"]
    extra_link_args += ["-O0", "-g"]
else:
    extra_compile_args += ["-DNDEBUG", "-g0"]


def get_version():
    HABANA_DEFAULT_VERSION = "0.0.0.0"
    version = os.getenv('RELEASE_VERSION')
    if version:
        build_number = os.getenv('RELEASE_BUILD_NUMBER')
        if build_number:
            return version + '.' + build_number
        else:
            return version + '.0'
    else:
        try:
            import subprocess
            import re

            describe = (
                subprocess.check_output(
                    ["git", "-C", root, "describe", "--abbrev=7", "--tags", "--dirty"])
                .decode("ascii").strip())
            sha = re.search(r"g([a-z0-9\-]+)", describe).group(1)
            return HABANA_DEFAULT_VERSION + "+" + sha
        except Exception as e:
            import sys
            print("Error getting version: {}".format(e), file=sys.stderr)
            return f"{HABANA_DEFAULT_VERSION}+unknown"


ext_src_root = os.path.join(root, "python_packages/habana_frameworks/torch")

extensions = [
    ("habana_frameworks.torch._core_C", glob.glob(f"{ext_src_root}/core/*.cpp")),
    ("habana_frameworks.torch._hpex_C", glob.glob(f"{ext_src_root}/hpex/csrc/*.cpp")),
    ("habana_frameworks.torch._hpu_C", glob.glob(f"{ext_src_root}/hpu/csrc/*.cpp")),
    ("habana_frameworks.torch.distributed._hccl_C", glob.glob(f"{ext_src_root}/distributed/hccl/*.cpp")),
    ("habana_frameworks.torch.utils._experimental_C", glob.glob(f"{ext_src_root}/utils/experimental/csrc/*.cpp")),
    ("habana_frameworks.torch.utils._profiler_C", glob.glob(f"{ext_src_root}/utils/profiler/csrc/*.cpp")),
    ("habana_frameworks.torch.utils._debug_C", glob.glob(f"{ext_src_root}/utils/debug/csrc/*.cpp")),
    ("habana_frameworks.torch.utils._activity_profiler_C", glob.glob(f"{ext_src_root}/activity_profiler/csrc/*.cpp")),
    ("habana_frameworks.torch.utils._event_dispatcher_C", glob.glob(f"{ext_src_root}/utils/event_dispatcher/csrc/*.cpp")),
]

if int(PT_VER[0]) >= 2:
    extensions.append(
        ("habana_frameworks.torch.dynamo.compile_backend._recipe_compiler_C", glob.glob(f"{ext_src_root}/dynamo/compile_backend/*.cpp")))


assert not any(
    ext for ext, src in extensions if len(src) == 0
), f"no sources for extension {next(e for e,s in extensions if len(s)==0)} extensions={extensions}]"


class BuildExt(cpp_extension.BuildExtension.with_options(no_python_abi_suffix=True, parallel=len(extensions))):
    def run(self):

        super(BuildExt, self).run()
        build_root = os.environ["BUILD_ROOT_LATEST"]
        libs = [
            os.path.join(build_root, l)
            for l in os.listdir(build_root)
            if "pytorch" in l
        ]

        # CI has dangling symlinks
        libs = [l for l in libs if os.path.exists(l)]

        libs_path = os.path.join(self.build_lib, "habana_frameworks", "torch", "lib")
        os.makedirs(libs_path, exist_ok=True)
        for lib in libs:
            copy_file(lib, libs_path)

        # copying exposed header files into package
        build_root = os.path.join(os.environ["PYTORCH_MODULES_ROOT_PATH"], "include", "habanalabs")
        include_path = os.path.join(self.build_lib, "habana_frameworks", "torch", "include")
        os.makedirs(include_path, exist_ok=True)
        headerfiles = (filename for filename in os.listdir(build_root) if filename.endswith(".h"))
        for header in headerfiles:
            copy_file(os.path.join(build_root, header), include_path)

    def build_extension(self, ext):
        build_root = "PYTORCH_MODULES_DEBUG_BUILD" if self.debug else "PYTORCH_MODULES_RELEASE_BUILD"
        ext_build_temp_root = os.path.join(os.environ[build_root], "ext_temp")
        #  build_temp is a property of extension builder, not extension. For this unfortunate reason normally
        #  individual extensions override one another which prevents parallel build of multiple extensions.
        #  This overrides build_temp per extension but in order to allow prallelism it is done by copying builder for each extension.
        ext_builder = copy.copy(self)
        ext_builder.build_temp = os.path.join(ext_build_temp_root, ext.name)
        super(BuildExt, ext_builder).build_extension(ext)


setup(
    name="habana-torch-plugin",
    description="This package provides PyTorch bridge interfaces and DL training support modules like optimizers, mixed precision configuration, fused kernels etc on Habana® Gaudi®",
    url="https://habana.ai/",
    license="See LICENSE.txt",
    license_files=("LICENSE.txt",),
    author="Habana Labs Ltd., an Intel Company",
    author_email="support@habana.ai",
    version=get_version(),
    zip_safe=False,
    packages=find_namespace_packages(include=["habana_frameworks.*", "torch_hpu"]),
    package_data={
        "habana_frameworks.torch.hpex.hmp": ["*.txt"],
        "habana_frameworks.torch": ["lib/*.so"],
    },
    ext_modules=[
        cpp_extension.CppExtension(
            name=ext_name,
            sources=ext_src,
            language="c++",
            include_dirs=include_dirs,
            library_dirs=[os.environ["BUILD_ROOT_LATEST"]],
            libraries=[], # libraries should be lazily loaded before loading pybinds
            runtime_library_dirs=["$ORIGIN/lib/"],
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
        )
        for ext_name, ext_src in extensions
    ],
    cmdclass={"build_ext": BuildExt},
)
