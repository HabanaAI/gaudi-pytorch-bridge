#!/usr/bin/env python

from setuptools import setup, find_namespace_packages
from distutils.file_util import copy_file
from torch.utils import cpp_extension

import os
import glob


def _check_env_flag(name, default=""):
    return os.getenv(name, default).upper() in ["ON", "1", "YES", "TRUE", "Y"]


root = os.environ["PYTORCH_MODULES_ROOT_PATH"]

DEBUG = _check_env_flag("DEBUG")

include_dirs = [
    root,
    os.path.join(root, "pytorch_helpers"),
    os.path.join(os.environ["SYNAPSE_ROOT"], "include"),
    os.path.join(os.environ["HCL_ROOT"], "include"),
    os.path.join(os.environ["THIRD_PARTIES_ROOT"], "abseil-cpp"),
    os.path.join(os.environ["THIRD_PARTIES_ROOT"], "pybind11", "include"),
    os.path.join(os.environ["THIRD_PARTIES_ROOT"], "spdlog", "include"),
    os.environ["SPECS_EXT_ROOT"],
]

libraries = [
    "habana_pytorch_plugin",
]

extra_compile_args = [
    "-std=c++17",
    "-DMAX_DEVICES_PER_BOX=8",
    "-fopenmp",
    "-fpermissive",
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
            print("Error getting version: {}".format(e), file=sys.stderr)
            return f"{HABANA_DEFAULT_VERSION}+unknown"


core_csrc = glob.glob("habana_frameworks/torch/core/*.cpp")
hccl_csrc = glob.glob("habana_frameworks/torch/core/hccl/*.cpp")
hpex_csrc = glob.glob("habana_frameworks/torch/hpex/csrc/*.cpp")
profiler_csrc = glob.glob("habana_frameworks/torch/profiler/csrc/*.cpp")


class BuildExt(cpp_extension.BuildExtension.with_options(no_python_abi_suffix=True)):
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
            name="habana_frameworks.torch._core_C",
            sources=core_csrc,
            language="c++",
            include_dirs=include_dirs,
            library_dirs=[os.environ["BUILD_ROOT_LATEST"]],
            libraries=libraries,
            runtime_library_dirs=["$ORIGIN/lib/"],
            extra_compile_args=extra_compile_args,
        ),
        cpp_extension.CppExtension(
            name="habana_frameworks.torch.core._hccl_C",
            sources=hccl_csrc,
            language="c++",
            include_dirs=include_dirs,
            library_dirs=[os.environ["BUILD_ROOT_LATEST"]],
            libraries=libraries,
            runtime_library_dirs=["$ORIGIN/lib/"],
            extra_compile_args=extra_compile_args,
        ),
        cpp_extension.CppExtension(
            name="habana_frameworks.torch._hpex_C",
            sources=hpex_csrc,
            language="c++",
            include_dirs=include_dirs,
            library_dirs=[os.environ["BUILD_ROOT_LATEST"]],
            libraries=libraries,
            runtime_library_dirs=["$ORIGIN/lib/"],
            extra_compile_args=extra_compile_args,
        ),
        cpp_extension.CppExtension(
            name="habana_frameworks.torch._profiler_C",
            sources=profiler_csrc,
            language="c++",
            include_dirs=include_dirs,
            library_dirs=[os.environ["BUILD_ROOT_LATEST"]],
            libraries=libraries,
            runtime_library_dirs=["$ORIGIN/lib/"],
            extra_compile_args=extra_compile_args,
        ),
    ],
    cmdclass={"build_ext": BuildExt},
)
