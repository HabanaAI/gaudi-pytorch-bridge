#!/usr/bin/env python

from setuptools import setup, find_packages
from torch.utils import cpp_extension

import os
import glob


def _check_env_flag(name, default=""):
    return os.getenv(name, default).upper() in ["ON", "1", "YES", "TRUE", "Y"]


root = os.path.join(os.environ["PYTORCH_MODULES_ROOT_PATH"])

DEBUG = _check_env_flag("DEBUG")

include_dirs = [
    root,
    os.path.join(root, "pytorch_helpers"),
    os.path.join(os.environ["SYNAPSE_ROOT"], "include"),
    os.path.join(os.environ["THIRD_PARTIES_ROOT"], "abseil-cpp"),
    os.path.join(os.environ["THIRD_PARTIES_ROOT"], "pybind11", "include"),
]

libraries = [
    "habana_pytorch_plugin",
]

extra_compile_args = [
    "-std=c++14",
    "-DMAX_DEVICES_PER_BOX=8",
]
extra_link_args = []

if DEBUG:
    extra_compile_args += ["-O0", "-g"]
    extra_link_args += ["-O0", "-g"]
else:
    extra_compile_args += ["-DNDEBUG", "-g0"]


def get_version():
    try:
        import subprocess
        import re

        describe = (
            subprocess.check_output(
                ["git", "-C", root, "describe", "--abbrev=7", "--tags", "--dirty"]
            )
            .decode("ascii")
            .strip()
        )
        version = re.search(r"\d+(\.\d+)*", describe).group(0)
        sha = re.search(r"g([a-z0-9\-]+)", describe).group(1)
        return version + "+" + sha
    except Exception as e:
        print("Error getting version: {}".format(e), file=sys.stderr)
        return "0.0.0+unknown"


hpex_csrc = glob.glob("habana_frameworks/torch/hpex/csrc/*.cpp")

setup(
    name="habana-torch",
    version=get_version(),
    zip_safe=False,
    packages=find_packages(exclude=["build"]),
    package_data={
        "": ["ops_bf16.txt", "ops_fp32.txt", "ops_multi_inputs.txt"],
    },
    ext_modules=[
        cpp_extension.CppExtension(
            name="_core_C",
            sources=["habana_frameworks/torch/core/bindings.cpp"],
            language="c++",
            include_dirs=include_dirs,
            libraries=libraries,
            library_dirs=[os.environ["BUILD_ROOT_LATEST"]],
            extra_compile_args=extra_compile_args,
        ),
        cpp_extension.CppExtension(
            name="_hpex_C",
            sources=hpex_csrc,
            language="c++",
            include_dirs=include_dirs,
            libraries=libraries,
            library_dirs=[os.environ["BUILD_ROOT_LATEST"]],
            extra_cflags=["-fopenmp -fpermissive"],
            extra_compile_args=extra_compile_args,
        ),
    ],
    cmdclass={
        "build_ext": cpp_extension.BuildExtension.with_options(
            no_python_abi_suffix=True
        )
    },
)
