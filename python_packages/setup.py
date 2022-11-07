#!/usr/bin/env python

from setuptools import setup, find_namespace_packages
from distutils.file_util import copy_file
from torch.utils import cpp_extension

import os
import glob
import copy


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
    os.path.join(os.environ["PYTORCH_FORK_ROOT"], "third_party", "pybind11", "include"),
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


ext_src_root = os.path.join(root, "python_packages/habana_frameworks/torch")

extensions = [
    ("habana_frameworks.torch._core_C", glob.glob(f"{ext_src_root}/core/*.cpp")),
    ("habana_frameworks.torch._hpu_C", glob.glob(f"{ext_src_root}/hpu/csrc/*.cpp")),
    ("habana_frameworks.torch.distributed._hccl_C", glob.glob(f"{ext_src_root}/distributed/hccl/*.cpp")),
    ("habana_frameworks.torch._hpex_C", glob.glob(f"{ext_src_root}/hpex/csrc/*.cpp")),
    ("habana_frameworks.torch.utils._experimental_C", glob.glob(f"{ext_src_root}/utils/experimental/csrc/*.cpp")),
    ("habana_frameworks.torch.utils._profiler_C", glob.glob(f"{ext_src_root}/utils/profiler/csrc/*.cpp")),
    ("habana_frameworks.torch.utils._debug_C", glob.glob(f"{ext_src_root}/utils/debug/csrc/*.cpp")),
    ("habana_frameworks.torch.utils._activity_profiler_C", glob.glob(f"{ext_src_root}/activity_profiler/csrc/*.cpp")),
]
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

        libs_path = os.path.join(self.build_lib, "habana_frameworks", "torch", "utils", "lib")
        os.makedirs(libs_path, exist_ok=True)
        for lib in libs:
            copy_file(lib, libs_path)

        libs_path = os.path.join(self.build_lib, "habana_frameworks", "torch", "distributed", "lib")
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
        "habana_frameworks.torch.utils": ["lib/*.so"],
        "habana_frameworks.torch.distributed": ["lib/*.so"],
    },
    ext_modules=[
        cpp_extension.CppExtension(
            name=ext_name,
            sources=ext_src,
            language="c++",
            include_dirs=include_dirs,
            library_dirs=[os.environ["BUILD_ROOT_LATEST"]],
            libraries=libraries,
            runtime_library_dirs=["$ORIGIN/lib/"],
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
        )
        for ext_name, ext_src in extensions
    ],
    cmdclass={"build_ext": BuildExt},
)
