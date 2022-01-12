# Copyright (c) 2021, Habana Labs Ltd.  All rights reserved.

from setuptools import setup, Extension
from torch.utils import cpp_extension
from pathlib import Path
import os
import sys

if not os.environ.get("BUILD_ROOT_LATEST"):
      print("Expected 'BUILD_ROOT_LATEST' to be set")
      sys.exit(1)

if not os.environ.get("THIRD_PARTIES_ROOT"):
      print("Expected 'THIRD_PARTIES_ROOT' to be set")
      sys.exit(1)

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

setup(name='habana-torch-dataloader',
      version=get_version(),
      description="Habana's Pytorch-specific optimized software dataloader",
      packages=["habana_dataloader"],
      ext_modules=[cpp_extension.CppExtension(  'habana_dataloader.habana_dl_app',
                                                ['main.cpp'],
                                                include_dirs=[
                                                      os.path.join(os.path.dirname(os.path.realpath(__file__)), 'include'),
                                                      os.path.join(os.environ["THIRD_PARTIES_ROOT"], "pybind11", "include"),
                                                      os.path.join(os.environ["THIRD_PARTIES_ROOT"], "json", "include")
                                                ],
                                                libraries=['aeon'],
                                                library_dirs=[
                                                      os.environ["BUILD_ROOT_LATEST"]
                                                ],
                                                runtime_library_dirs=[
                                                      os.environ["BUILD_ROOT_LATEST"],
                                                      cpp_extension.TORCH_LIB_PATH
                                                ]
                                              )
                  ],
      cmdclass={'build_ext': cpp_extension.BuildExtension})
