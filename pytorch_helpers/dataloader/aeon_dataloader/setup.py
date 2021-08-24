# Copyright (c) 2021, Habana Labs Ltd.  All rights reserved.

from setuptools import setup, Extension
from torch.utils import cpp_extension
from pathlib import Path
import os
import sys

if not os.environ.get("PYTORCH_MODULES_ROOT_PATH"):
      print("Expected 'PYTORCH_MODULES_ROOT_PATH' to be set")
      sys.exit(1)

if not os.environ.get("DATA_LOADER_AEON_LIB_PATH"):
      print("Expected 'DATA_LOADER_AEON_LIB_PATH' to be set")
      sys.exit(1)

setup(name='habana_aeon_dataloader',
      version='1.0',
      description="Habana's Pytorch-specific dataloader based on Aeon dataloader",
      packages=["habana_aeon_dataloader"],
      ext_modules=[cpp_extension.CppExtension(  'habana_aeon_dataloader.aeon_app',
                                                ['main.cpp'],
                                                include_dirs=[
                                                      os.path.join(os.path.dirname(os.path.realpath(__file__)), 'include'),
                                                      os.path.join(os.environ["PYTORCH_MODULES_ROOT_PATH"], "third_party", "pybind11", "include"),
                                                      os.path.join(os.environ["THIRD_PARTIES_ROOT"], "json", "include")
                                                ],
                                                libraries=['aeon'],
                                                library_dirs=[
                                                      os.path.dirname(os.environ["DATA_LOADER_AEON_LIB_PATH"])
                                                ],
                                                runtime_library_dirs=[
                                                      os.path.dirname(os.environ["DATA_LOADER_AEON_LIB_PATH"]),
                                                      cpp_extension.TORCH_LIB_PATH
                                                ]
                                              )
                  ],
      cmdclass={'build_ext': cpp_extension.BuildExtension})
