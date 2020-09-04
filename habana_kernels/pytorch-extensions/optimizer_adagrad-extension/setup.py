from setuptools import setup, Extension
from torch.utils import cpp_extension
import os
setup(name='habanaOptimizerSparseAdagrad_cpp',
      ext_modules=[cpp_extension.CppExtension('habanaOptimizerSparseAdagrad_cpp', ['habanaOptimizerSparseAdagrad.cpp'],
      libraries=['habana_pytorch_plugin'],
      library_dirs=[os.environ['BUILD_ROOT_LATEST']])],
      include_dirs=[os.environ['PYTORCH_MODULES_ROOT_PATH'] + "/third_party/pybind11/include"],
      cmdclass={'build_ext': cpp_extension.BuildExtension})
