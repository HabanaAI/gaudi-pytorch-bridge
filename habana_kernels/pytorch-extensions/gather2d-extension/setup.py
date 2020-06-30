from setuptools import setup, Extension
from torch.utils import cpp_extension
import os
setup(name='gather2d_cpp',
      ext_modules=[cpp_extension.CppExtension('gather2d_cpp', ['gather2d.cpp'],
      libraries=['habana_pytorch_plugin'],
      library_dirs=[os.environ['BUILD_ROOT_LATEST']])],
      cmdclass={'build_ext': cpp_extension.BuildExtension})
