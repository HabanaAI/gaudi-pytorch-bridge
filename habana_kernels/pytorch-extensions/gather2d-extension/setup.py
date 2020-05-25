from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name='gather2d_cpp',
      ext_modules=[cpp_extension.CppExtension('gather2d_cpp', ['gather2d.cpp'])],
      cmdclass={'build_ext': cpp_extension.BuildExtension})
