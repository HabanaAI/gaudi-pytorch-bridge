from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name='preproc_cpp',
      ext_modules=[cpp_extension.CppExtension('preproc_cpp', 
      sources=['preproc.cpp','preproc_main.cpp'],
      #extra_compile_args=['-std=c++11 -fpermissive -march=native -fopenmp'])],
      extra_cflags=['-std=c++11 -fopenmp -fpermissive'],
      extra_compile_args=['-g'])],
      cmdclass={'build_ext': cpp_extension.BuildExtension})

