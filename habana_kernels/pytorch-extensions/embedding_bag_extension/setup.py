from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name='HabanaEmbeddingBag_cpp',
      ext_modules=[cpp_extension.CppExtension('HabanaEmbeddingBag_cpp', ['embedding_bag_sum.cpp'])],
      cmdclass={'build_ext': cpp_extension.BuildExtension})
