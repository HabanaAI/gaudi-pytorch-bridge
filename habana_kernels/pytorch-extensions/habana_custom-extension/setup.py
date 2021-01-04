from setuptools import find_packages, setup, Extension
from torch.utils import cpp_extension
import os
setup(
    name='hb_custom',
    ext_modules=[cpp_extension.CppExtension(
        'hb_custom_C', ['csrc/hb_custom.cpp'],
        libraries=['habana_pytorch_plugin'],
        library_dirs=[os.environ['BUILD_ROOT_LATEST']])],
    include_dirs=[os.environ['PYTORCH_MODULES_ROOT_PATH'] + "/third_party/pybind11/include"],
    cmdclass={'build_ext': cpp_extension.BuildExtension},
    packages=find_packages(
        exclude=(
            'build',
            'csrc',
            'tests',
            'dist',
            'tests',
            'hb_custom.egg-info',))
)

