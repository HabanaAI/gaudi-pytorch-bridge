#!/usr/bin/env python

from setuptools import setup, find_packages
from torch.utils.cpp_extension import CppExtension
import os

def _check_env_flag(name, default=''):
  return os.getenv(name, default).upper() in ['ON', '1', 'YES', 'TRUE', 'Y']

root = os.environ['PYTORCH_MODULES_ROOT_PATH']

DEBUG = _check_env_flag('DEBUG')

include_dirs = [
    os.path.join(root, 'pytorch_helpers'),
    os.path.join(os.environ['SYNAPSE_ROOT'], 'include'),
    os.path.join(os.environ['THIRD_PARTIES_ROOT'], 'abseil-cpp'),
    os.path.join(os.environ['THIRD_PARTIES_ROOT'], 'pybind11', 'include'),
]

libraries = [
    'habana_pytorch_plugin',
]

csrc = [
    os.path.join(root, 'habana_lazy/hblazy/csrc/bindings.cpp'),
]

extra_compile_args = [
    '-std=c++14',
    '-DMAX_DEVICES_PER_BOX=8',
]
extra_link_args = []

if DEBUG:
  extra_compile_args += ['-O0', '-g']
  extra_link_args += ['-O0', '-g']
else:
  extra_compile_args += ['-DNDEBUG', '-g0']

setup(
    name            = 'torch_habana_lazy',
    version         = "0.0.1",
    zip_safe        = False,
    packages        = find_packages(exclude=['build']),
    ext_modules     = [CppExtension(
        name            = '_hblazy',
        sources         = csrc,
        language        = 'c++',
        include_dirs    = include_dirs,
        libraries       = libraries,
        library_dirs    = [os.environ['BUILD_ROOT_LATEST']],
        extra_compile_args = extra_compile_args,
    )],
)
