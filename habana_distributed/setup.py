# Copyright (c) 2021, Habana Labs Ltd.  All rights reserved.
import os
import sys
import pathlib
import shutil
import multiprocessing
from subprocess import check_call, check_output

from torch.utils.cpp_extension import include_paths, library_paths
from setuptools import setup, Extension, distutils
from setuptools.command.build_ext import build_ext
from distutils.command.clean import clean
from tools.setup.cmake import CMakeExtension
import subprocess
# Constant known variables used throughout this file
CWD = os.path.dirname(os.path.abspath(__file__))
TORCH_HCL_PATH = os.path.join(CWD, "habana_torch_hcl")

def check_file(f):
    if not os.path.exists(f):
        print("Could not find {}".format(f))
        print("Did you run 'git submodule update --init --recursive'?")
        sys.exit(1)

class BuildCMakeExt(build_ext):
    """
    Builds using cmake instead of the python setuptools implicit build
    """
    def run(self):
        """
        Perform build_cmake before doing the 'normal' stuff
        """
        cmake_extensions = [ext for ext in self.extensions if isinstance(ext, CMakeExtension)]
        for ext in cmake_extensions:
            self.build_cmake(ext)

        self.extensions = [ext for ext in self.extensions if not isinstance(ext, CMakeExtension)]
        super(BuildCMakeExt, self).run()
        build_py = self.get_finalized_command('build_py')
        build_py.data_files = build_py._get_data_files()
        build_py.run()

    def build_cmake(self, extension: CMakeExtension):
        """
        The steps required to build the extension
        """
        build_dir = pathlib.Path('.'.join([self.build_temp, extension.name]))

        build_dir.mkdir(parents=True, exist_ok=True)
        install_dir = TORCH_HCL_PATH

        pybind11_path = os.path.join(os.environ['PYTORCH_MODULES_ROOT_PATH'], 'habana_distributed/third_party/pybind11')
        try:
            subprocess.check_call(["git", "-C", pybind11_path, 'checkout', 'v2.4.3'])
        except subprocess.CalledProcessError:
            print('git checkout failed')
        # Now that the necessary directories are created, build
        my_env = os.environ.copy()
        include_path_values = include_paths()
        include_path_values = [path for path in include_path_values if not path.endswith("THC")]
        include_path_values.append(os.environ['PYTORCH_MODULES_ROOT_PATH'] + "/habana_distributed/third_party/pybind11/include")
        build_options = {
            # The value cannot be easily obtained in CMakeLists.txt.
            'PYTHON_INCLUDE_DIRS': str(distutils.sysconfig.get_python_inc()),
            'PYTORCH_INCLUDE_DIRS': CMakeExtension.convert_cmake_dirs(include_path_values),
            'PYTORCH_LIBRARY_DIRS': CMakeExtension.convert_cmake_dirs(library_paths()),
        }

        extension.generate(build_options, my_env, build_dir, install_dir)

        max_jobs = os.getenv('MAX_JOBS', str(multiprocessing.cpu_count()))
        build_args = ['-j', max_jobs]
        check_call(['make', 'habana_torch_hcl'] + build_args, cwd=str(build_dir), env=my_env)
        check_call(['make', 'install'], cwd=str(build_dir), env=my_env)


class Clean(clean):
    def run(self):
        import glob
        import re
        shutil.rmtree(os.path.join(CWD, "build"), ignore_errors=True)

        clean.run(self)


def get_python_c_module():
    main_compile_args = []
    main_libraries = ['habana_torch_hcl']
    main_link_args = []
    main_sources = ["habana_torch_hcl/csrc/_C.cpp"]
    lib_path = os.path.join(TORCH_HCL_PATH, "lib")
    library_dirs = [lib_path]
    include_path = os.path.join(CWD, "src")
    include_dirs = include_paths()
    include_dirs.append(include_path)
    include_dirs.append(os.environ['PYTORCH_MODULES_ROOT_PATH'] + "/habana_distributed/third_party/pybind11/include")
    extra_link_args = []
    extra_compile_args = [
        '-Wall',
        '-Wextra',
        '-Wno-strict-overflow',
        '-Wno-unused-parameter',
        '-Wno-missing-field-initializers',
        '-Wno-write-strings',
        '-Wno-unknown-pragmas',
        # This is required for Python 2 declarations that are deprecated in 3.
        '-Wno-deprecated-declarations',
        # Python 2.6 requires -fno-strict-aliasing, see
        # http://legacy.python.org/dev/peps/pep-3123/
        # We also depend on it in our code (even Python 3).
        '-fno-strict-aliasing',
        # Clang has an unfixed bug leading to spurious missing
        # braces warnings, see
        # https://bugs.llvm.org/show_bug.cgi?id=21629
        '-Wno-missing-braces',
    ]

    def make_relative_rpath(path):
        return '-Wl,-rpath,$ORIGIN/' + path

    _c_module = Extension("habana_torch_hcl._C",
                          libraries=main_libraries,
                          sources=main_sources,
                          language='c',
                          extra_compile_args=main_compile_args + extra_compile_args,
                          include_dirs=include_dirs,
                          library_dirs=library_dirs,
                          extra_link_args=extra_link_args + main_link_args + [make_relative_rpath('lib')])

    return _c_module


if __name__ == '__main__':
    #version = create_version()
    c_module = get_python_c_module()
    cmake_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "CMakeLists.txt")
    modules = [CMakeExtension("libhabana_torch_hcl", cmake_file), c_module]
    setup(
        name='habana_torch_hcl',
        #version=version,
        ext_modules=modules,
        packages=['habana_torch_hcl'],
        #install_requires=['torch'],
        package_data={
            'habana_torch_hcl': [
                '*.py',
                '*/*.h',
                '*/*.hpp',
                'lib/*.so*',
                'bin/*',
                'env/*',
                'etc/*',
                'examples/*',
                'include/native_device_api/*.h*',
                'include/native_device_api/l0/*.h*',
                'include/*.h*',
                'lib/lib*',
                'lib/prov/lib*',
                'licensing/*',
                'modulefiles/*',
            ]},
        cmdclass={
            'build_ext': BuildCMakeExt,
            'clean': Clean,
        }
    )
