###############################################################################
# Copyright (C) 2021-2023 Habana Labs, Ltd. an Intel Company
# All Rights Reserved.
#
# Unauthorized copying of this file or any element(s) within it, via any medium
# is strictly prohibited.
# This file contains Habana Labs, Ltd. proprietary and confidential information
# and is subject to the confidentiality and license agreements under which it
# was provided.
#
###############################################################################

import os
import shutil
import sys
from glob import glob

from setuptools import Extension
from setuptools.command.build_ext import build_ext
from setuptools.command.install_lib import install_lib


class SkipBuildExt(build_ext):
    def run(self):
        pass


def InstallCMakeLibs(module_namespace, wheel_name, wheel_build_dir, wheel_pt_vers, ignore_func):
    class _InstallCMakeLibs(install_lib):
        def __init__(self, dist):
            super().__init__(dist)

        def run(self):
            super(_InstallCMakeLibs, self).run()

        def install(self):
            installed_files = super(_InstallCMakeLibs, self).install()

            for ext in self.distribution.ext_modules:
                if isinstance(ext, PrebuiltPtExtension):
                    files = self._install_module()
                    if not installed_files:
                        installed_files = []
                    installed_files.extend(files)

            return installed_files

        def _install_module(self):
            install_dir = os.path.abspath(self.install_dir)

            pt_modules_bin_dir = os.path.join(install_dir, module_namespace)
            for pt_ver in wheel_pt_vers.split(","):
                shutil.copytree(
                    os.path.join(wheel_build_dir, f"pt{pt_ver.replace('.', '_')}", wheel_name),
                    os.path.join(pt_modules_bin_dir),  # TODO multiversion
                    ignore=ignore_func,
                    dirs_exist_ok=True,
                )
            return list(glob(install_dir + "/**", recursive=True))

    return _InstallCMakeLibs


class PrebuiltPtExtension(Extension):
    def __init__(self, name, build_dir):
        Extension.__init__(self, name, sources=[])
        self.build_dir = os.path.abspath(build_dir)


def get_version():
    version = os.getenv("RELEASE_VERSION")
    if not version:
        version = "0.0.0"
    build_number = os.getenv("RELEASE_BUILD_NUMBER")
    if build_number:
        return version + "." + build_number
    else:
        try:
            import subprocess

            root = os.environ["PYTORCH_MODULES_ROOT_PATH"]
            sha = subprocess.check_output(["git", "-C", root, "rev-parse", "--short", "HEAD"]).decode("ascii").strip()
            return f"{version}+git{sha}"
        except Exception as e:
            print("Error getting version: {}".format(e), file=sys.stderr)
            return f"{version}+unknown"
