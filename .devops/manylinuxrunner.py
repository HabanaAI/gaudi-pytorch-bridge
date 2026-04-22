#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2025-2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
###############################################################################

import argparse
import logging
import os
import shutil
import subprocess as sp  # nosec
import sys
from abc import abstractmethod
from collections.abc import Iterable

from build_profiles.version import Version
from icecc_utils import ensure_icecc_setup

log = logging.getLogger(__file__)


class GenericManylinuxRunner:
    DEFAULT_PY_VERSIONS = {Version(sys.version_info)}

    def __init__(self, with_icecc=False, py_versions=DEFAULT_PY_VERSIONS):
        self.policy = "manylinux_2_28"
        self.arch = "x86_64"
        self.py_versions = py_versions
        self.with_icecc = with_icecc
        if with_icecc:
            self.image_name = f"artifactory-kfs.habana-labs.com/docker/manylinux/{self.policy}-with-icecc"
        else:
            self.image_name = f"artifactory-kfs.habana-labs.com/docker/manylinux/{self.policy}"

    def run(self, recreate_venv: bool, raw_args: Iterable[str]):
        log.info("Performing a manylinux build")
        self.pull_manylinux_container()

        for py_version in self.py_versions:
            self.rerun_build_in_manylinux(recreate_venv, raw_args, py_version)

    def pull_manylinux_container(self):
        log.debug(f"Pulling {self.image_name} Docker image")
        sp.check_call(f"docker pull {self.image_name}".split())

    def rerun_build_in_manylinux(self, recreate_venv: bool, raw_args: Iterable[str], py_version: Version):
        venv_base_dir = os.path.join(os.environ["HOME"], ".venvs")
        manylinux_venvs_dir = os.path.join(venv_base_dir, self.policy)
        os.makedirs(manylinux_venvs_dir, exist_ok=True)
        manylinux_pip_cache_dir = os.path.join(manylinux_venvs_dir, "cache", "pip")
        os.makedirs(manylinux_pip_cache_dir, exist_ok=True)
        ccache_dir = os.path.join(os.environ["HOME"], ".cache/manylinux/ccache")
        os.makedirs(ccache_dir, exist_ok=True)

        release_build_number = os.environ.get("RELEASE_BUILD_NUMBER", "")
        proxy_keys = " -e ".join(f"{k}={os.environ[k]}" for k in os.environ if "proxy" in k.lower())
        proxy_keys = f" -e {proxy_keys}" if proxy_keys else ""

        host_dir_for_venv = os.path.join(manylinux_venvs_dir, f"py{py_version}")
        if recreate_venv:
            log.info("Recreating main (mounted) manylinux virtual environment")
            shutil.rmtree(host_dir_for_venv, ignore_errors=True)
        os.makedirs(host_dir_for_venv, exist_ok=True)

        docker_venv_dir = os.path.join(os.environ["HOME"], ".venv")
        if not os.path.exists(docker_venv_dir):
            log.info("Symlinking $HOST/.venv to the main manylinux virtual environment to enable C++ test execution")
            os.symlink(host_dir_for_venv, docker_venv_dir, target_is_directory=True)
        elif not os.path.samefile(os.path.realpath(docker_venv_dir), host_dir_for_venv):
            log.warning(
                "Warning: $HOST/.venv is not the same as the main manylinux virtual environment. "
                "You might have issues with C++ tests execution."
            )

        interactive = "-it" if os.isatty(sys.stdin.fileno()) else ""
        options = (
            f" -e AUDITWHEEL_ARCH={self.arch}"
            f" -e AUDITWHEEL_POLICY={self.policy}"
            f" -e AUDITWHEEL_PLAT={self.policy}_{self.arch}"
            f" -e PLAT={self.policy}_{self.arch}"
            f" -e HABANA_PYTHON_VERSION={py_version}"
            f" -e HOST_USER={os.environ['USER']}"
            f" -e HOST_UID={os.getuid()}"
            f" -e HOST_GID={os.getgid()}"
            f" -e IN_MANYLINUX_ENV=1"
            f" -e RELEASE_BUILD_NUMBER={release_build_number}"
            f"{proxy_keys}"
            f" -v ~/.ssh:/.ssh-host:ro"
            f" -v {os.environ['HABANA_SOFTWARE_STACK']}:{os.environ['HABANA_SOFTWARE_STACK']}"
            f" -e _STACK={os.environ['HABANA_SOFTWARE_STACK']}"
            f" -v {os.environ['BUILD_ROOT']}:{os.environ['BUILD_ROOT']}"
            f" -e _BUILD={os.environ['BUILD_ROOT']}"
            f" -v {manylinux_venvs_dir}:$HOME/.venvs"
            f" -v {host_dir_for_venv}:$HOME/.venv"
            f" -v {manylinux_pip_cache_dir}:$HOME/.cache/pip"  # for faster venv restoration
            f" -v {ccache_dir}:$HOME/.ccache "
        )
        if self.with_icecc:
            options += (
                " --net=host -p ::10246/tcp -p ::8765/tcp -p ::8766/tcp -p ::8765/udp"
                " -e CCACHE_PREFIX=icecc -e CCACHE_PREFIX_CPP=icecc -e CCACHE_DEPEND=true -e ICECC_REMOTE_CPP=1"
            )

        command = (
            f"docker run --rm {interactive}{options} {self._get_memory_limit_flag()} {self.image_name} "
            + self.get_bash_command(raw_args)
        )

        log.debug(f"Running command: {command}")
        sp.check_call(command, shell=True)  # noqa S602

    @abstractmethod
    def get_bash_command(self, raw_args: Iterable[str]):
        pass

    @staticmethod
    def _get_memory_limit_flag():
        """Dockerized build that triggers kernel OOM can bring down the whole
        system. This may happen easily when too many build jobs are set with
        -j flag. To prevent this try limiting memory for docker build to 90%
        of current free memory.
        """

        try:
            with open("/proc/meminfo") as f:
                free_memory = next(line for line in f.readlines() if "MemAvailable" in line).strip().split(" ")
            assert free_memory[-1] == "kB", "unexpected memory unit in procinfo"
            memory_limit = int(0.9 * int(free_memory[-2])) // 1024
            return f"--memory={memory_limit}m"
        except OSError:
            log.warning("Failed to determine free system memory, build will run without max memory restriction.")
            return ""


class DefaultManyLinuxRunner(GenericManylinuxRunner):
    def get_bash_command(self, raw_args: Iterable[str]):
        return " ".join(raw_args)


def parse_args():
    from build import supported_python_versions

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--use-icecc",
        action="store_true",
        help="Build with icecc (distributed compilation). Currently only does something combined with --manylinux argument",
    )
    parser.add_argument(
        "--python-versions",
        "--py-versions",
        choices=supported_python_versions + ("all", "current"),
        nargs="+",
        default=("current",),
        help="Python versions to include. By default this option is set to "
        "'current' to only build for the system-supplied python3 (ATM it's "
        f"{Version(sys.version_info)})",
    )
    args, bash_command = parser.parse_known_args()

    return args, bash_command


def main():
    args, bash_command = parse_args()

    if args.use_icecc:
        ensure_icecc_setup()

    DefaultManyLinuxRunner(with_icecc=args.use_icecc, py_versions=args.python_versions).run(False, bash_command)
    sys.exit()


if __name__ == "__main__":
    main()
