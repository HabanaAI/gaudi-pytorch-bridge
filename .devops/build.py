#!/usr/bin/env python3
###############################################################################
# Copyright (C) 2021-2022 Habana Labs, Ltd. an Intel Company
# All Rights Reserved.
#
# Unauthorized copying of this file or any element(s) within it, via any medium
# is strictly prohibited.
# This file contains Habana Labs, Ltd. proprietary and confidential information
# and is subject to the confidentiality and license agreements under which it
# was provided.
#
###############################################################################

import subprocess as sp
import argparse
import sys
import shutil
import os
import glob
import json
import logging
import tempfile
from collections import namedtuple, defaultdict
from contextlib import contextmanager
from build_profiles import profile_getter
from build_profiles.version import (
    Version,
    is_official_stable_cpu_version,
    is_pt_fork_version,
    is_wheel_version,
)
from typing import Dict, Iterable, List, Optional, Set, Tuple, Union, Any
from io import StringIO

import inspect

log = logging.getLogger(__file__)

BuildEnv = namedtuple("BuildEnv", ["py_ver", "pt_ver", "venv_dir", "optional"])
WheelConfig = namedtuple(
    "WheelConfig",
    [
        "target",
        "full_wheel_name",
        "py_ver",
        "pt_vers",
        "optional",
        "file_path_pattern",
        "source_dir",
    ],
)
venv_base_dir = os.path.join(os.environ["HOME"], ".venvs")


supported_pt_versions = tuple(
    map(lambda ver: Version(ver), profile_getter.get_available_versions())
)

recommended_pt_version = Version(profile_getter.get_version_literal("current"))

supported_python_versions = (
    Version("3.8"),
    Version("3.10"),
)

min_venv_python = supported_python_versions[0]
min_pip_version = Version("19.3.1")
build_py = os.path.realpath(__file__)

build_root = os.environ.get("BUILD_ROOT", None)
if not build_root:
    log.fatal(f"$BUILD_ROOT not set or is empty.")
    sys.exit(1)

build_dir_suffix = "pytorch_modules_multi_build"
build_dir = os.path.join(build_root, build_dir_suffix)

default_job_count = len(os.sched_getaffinity(0))


def call_with_error_logging(cmd):
    with tempfile.TemporaryFile() as tmp_out, tempfile.TemporaryFile() as tmp_err:
        try:
            return sp.call(cmd.split(), stdout=tmp_out, stderr=tmp_err)
        except Exception:
            log.error(f"Unexpected error when calling `{cmd}`:")
            log.error("stdout:")
            log.error(tmp_out.readlines())
            log.error("stderr:")
            log.error(tmp_err.readlines())
            raise


def ensure_icecc_setup():
    if "Ubuntu 20.04" not in sp.check_output(
        "lsb_release -d".split(), encoding="ascii"
    ):
        log.fatal("--use-icecc flag only supported for Ubuntu 20.04")
        sys.exit(1)

    icecc_installed = call_with_error_logging("dpkg -s icecc") == 0

    if icecc_installed:
        ensure_iceccd_started()
    else:
        log.info("icecc not installed. Installing and doing setup...")
        sp.check_call("sudo apt update".split())
        sp.check_call("sudo apt install icecc -y".split())
        sp.check_call(
            [
                "sudo",
                "sed",
                "-i",
                's/ICECC_NICE_LEVEL="5"/ICECC_NICE_LEVEL="10"/',
                "/etc/icecc/icecc.conf",
            ]
        )
        sp.check_call("sudo systemctl restart iceccd".split())


def ensure_iceccd_started():
    iceccd_stopped = call_with_error_logging("systemctl status iceccd") != 0

    if iceccd_stopped:
        log.info("iceccd was stopped. Trying to start it...")
        sp.check_call("sudo systemctl start iceccd".split())


def get_release_version():
    ver_str = []
    with open(os.path.join(os.getenv("SPECS_EXT_ROOT"), "version.h"), "r") as file:
        for line in file:
            if (
                "HL_DRIVER_MAJOR" in line
                or "HL_DRIVER_MINOR" in line
                or "HL_DRIVER_PATCHLEVEL" in line
            ):
                ver_str += [s for s in line.split() if s.isdigit()]
    if not ver_str:
        raise Exception("Could not retrieve version")
    if len(ver_str) < 3:
        raise Exception(
            f"Version table too small. Something was not retrieved: {ver_str}"
        )

    return ".".join(ver_str)


def get_supported_version(
    candidate, supported_list: Iterable[Version]
) -> Optional[Version]:
    for supported in supported_list:
        if supported == "nightly":
            if "dev" in str(candidate):
                return supported
            continue
        if supported.significant_matches(candidate):
            log.debug(f"Matched supported version: {supported}")
            return supported
    return None


@contextmanager
def env_var(var, val):
    current = os.environ.get(var, "")
    os.environ[var] = val
    yield
    os.environ[var] = current


@contextmanager
def chdir(path):
    pwd = os.getcwd()
    os.chdir(path)
    yield
    os.chdir(pwd)


@contextmanager
def elapsed_time_logger():
    import time
    from datetime import timedelta

    start = time.time()
    yield
    delta = timedelta(seconds=time.time() - start)
    log.info(f"Elapsed time: {delta}")


def prepare_env(venv_dir):
    """
    Prepares env dict for running a command as if a virtual environment was active.
    :param venv_dir
        When 'None' only attempt to deactivate current virtualenv.
        When string '.' keep current env (do nothing, return None).
        Otherwise assume venv_dir is a directory path and switch virtual env there.
    """

    if venv_dir == ".":
        return None
    env = dict()
    env.update(os.environ)
    venv = env.get("VIRTUAL_ENV", None)

    #  ordinary venv activate/deactivate use _OLD_VIRTUAL_* variables to
    #  restore the original env, but these variables aren't exported. This
    #  function looks for VIRTUAL_ENV variable and filters-out all PATH
    #  components pointing to current venv.
    if venv:
        path = filter(lambda p: not p.startswith(venv), env.get("PATH", "").split(":"))
        env["PATH"] = ":".join(path)

    if "PYTHONHOME" in env:
        del env["PYTHONHOME"]
    if venv_dir:
        env["PATH"] = venv_dir + "/bin:" + env.get("PATH", "")
        env["VIRTUAL_ENV"] = venv_dir
    return env


def run(*args, venv=".") -> None:
    log.info(f"In venv {venv} calling `{' '.join(args)}`")
    log.debug(f"^^^ called by {inspect.stack()[1].function} at {inspect.stack()[1].filename}:{inspect.stack()[1].lineno}")
    # must run through shell because otherwise changing PATH has no effect
    sp.check_call(
        " ".join(args), env=prepare_env(venv), shell=True, executable="/bin/bash"
    )


def outof(*args, venv=".") -> str:
    log.debug(f"In {venv} capturing output of `{' '.join(args)}`")
    # must run through shell because otherwise changing PATH has no effect
    result = sp.check_output(
        " ".join(args), encoding="ascii", env=prepare_env(venv), shell=True
    )
    log.debug(f"====\n{result}====")
    return result


def remove_venv(venv_dir):
    log.info(f"Removing virtual environment at {venv_dir} as requested")

    if os.path.islink(venv_dir):
        log.info(
            f"Virtual environment at {venv_dir} is just a link, removing " f"fearlessly"
        )
        os.remove(venv_dir)
        return
    if os.path.isdir(venv_dir):
        log.warning(f"Removing directory '{venv_dir}'.")
        shutil.rmtree(venv_dir)
    else:
        log.info(f"Not a directory: {venv_dir}")


def rm_link_or_dir(location):
    if os.path.islink(location):
        os.remove(location)
    elif os.path.isdir(location):
        shutil.rmtree(location)
    elif os.path.exists(location):
        log.error(f"Cannot remove {location}")
        sys.exit(1)


class RecreateVenv:
    FORCE = "force"
    AS_NEEDED = "as_needed"
    NEVER = "never"

    @staticmethod
    def choices():
        return RecreateVenv.FORCE, RecreateVenv.AS_NEEDED, RecreateVenv.NEVER


def query_installed_pt_ver(venv_dir, venv_python, label=None) -> Optional[Version]:
    verbose = " --verbose" if log.isEnabledFor(logging.DEBUG) else ""
    installed_pt_ver = outof(
        venv_python, build_py, "--get-pt-version" + verbose, venv=venv_dir
    ).strip()
    if installed_pt_ver == "None":
        return None
    return Version(installed_pt_ver, label=label)


# TODO: support RC builds
def resolve_pip_args(pt_ver: Union[str, Version]) -> Tuple[str, ...]:
    """Returns a tuple with pip arguments required for installing the PT wheel."""
    if pt_ver == "nightly":
        return "--pre", "--extra-index-url", "https://download.pytorch.org/whl/nightly/cpu"

    if is_wheel_version(pt_ver) or is_pt_fork_version(pt_ver):
        return tuple()
    elif is_official_stable_cpu_version(pt_ver):
        return "--extra-index-url", "https://download.pytorch.org/whl/cpu"
    else:
        raise ValueError(
            f"Unable to deduce a supported build type from PT version: {pt_ver}"
        )


def install_pt(pt_ver, venv_python, venv_dir, user):
    if is_wheel_version(pt_ver):
        required_pt = pt_ver.wheel_path
    else:
        required_pt = profile_getter.get_required_pt(
            pt_ver, profile_getter.RequirementPurpose.BUILD
        )  # e.g. 'torch==1.12.0'
        # TODO: if PT from fork: fetch it from artifactory or build_pt_fork(venv_python)
        if is_pt_fork_version(pt_ver):
            from os.path import expanduser
            home = expanduser("~")
            required_pt = os.path.join(home, "wheelhouse", "torch", "torch-1.12.0a0+git314b0ed-cp38-cp38-linux_x86_64.whl")
            if not os.path.isfile(required_pt):
                raise FileNotFoundError("Please download or compile PT-fork so you have it at: " + required_pt)
    version_specific_pip_args = resolve_pip_args(pt_ver)
    run(
        venv_python,
        "-m",
        "pip",
        "install",
        "-U",
        *user,
        required_pt,
        *version_specific_pip_args,
        venv=venv_dir,
    )


def install_requirements(pt_modules_root, pt_ver, venv_dir, venv_python, label=None):
    user = tuple()
    if venv_dir is None:
        user = ("--user",)

    run(
        venv_python,
        "-m",
        "pip",
        "install",
        "-U",
        *user,
        "-r",
        f"{pt_modules_root}/requirements.txt",
        venv=venv_dir,
    )

    install_pt(pt_ver, venv_python, venv_dir, user)

    # TODO: support parallel builds with different kinetos/pybinds

    kineto_root = os.environ["KINETO_ROOT"]
    if kineto_root:
        log.info("git submodule update for Kineto")
        run(f"cd {kineto_root} && git submodule update --init --recursive")

    # TODO: where to get Kineto from for non-PT-fork builds?

    # # Chances are system yarn won't work for you. You can try installing a
    # # working one using brew or from other sources. U20's default version (0.X)
    # # doesn't know the install command.
    # run(
    #     f"source {os.environ['PYTORCH_MODULES_ROOT_PATH']}/.ci/scripts/build.sh && ",
    #     "__get_func_name() { true; } && __check_mandatory_pkgs() { true; } &&"
    #     "__running_in_venv() { " + str(venv_dir is not None).lower() + "; } && "
    #     f"build_pytorch_tb_plugin --install -c",
    #     venv=venv_dir,
    # )

    log.info("git submodule update for pybind11")
    run(
        f"cd {os.environ['PYTORCH_MODULES_ROOT_PATH']} &&",
        "git submodule sync && ",
        "git submodule update --init --recursive",
    )

    return query_installed_pt_ver(venv_dir, venv_python, label=label)


def prepare_venv(
    python_ver: Version,
    pt_ver: Union[str, Version],
    pt_modules_root: str,
    recreate_venv=RecreateVenv.AS_NEEDED,
):
    """Prepares virtual environment to build PyTorch modules against the
    given PyTorch and Python version.

    :param python_ver  Python version
    :param pt_ver  PyTorch version as either Version or a symbolic label such
                   as "nightly".
    :param pt_modules_root  pytorch-integration root dir
    :param recreate_venv  Policy for virtual env reuse/rebuild
    :return (venv_dir, pt_ver) tuple with directory of the venv and Version
                               object containing the actual PT version offered
                               by this venv.
                               In case input pt_ver is symbolic (e.g.
                               'nightly') then this symbol is preserved as the
                               label property of the returned Version.
                               Eventually label property is used as virtual
                               environment location. This way "normal" builds
                               have venvs in .venv/.../ptx.y.z, but nightly is
                               always in .venv/.../nightly even though it's
                               exact version is updated on a daily basis.
    """

    venv_dir = os.path.join(
        venv_base_dir,
        profile_getter.get_required_pt_package_name(
            pt_ver, profile_getter.RequirementPurpose.BUILD
        ),
        f"py{python_ver}",
        f"pt{pt_ver}",
    )
    log.debug(f"Working on {venv_dir}")
    update = False
    if recreate_venv == RecreateVenv.FORCE:
        remove_venv(venv_dir)
    if not os.path.isdir(venv_dir):
        if recreate_venv == RecreateVenv.NEVER:
            log.fatal(
                f"Virtual env at {venv_dir} is missing, but recreating it was forbidden"
            )
            sys.exit(1)
        #  Create a new virtualenv making sure that system-level python3 is used.
        #  In testing a child venv made while a parent venv is active makes the
        #  child inherit all the parent's packages, possibly including a wrong
        #  PT installation.
        run(f"python{python_ver}", "-m", "venv", "--copies", venv_dir, venv=None)
        log.info(f"Created virtualenv at {venv_dir}")
        update = True
    else:
        log.debug(f"Virtual env at {venv_dir} already exists")
    venv_python = os.path.join(venv_dir, "bin", f"python{python_ver}")

    if not os.path.isfile(venv_python):
        log.fatal(f"Python interpreter f{venv_python} doesn't exist")
        sys.exit(1)

    v = outof(venv_python, "--version", venv=venv_dir).strip()
    v = v.split(" ")
    if v[0] != "Python":
        log.fatal(f"{venv_python} is not a Python interpreter!")
        sys.exit(1)
    v = Version(v[1])
    if min(min_venv_python, v) != min_venv_python:
        log.fatal(f"{venv_python} version is not supported ({v})")
        sys.exit(1)

    pip_ver = Version(outof("pip", "--version", venv=venv_dir).split(" ")[1])
    if pip_ver < min_pip_version:
        if recreate_venv != RecreateVenv.NEVER:
            run("pip", "install", "-U", "pip", venv=venv_dir)
        else:
            log.error(
                "Insufficient pip version in virtual env that I was forbidden to update"
            )
    label = None
    if not update:
        if pt_ver in ("nightly",):
            update = True
            label = pt_ver
        installed_pt_ver = query_installed_pt_ver(venv_dir, venv_python, label=label)
        if installed_pt_ver is None:
            update = True

    if recreate_venv == RecreateVenv.NEVER:
        update = False

    if update:
        installed_pt_ver = install_requirements(
            pt_modules_root,
            pt_ver,
            venv_dir,
            venv_python,
            label=label,
        )
    else:
        installed_pt_ver = query_installed_pt_ver(venv_dir, venv_python, label=label)

    if installed_pt_ver is None or not get_supported_version(
        installed_pt_ver, (pt_ver,)
    ):
        log.error(
            f"PyTorch version in {venv_dir} has wrong version ({installed_pt_ver}) recreate_venv={recreate_venv}."
        )
        sys.exit(1)
    return venv_dir, installed_pt_ver


class WheelSpec:
    def __init__(self, **kwargs):
        if "serialized_spec" in kwargs:
            spec = kwargs.get("serialized_spec").split(":")
            self.wheel_name = spec[0]
            self.pt_versions = [
                ver if ver == "nightly" else Version(ver) for ver in spec[1].split(",")
            ]
            self.optional = (True if spec[2] == "optional" else False,)
            self.wheel_src_dir = spec[3]
        elif "wheel_name" in kwargs and "pt_versions" in kwargs:
            self.wheel_name = kwargs.get("wheel_name")
            self.pt_versions = kwargs.get("pt_versions")
            self.optional = False
            self.wheel_src_dir = kwargs.get("wheel_src_dir")
        else:
            log.error("Internal error: Incorrect signature in WheelSpec")
            sys.exit(1)


def parse_wheel_spec(wheel_spec):
    retval = list(map(lambda x: WheelSpec(serialized_spec=x), wheel_spec))
    whl_name_list = list(map(lambda x: x.wheel_name, retval))
    if len(whl_name_list) != len(set(whl_name_list)):
        raise RuntimeError("Duplicate wheel names detected in current configuration")
    return retval


def get_installed_packages():
    reqs = sp.check_output([sys.executable, "-m", "pip", "freeze"])
    return [r.decode().split("==")[0] for r in reqs.split()]


WheelNameAndSource = namedtuple("WheelTarget", ["wheel_name", "src_dir"])


def prepare_build_envs(
    py_versions,
    wheel_specs,
    pt_modules_root,
    current_python_version,
    current_pt_version=None,
    recreate_venv=RecreateVenv.AS_NEEDED,
) -> Dict[BuildEnv, List[WheelNameAndSource]]:
    """Based on selected Python versions and PT versions/wheel spec,
    prepares build environments needed to build all requested configurations.
    Also, for each build env we are mapping wheels, that should contain binaries
    produced in the respective build env.
    Args:
        py_versions: Iterable of requested Python versions
        wheel_specs: dict of PT versions requested per wheel
        pt_modules_root: pytorch-integration root directory
        current_python_version: Python version in current environment
        current_pt_version: PT version present in the current environment
        recreate_venv: enum which describes the behavior of venv creation
    Returns:
        result: dict {BuildEnvs: list(WheelTarget)}
    """
    result = defaultdict(list)
    created_venvs = dict()
    installed_packages = get_installed_packages()

    for wheel_spec in wheel_specs:
        for py_ver in py_versions:
            use_current_py = current_python_version and bool(
                get_supported_version(current_python_version, (py_ver,))
            )
            for pt_ver in wheel_spec.pt_versions:
                required_pt_package_name = profile_getter.get_required_pt_package_name(
                    pt_ver, profile_getter.RequirementPurpose.BUILD
                )
                use_current_pt = (
                    current_pt_version
                    and get_supported_version(current_pt_version, (pt_ver,))
                    and required_pt_package_name in installed_packages
                )
                if (
                    recreate_venv != RecreateVenv.FORCE
                    and use_current_pt
                    and use_current_py
                ):
                    log.info(
                        f"Need {required_pt_package_name}=={pt_ver} and python=={py_ver} and will "
                        f"use {required_pt_package_name} {current_pt_version} and python "
                        f"{current_python_version} from the current env. "
                    )
                    venv_dir = os.environ.get("VIRTUAL_ENV", ".")
                else:
                    venv_dir_key = (py_ver, required_pt_package_name, pt_ver)
                    if venv_dir_key not in created_venvs:
                        venv_dir, pt_ver = prepare_venv(
                            py_ver,
                            pt_ver,
                            pt_modules_root,
                            recreate_venv=recreate_venv,
                        )
                        created_venvs[venv_dir_key] = (venv_dir, pt_ver)
                    else:
                        venv_dir, pt_ver = created_venvs[venv_dir_key]
                build_env = BuildEnv(py_ver, pt_ver, venv_dir, wheel_spec.optional)
                result[build_env].append(
                    WheelNameAndSource(wheel_spec.wheel_name, wheel_spec.wheel_src_dir)
                )
                log.debug(f"Build env {build_env} ready")
    return result


def prepare_build_dirs(
    build_root_dir,
    wheels_per_build_envs,
    cmake_configurations,
    pt_modules_root,
    clean=False,
) -> Tuple[List, List[WheelConfig]]:
    """
    Prepares build dirs for requested configurations
    Args:
        build_root_dir: root directory in which all build dirs will be created
        wheels_per_build_envs: dict of list(wheel names) per BuildEnv. For a build env lists wheels that should contain built binaries
        cmake_configurations: CMake flags
        pt_modules_root : pytorch-integration root directory
        clean: whether to remove build directories to rebuild from scratch
    Returns: A tuple of 3 lists:
             - CMake build configurations,
             - wheel configurations,
    """
    cmake_build_configs = []

    os.makedirs(build_root_dir, exist_ok=True)

    whl_build_dir = f"{build_root_dir}/whl_build_dir"
    with chdir(build_root_dir), open("Makefile", "w") as makefile:
        if clean:
            remove_artifacts_directories(cmake_configurations, whl_build_dir)

        def pmake(*args):
            print(*args, file=makefile)

        envs = wheels_per_build_envs.keys()
        define_top_level_targets(envs, cmake_configurations, pmake)

        combinations = collect_build_combinations(
            wheels_per_build_envs, cmake_configurations
        )
        log.info(f"Preparing for the following builds: {combinations}")

        for build_envs, cmake_config in combinations:
            common_venv_build_env = build_envs[0]
            pt_ver_dir = common_venv_build_env.pt_ver.label.replace(".", "_")

            # needs to do explicit copy, to support multiple -DPYTHON_EXECUTABLE flags
            cmake_flags = CMakeFlags(cmake_configurations[cmake_config].copy())
            log.info(
                f"In {build_root_dir}, preparing {cmake_config} build for pt{common_venv_build_env.pt_ver} python{common_venv_build_env.py_ver}."
            )
            current_ver_build_dir = os.path.join(
                build_root_dir,
                target_reldir(
                    common_venv_build_env.py_ver,
                    common_venv_build_env.pt_ver.label,
                    cmake_config,
                ),
            )

            if clean:
                log.info(f"Removing {current_ver_build_dir} to reconfigure")
                rm_link_or_dir(current_ver_build_dir)

            os.makedirs(current_ver_build_dir, exist_ok=True)
            log.info(
                f"Building {cmake_config} pt{common_venv_build_env.pt_ver} python{common_venv_build_env.py_ver} in {current_ver_build_dir}"
            )

            optional = all([build_env.optional for build_env in build_envs])
            cmake_build_configs.append(
                (current_ver_build_dir, common_venv_build_env.venv_dir, optional)
            )
            with chdir(current_ver_build_dir):
                prepare_single_build_directory(
                    pt_modules_root,
                    clean,
                    whl_build_dir,
                    pmake,
                    build_envs,
                    cmake_config,
                    common_venv_build_env,
                    pt_ver_dir,
                    current_ver_build_dir,
                    cmake_flags,
                )

        wheel_configs = create_wheel_targets(
            wheels_per_build_envs, whl_build_dir, pmake
        )

        create_ctest_target(pmake)
        create_collect_binaries_target(
            pmake, wheels_per_build_envs, cmake_configurations, ("all",)
        )

    return cmake_build_configs, wheel_configs


def create_ctest_target(pmake):
    pmake(f"ctest: $(addsuffix /ctest,$(SUBNAMES))")


def target_reldir(py_ver, pt_ver, cmake_config, target=None):
    pt_package_name = profile_getter.get_required_pt_package_name(
        pt_ver, profile_getter.RequirementPurpose.BUILD
    )
    subdir = f"{pt_package_name}/py{py_ver}/pt{pt_ver}/{cmake_config}"
    return f"{subdir}/{target}" if target else subdir


def target_absdir(py_ver, pt_ver, cmake_config, target=None):
    return os.path.abspath(target_reldir(py_ver, pt_ver, cmake_config, target=target))


def create_collect_binaries_target(
    pmake, wheels_per_build_envs, cmake_configurations, targets
):
    """Using pmake produce gnu-makefile with the following dependency pattern:
    <target> <- $PYTORCH_MODULES_RELEASE_BUILD/<target> <- pytorch/py3.6/pt1.12.0a0/Release/<target>
                                                        <- pytorch/py3.6/pt1.12.0a0/Debug/<target>
    Also pt_modules_*_build/* recipes perform a cp of selected
    files from pt_modules_multi_build/* to pt_modules_*_build.

    Args:
        pmake: fn used to write Makefile
        wheels_per_build_envs: result
        cmake_configurations: cmake flags
        targets: Iterable of targets
    """
    py_ver = list(wheels_per_build_envs.keys())[0].py_ver
    lib_versions = set()
    for e in wheels_per_build_envs.keys():
        if e.py_ver == py_ver:
            lib_versions.add(e.pt_ver)

    log.info(
        f"Artifacts built for python{py_ver} will be used by collect binaries targets."
    )

    for cmake_config in cmake_configurations.keys():
        destination = os.environ[f"PYTORCH_MODULES_{cmake_config.upper()}_BUILD"]
        for target in targets:
            # all rules are phony because these are not actual files
            pmake(f".PHONY: {destination}/{target}")
            deps = " ".join(
                target_reldir(py_ver, pt_ver.label, cmake_config, target)
                for pt_ver in lib_versions
            )
            pmake(f"{destination}/{target}: {deps}")
            pmake(f"\tDESTINATION=$(dir $@);\\")
            pmake(f"\tset -x;\\")
            pmake(f"\trm -r $$DESTINATION;\\")
            pmake(f"\tmkdir -p $$DESTINATION && \\")
            for pt_ver in lib_versions:
                source = target_absdir(py_ver, pt_ver.label, cmake_config)
                pmake(
                    f'\techo "Copying {pt_ver} targets from {source} to $$DESTINATION"&&\\'
                )
                pmake(f"\tcp -f {source}/*.so.{pt_ver}* $$DESTINATION &&\\")
            source = target_absdir(py_ver, next(iter(lib_versions)).label, cmake_config)
            pmake(
                f'\techo "Copying remaining targets from {source} to $$DESTINATION"&&\\'
            )
            pmake(f"\tcp -f {source}/*.so* $$DESTINATION && \\")
            pmake(f"\tcp -f {source}/*.py $$DESTINATION && \\")
            cmake_config_upper = cmake_config.upper()
            pmake(
                '\tfind -D exec $${DESTINATION} "(" -name "*.so*" -o -name "*.py" ")" '
                '-a -not -name "libtorch.so*" '
                "-exec cp -fs {} "
                f"$$BUILD_ROOT_{cmake_config_upper} \;"
                " -exec cp -fs {} $$BUILD_ROOT_LATEST \; &&\\"
            )
            pmake(f"\ttrue")
            pmake(f"{target}: {destination}/{target}")


# TODO: linking to latest
# if building debug:
#     cp -fs $PYTORCH_MODULES_DEBUG_BUILD/*.so $BUILD_ROOT_DEBUG;
#     if [ -z "$__all" ]; then
#         cp -fs $PYTORCH_MODULES_DEBUG_BUILD/*.so $BUILD_ROOT_LATEST;
#     fi;
# if building release:
#     cp -fs $PYTORCH_MODULES_RELEASE_BUILD/*.so $BUILD_ROOT_RELEASE;
#     cp -fs $PYTORCH_MODULES_RELEASE_BUILD/*.so $BUILD_ROOT_LATEST;
# fi;


def create_wheel_target_for_single_python(
    pmake,
    py_ver,
    wheel_name_and_src,
    optional,
    whl_build_dir,
    serializer,
    pt_vers,
    venv_dir,
):
    pt_wheel_vers = ",".join(map(lambda x: str(x), pt_vers))

    wheel_name = wheel_name_and_src.wheel_name
    wheel_target = "wheel_" + wheel_name
    full_wheel_name = wheel_name
    whl_source_dir = wheel_name_and_src.src_dir
    activate = f"source {venv_dir}/bin/activate" if venv_dir != "." else "true"

    pmake(f".PHONY: {wheel_target}/linux")
    pmake(f"{wheel_target}/linux:\n\t")

    new_serializer = f"py{py_ver}/{wheel_name}/{wheel_target}/linux_serial"
    pmake(f".PHONY: py{py_ver}/{wheel_name}/{wheel_target}/linux {new_serializer}")
    pmake(
        f"py{py_ver}/{wheel_name}/{wheel_target}/linux {new_serializer}:"
        f"$(addsuffix /wheel_install, $(SUBNAMES_PY_{py_ver}_RELEASE))|${{PYTORCH_MODULES_RELEASE_BUILD}}/pkgs"
    )
    pmake(
        f"\t{'-' if optional else ''}cd $$PYTORCH_MODULES_ROOT_PATH/{whl_source_dir} && {activate} &&\\"
    )
    pmake(
        f'\tRELEASE_VERSION="{get_release_version()}" PT_WHEEL_VERS="{pt_wheel_vers}" PT_WHEEL_NAME="{full_wheel_name}" '
        f"PYTORCH_MODULES_WHL_BUILD_DIR={whl_build_dir}/py{py_ver} "  # TODO: use a separate build dir for each wheel version built (one per python)
        f"python3 setup.py --verbose bdist_wheel &&\\"
    )
    pmake(
        f"\tmv $$PYTORCH_MODULES_ROOT_PATH/{whl_source_dir}/dist/*.whl ${{PYTORCH_MODULES_RELEASE_BUILD}}/pkgs/"
    )

    pmake(f"{new_serializer}: {serializer}")

    expected_wheel_pattern = (
        f"{os.environ['PYTORCH_MODULES_RELEASE_BUILD']}/pkgs/"
        f"{full_wheel_name.replace('-', '_')}-*-cp{str(py_ver).replace('.', '')}*.whl"
    )

    # make final wheel(s) target depend on py-version specific parts
    pmake(f"{wheel_target}/linux: {new_serializer}")

    wheel_config = WheelConfig(
        wheel_target,
        full_wheel_name,
        py_ver,
        pt_vers,
        optional,
        expected_wheel_pattern,
        whl_source_dir,
    )

    return new_serializer, wheel_config


def create_wheel_targets(
    wheels_per_build_envs: Dict[BuildEnv, WheelNameAndSource], whl_build_dir, pmake
) -> List[WheelConfig]:
    """Returns a list of wheel configs to be built"""
    wheel_configs = []

    pt_vers_config = defaultdict(list)
    ref_venv_configs = dict()
    for e, wheels in wheels_per_build_envs.items():
        for wheel in wheels:
            pt_vers_config[(e.py_ver, wheel)].append(e.pt_ver)
            ref_venv_configs[(e.py_ver, wheel)] = (
                e.venv_dir,
                e.optional,
            )
    serializer = ""
    for key, val in ref_venv_configs.items():
        venv_dir, optional = val
        py_ver, wheel_name_and_src = key
        pt_vers = pt_vers_config[key]

        serializer, wheel_config = create_wheel_target_for_single_python(
            pmake,
            py_ver,
            wheel_name_and_src,
            optional,
            whl_build_dir,
            serializer,
            pt_vers,
            venv_dir,
        )
        wheel_configs.append(wheel_config)

    pmake(
        f"${{PYTORCH_MODULES_RELEASE_BUILD}} ${{PYTORCH_MODULES_DEBUG_BUILD}}:\n\tmkdir $@"
    )
    pmake(f"${{PYTORCH_MODULES_RELEASE_BUILD}}/pkgs:\n\tmkdir -p $@\n")

    create_wheel_finalization_target(wheel_configs, pmake)

    return wheel_configs


def create_wheel_finalization_target(wheel_configs: List[WheelConfig], pmake) -> str:
    """Puts wheels in wheelhouse and repairs them for manylinux"""

    wheelhouse = "wheelhouse"
    moving_target = add_target_for_moving_wheels_to_wheelhouse(
        wheel_configs, pmake, wheelhouse
    )
    return add_target_to_repair_all_wheels(pmake, moving_target, wheelhouse)


def add_target_for_moving_wheels_to_wheelhouse(wheel_configs, pmake, wheelhouse) -> str:
    linux_targets = [f"{wheel.target}/linux" for wheel in wheel_configs]
    wheel_files = " ".join(
        [f"{wheel_config.file_path_pattern}" for wheel_config in wheel_configs]
    )

    moving_target = "wheel/put_in_wheelhouse"
    pmake(f".PHONY: {moving_target}")
    pmake(f"{moving_target}: {' '.join(linux_targets)}")
    pmake(f"\tmkdir -p {wheelhouse} && \\")
    pmake(f"\tfind {wheelhouse} -type f -delete && \\")
    pmake(f"\tmv {wheel_files} {wheelhouse}")
    return moving_target


def add_target_to_repair_all_wheels(pmake, moving_target, wheelhouse) -> str:
    """Repairing wheels makes them manylinux ones"""
    target = "wheel/manylinux"
    pmake(f".PHONY: {target}")
    pmake(f"{target}: {moving_target}")
    pmake(
        f"\tfind {wheelhouse} -name '*-linux*.whl' -exec ${{PYTORCH_MODULES_ROOT_PATH}}/.devops/manylinux/repair_wheel.py "
        f"--wheel-dir=${{PYTORCH_MODULES_RELEASE_BUILD}} {{}} \\;"
    )
    return target


class CMakeFlags:
    """Helps modify CMake flags"""

    def __init__(self, flags: List[str]):
        self.flags = flags

    def __copy__(self):
        return CMakeFlags(self.flags.copy())

    def append_to_list(self, flag: str, value: str) -> None:
        """Appends a new value to flags storing CMake lists (separated by
        semicolons), e.g. CMAKE_PREFIX_PATH
        """
        self.override(flag, self[flag] + "\;" + value)

    def contains(self, flag: str) -> bool:
        return any(f for f in self.flags if CMakeFlags._flag_name_equals(f, flag))

    def insert(self, flag: str, value: str) -> None:
        self.flags.append(f"-D{flag}={value}")

    def remove(self, flag: str) -> None:
        self.flags = list(
            filter(lambda f: not CMakeFlags._flag_name_equals(f, flag), self.flags)
        )

    def override(self, flag: str, value: str) -> None:
        self.remove(flag)
        self.flags.append(f"-D{flag}={value}")

    def set_if_missing(self, flag: str, value: str):
        if not self.contains(flag):
            self.flags.append(f"-D{flag}={value}")

    def __getitem__(self, flag: str) -> str:
        item = list(filter(lambda f: CMakeFlags._flag_name_equals(f, flag), self.flags))
        assert len(item) < 2
        return item[0].split("=")[1:] if item else ""

    @staticmethod
    def _flag_name_equals(stored_flag: str, flag_name: str) -> bool:
        """Stored flag is in the format: `-Dflag_name=value`"""
        return stored_flag[2:].split("=")[0] == flag_name


def prepare_single_build_directory(
    pt_modules_root,
    clean,
    whl_build_dir,
    pmake,
    build_envs,
    cmake_config,
    common_venv_build_env,
    pt_ver_dir,
    current_ver_build_dir,
    cmake_flags: CMakeFlags,
):
    if clean or not os.path.exists(os.path.join(current_ver_build_dir, "Makefile")):
        run_cmake_build_generation(
            pt_modules_root, cmake_config, common_venv_build_env, cmake_flags
        )
        # emit implicit rule to pass target to a recursive make

    subtarget = target_reldir(
        common_venv_build_env.py_ver,
        common_venv_build_env.pt_ver.label,
        cmake_config,
    )
    activate = (
        f"source {common_venv_build_env.venv_dir}/bin/activate"
        if common_venv_build_env.venv_dir != "."
        else "true"
    )
    pmake(f".PHONY: {subtarget}/all {subtarget}/wheel {subtarget}/ctest")
    pmake(f"{subtarget}/all:")
    pmake(f"\t{activate} && $(MAKE) -C {current_ver_build_dir} $(notdir $@)")
    pmake(f"SUBNAMES += {subtarget}")
    for build_env in build_envs:
        pmake(f"SUBNAMES_PY_{build_env.py_ver}_{cmake_config.upper()} += {subtarget}")
    pmake(f"{subtarget}/wheel_install:")
    wheel_installs = [
        f"\t{'-' if build_env.optional else ''}$(MAKE) -C {current_ver_build_dir} DESTDIR={whl_build_dir}/py{build_env.py_ver}/pt{pt_ver_dir} install && echo $@ is finished"
        for build_env in build_envs
    ]
    pmake("\n".join(wheel_installs))
    pmake(f"{subtarget}/ctest: {subtarget}/all")
    pmake(
        f"\tcd {current_ver_build_dir} && {activate} && LD_LIBRARY_PATH={current_ver_build_dir}:$$LD_LIBRARY_PATH ctest --output-on-failure"
    )


def run_cmake_build_generation(
    pt_modules_root, cmake_config, common_venv_build_env, cmake_flags: CMakeFlags
):
    cmake_flags = add_python_env_flags(cmake_flags, common_venv_build_env)
    cmake_flags = append_cmake_torch_path(cmake_flags, common_venv_build_env)
    try:
        #  TODO: -DBUILD_PKGS=$__build_ext -DINSTALL_PKGS=$__install_ext pt_vers py_ver
        run(
            "cmake",
            pt_modules_root,
            *cmake_flags.flags,
            # TODO: "-GNinja",
            venv=common_venv_build_env.venv_dir,
        )
    except sp.CalledProcessError as e:
        log.fatal(
            f"CMake for pt{common_venv_build_env.pt_ver} python{common_venv_build_env.py_ver}, {cmake_config} failed with retcode {e.returncode}"
        )
        sys.exit(1)


def collect_build_combinations(
    wheels_per_build_envs, cmake_configurations
) -> List[Tuple[list, Any]]:
    """Returns a list of pairs: venv path and CMake flags"""
    build_envs_by_venv = defaultdict(list)
    for e in wheels_per_build_envs.keys():
        build_envs_by_venv[e.venv_dir].append(e)

    combinations = list(
        (e, cmake_config)
        for e in build_envs_by_venv.values()
        for cmake_config in cmake_configurations.keys()
    )
    log.debug(f"wheels_per_build_envs {wheels_per_build_envs}")
    log.debug(f"build_envs_by_venv {build_envs_by_venv}")
    return combinations


def define_top_level_targets(build_envs, cmake_configurations, pmake):
    pmake(f"# This file has been autogenerated with {__file__}")
    pmake(".SUFFIXES:")
    pmake("SUBNAMES =")
    pmake("SHELL := /bin/bash")
    #  TODO: proper debug build support in subnames
    pmake(
        "\n".join(
            f"SUBNAMES_PY_{e.py_ver}_{cmake_config.upper()} ="
            for e in build_envs
            for cmake_config in cmake_configurations.keys()
        )
    )

    pmake(".PHONY: all")
    # declare "all" first so it's the default target
    pmake("all:")


def remove_artifacts_directories(cmake_configurations, whl_build_dir):
    if os.path.exists(whl_build_dir):
        log.info(f"Cleaning {whl_build_dir}")
        shutil.rmtree(whl_build_dir)

    for config in cmake_configurations.keys():
        env = os.environ.get(f"PYTORCH_MODULES_{config.upper()}_BUILD", None)
        if env and os.path.exists(env):
            log.info(f"Cleaning {env}")
            shutil.rmtree(env)
            os.makedirs(env)


def build(
    build_dir,
    jobs=default_job_count,
    targets=("all",),
    verbose=False,
    extra_make_flags=tuple(),
    use_icecc=False,
):
    with chdir(build_dir):
        log.debug(f"extra_make_flags={extra_make_flags}")
        jobs = ("-j", str(jobs)) if jobs else tuple()
        verbose = ("VERBOSE=1",) if verbose >= 2 else tuple()
        use_icecc = ("CCACHE_PREFIX=icecc",) if use_icecc else tuple()
        try:
            run(
                *use_icecc,
                "make",
                "-C",
                build_dir,
                *jobs,
                *targets,
                *verbose,
                *extra_make_flags,
            )
        except sp.CalledProcessError as e:
            log.error(f"Compilation process failed with error code {e.returncode}")
            sys.exit(e.returncode)


def get_current_pt_version() -> Optional[Version]:
    """Figure out the PT version available in the current environment.
    This is used for an internal call as well as via a subprocess call to
    `build.py --get-pt-version` to probe virtual build environments.
    """
    log.debug(f"Python executable: {sys.executable}")
    try:
        sys.path = [path for path in sys.path if path != os.getcwd()]
        import torch as pt

        log.debug(f"PyTorch path: {pt.__path__}")
        return Version(pt.__version__)
    except ImportError as e:
        log.debug(e)
        return None


def print_current_pt_version_and_exit():
    log.name = "get-pt-version"
    print(get_current_pt_version())
    sys.exit(0)


def is_running_in_venv():
    native = hasattr(sys, "real_prefix") or (
        hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix
    )
    return native


def get_cmake_configurations(args) -> Dict[str, str]:
    """Compiles user-supplied cmd args into a mapping of configurations names
    and lists of CMake flags.
    Deals with conflicts like passing -DCMAKE_BUILD_TYPE=Debug together with -r,
    in which case Release wins.

    Produces dict <config>:[CMake flags...]
    """
    cmake_flags = CMakeFlags(args.cmake_flag if args.cmake_flag else [])
    if args.no_swig:
        cmake_flags.set_if_missing("SWIG", "")
    if args.no_tidy:
        cmake_flags.set_if_missing("CLANG_TIDY", "")
    if args.no_iwyu:
        cmake_flags.set_if_missing("IWYU", "")
    if args.sanitize:
        cmake_flags.set_if_missing("SANITIZER", "ON")
    if args.no_cpp_tests:
        cmake_flags.set_if_missing("BUILD_TESTS", "OFF")
    if args.upstream_compile:
        cmake_flags.set_if_missing("UPSTREAM_COMPILE", "ON")

    build_type = "CMAKE_BUILD_TYPE"
    debug = "Debug"
    release = "Release"

    if args.build_all:
        cmake_flags.remove(build_type)
        return {
            debug: cmake_flags.flags + [f"-D{build_type}={debug}"],
            release: cmake_flags.flags + [f"-D{build_type}={release}"],
        }
    elif args.release:
        cmake_flags.override(build_type, release)
        return {release: cmake_flags.flags}
    else:
        cmake_flags.set_if_missing(build_type, debug)
        return {debug: cmake_flags.flags}


def add_python_env_flags(cmake_flags: CMakeFlags, build_env: BuildEnv) -> CMakeFlags:
    venv_python = get_python_exec(build_env)

    cmake_flags.set_if_missing("PYTHON_EXECUTABLE", venv_python)

    py_include_dirs = outof(
        venv_python,
        "-c",
        '"from distutils.sysconfig import get_python_inc; ' 'print(get_python_inc())"',
        venv=build_env.venv_dir,
    ).strip()
    cmake_flags.set_if_missing("PYTHON_INCLUDE_DIR", py_include_dirs)

    py_lib = outof(
        venv_python,
        "-c",
        '"import distutils.sysconfig as sysconfig; '
        "print(sysconfig.get_config_var('LIBDIR'))\"",
        venv=build_env.venv_dir,
    ).strip()
    cmake_flags.set_if_missing("PYTHON_LIBRARY", py_lib)

    return cmake_flags


def append_cmake_torch_path(cmake_flags: CMakeFlags, build_env: BuildEnv) -> CMakeFlags:
    venv_python = get_python_exec(build_env)
    torch_path = outof(
        venv_python,
        "-c",
        '"import os, torch;',
        'print(os.path.dirname(torch.__file__))"',
        venv=build_env.venv_dir,
    ).strip()
    cmake_flags.append_to_list("CMAKE_PREFIX_PATH", torch_path)
    return cmake_flags


def get_python_exec(build_env):
    if build_env.venv_dir == ".":
        return shutil.which(f"python{build_env.py_ver}")
    else:
        return os.path.join(build_env.venv_dir, "bin", f"python{build_env.py_ver}")


def run_ctest_on_dirs(cmake_build_configs):
    for cfg in cmake_build_configs:
        # env_var is a WA to avoid linking against libtorch from latest/.
        # In latest/ there is a version just for one PT, and in each build dir
        # there is correct version per build
        with chdir(cfg[0]), env_var(
            "LD_LIBRARY_PATH", cfg[0] + ":" + os.environ.get("LD_LIBRARY_PATH", "")
        ):
            try:
                run("ctest", "--output-on-failure", venv=cfg[1])
            except Exception as e:
                if cfg[2]:
                    log.warning(
                        f"Failed to run ctest on optional build with error: {str(e)}"
                    )
                    continue
                else:
                    log.error(f"Failed to run ctest with error: {str(e)}")
                    sys.exit(1)


def install_wheel():
    if not is_running_in_venv():
        run("pip", "install", "--user", "wheel")
    else:
        run("pip", "install", "wheel")


system_python_version = Version(sys.version_info)


class SmartFormatter(argparse.RawDescriptionHelpFormatter):
    def _split_lines(self, text, width):
        if text.startswith("R|"):
            return text[2:].splitlines()
        # this is the RawTextHelpFormatter._split_lines
        return argparse.HelpFormatter._split_lines(self, text, width)


# TODO: ensure all args from original PT script are supported
def parse_args():
    parser = argparse.ArgumentParser(
        description="Build Habana PT modules for multiple PyTorch versions.",
        epilog=f"""
  Virtual environments
  --------------------
 This tool uses python virtual environments located at $HOME/.venv_<pt_version>
 to supply different versions of PyTorch binary for building. User may reuse
 existing virtual environments by symbolically linking them to these locations.
 In case a virtual env is not present, it will be created. In case a virtual env
 does not offer proper version of pytorch then it will be installed. The exact
 PT version to be installed is determined based on .devops/build_profiles/profiles.json
 If a virtual env offers invalid version of PT, then this script fails.
 For example assume user is building --pt_version=current, then:
     PT version available  | script result
     ----------------------+---------------------------------
                     none  | install pytorch=={profile_getter.get_version_literal("current")}
                    {profile_getter.get_version_literal("previous")} | use pytorch=={profile_getter.get_version_literal("previous")}
                    1.14   | fail

  Build directories
  -----------------
 This tool creates and uses multiple CMake build directories under
 {build_root}/{build_dir_suffix}/<pt_version>/<cmake_build_type>.
 Build artifacts are linked to $PYTORCH_MODULES_<cmake_build_type>_BUILD
 so that all targets with PT version suffix are taken from respective
 build_root subdirectory, and all PT-version-agnostic files are taken as
 compiled for the newest PT.""",
        formatter_class=SmartFormatter,
    )

    parser.add_argument(
        "--python-versions",
        choices=supported_python_versions + ("all", "current"),
        nargs="+",
        default="current",
        help="Python versions to include. By default this option is set to "
        "'current' to only build for the system-supplied python3 (ATM it's "
        f"{system_python_version})",
    )
    version_args = parser.add_mutually_exclusive_group()
    # Original --pt-version dropped as it's broken in master - likely unused
    version_args.add_argument(
        "--pt-versions",
        # choices=supported_pt_versions + ("all", "current", "nightly"),
        nargs="+",
        default="current",
        help="pt versions to include. By default this option is set to "
        "'current' to only build for the PyTorch version that is available"
        "in the current environment.",
    )
    version_args.add_argument(
        "--wheel-spec",
        action="store",
        nargs="+",
        help="R|Used to build multiple wheels. Specifies wheel suffix and which versions belong to that wheel.\n"
        "Format: --wheel-spec <whl-name1>:<ver1>,<ver2>,...:standard/optional\n"
        "Example: --wheel-spec habana_pytorch:2.5.2,current:standard habana_pytorch_internal:nightly:optional",
    )

    parser.add_argument(
        "-n",
        "--no-ext-build",
        action="store_true",
        help="Skip wheel build for extensions.",
    )

    # TODO: wheel installation flag support

    parser.add_argument(
        "--manylinux",
        action="store_true",
        help="Build in a pt-manylinux container instead of the current OS.",
    )
    parser.add_argument(
        "--use-icecc",
        action="store_true",
        help="Build with icecc (distributed compilation). Currently only does something combined with --manylinux argument",
    )
    parser.add_argument(
        "-c",
        "--configure",
        action="store_true",
        help="Configure before build. Also clear $PYTORCH_MODULES_RELEASE_BUILD, "
        "$PYTORCH_MODULES_DEBUG_BUILD or both depending on the selected "
        "configuration.",
    )
    parser.add_argument(
        "--get-pt-version",
        action="store_true",
        help="Only print current PyTorch version or 'None' if not installed",
    )
    parser.add_argument(
        "--recreate-venv",
        choices=RecreateVenv.choices(),
        default=RecreateVenv.AS_NEEDED,
        help="Recreate virtual environments used for building.",
    )

    parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        help="number of parallel jobs used for building. "
        f"The default value depends on system, here it's {default_job_count}.",
        default=default_job_count,
    )
    parser.add_argument(
        "-r",
        "--release",
        action="store_true",
        help="Build release configuration, ignore -DCMAKE_BUILD_TYPE if given",
    )
    parser.add_argument(
        "-a",
        "--build-all",
        action="store_true",
        help="Build both Debug and Release configurations, ignore -DCMAKE_BUILD_TYPE if given",
    )
    parser.add_argument(
        "-s", "--sanitize", action="store_true", help="Build with sanitizers"
    )
    parser.add_argument(
        "-l",
        "--no_cpp_tests",
        action="store_true",
        help="Don't build tests. Toggling between -l and full builds requires -c",
    )
    parser.add_argument(
        "--no-swig",
        action="store_true",
        help="Build without swig even if it's available",
    )
    parser.add_argument(
        "-y", "--no-tidy", action="store_true", help="Build without clang-tidy"
    )
    parser.add_argument(
        "--no-iwyu", action="store_true", help="Build without Include What You Use"
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Enable diagnostic output. Single -v shows output from build "
        "script, double -vv additionally enables printing of compilation "
        "command lines.",
    )

    parser.add_argument(
        "--cmake-flag",
        action="append",
        help="Args forwarded to CMake. "
        "Unless otherwise noted, when conflicting with flags imposed by other"
        "arguments, the effective setting is the one given explicitly.",
    )
    available_profiles = profile_getter.get_available_profiles()
    parser.add_argument(
        "--profile",
        action="store",
        choices=[*available_profiles],
        help="Load arguments from json file. All other arguments are ignored",
    )
    parser.add_argument(
        "--describe-profile",
        action="store_true",
        help="Debug command. Prints equivalent non-profile build.py invocation. Without --profile does nothing.",
    )
    parser.add_argument(
        "--run-ctest",
        action="store_true",
        help="(experimental) run CTest on every build",
    )
    parser.add_argument(
        "--upstream_compile",
        action="store_true",
        help="Compile for upstream workspace",
    )

    args = parser.parse_args()

    if args.profile:
        profile_args = profile_getter.get_args_for_profile(args.profile)
        raw_args = profile_args
        if args.describe_profile:
            print("build.py " + " ".join(profile_args))
            exit(0)
        args = parser.parse_args(args=profile_args)
    else:
        raw_args = sys.argv[1:]

    return args, raw_args


class ManylinuxRunner(object):
    def __init__(self, with_icecc=False):
        self.with_icecc = with_icecc
        if with_icecc:
            self.image_name = "artifactory-kfs.habana-labs.com/developers-docker-dev-local/pytorch-sigs/pt-manylinux-with-icecc"
        else:
            self.image_name = (
                "artifactory-kfs.habana-labs.com/docker/pytorch-sigs/pt-manylinux"
            )

    def run(self, args):
        log.info("Performing a manylinux build")
        self.pull_manylinux_container()
        self.rerun_build_in_manylinux(args)

    def pull_manylinux_container(self):
        log.debug(f"Pulling {self.image_name} Docker image")
        sp.check_call(f"docker pull {self.image_name}".split())

    def rerun_build_in_manylinux(self, args):
        if self.with_icecc:
            args.remove("--use-icecc")
        args.remove("--manylinux")

        manylinux_venvs_dir = os.path.join(venv_base_dir, "manylinux2014")
        os.makedirs(manylinux_venvs_dir, exist_ok=True)
        manylinux_pip_cache_dir = os.path.join(manylinux_venvs_dir, "cache", "pip")
        os.makedirs(manylinux_pip_cache_dir, exist_ok=True)
        ccache_dir = os.path.join(os.environ["HOME"], ".ccache")
        os.makedirs(ccache_dir, exist_ok=True)

        release_build_number = os.environ.get("RELEASE_BUILD_NUMBER", "")
        proxy_keys = " -e ".join(
            (f"{k}={os.environ[k]}" for k in os.environ if "proxy" in k.lower())
        )
        proxy_keys = f" -e {proxy_keys}" if proxy_keys else ""

        # Dockerized build that triggers kernel OOM can bring down the whole
        # system. This may happen easily when too many build jobs are set with
        # -j flag. To prevent this try limiting memory for docker build to 90%
        # of current free memory.

        memory_limit = ""
        try:
            free_memory = (
                next(
                    l for l in open("/proc/meminfo").readlines() if "MemAvailable" in l
                )
                .strip()
                .split(" ")
            )
            assert free_memory[-1] == "kB", "unexpected memory unit in procinfo"
            memory_limit = int(0.9 * int(free_memory[-2])) // 1024
            memory_limit = f"--memory={memory_limit}m"
        except:
            log.warning(
                "Failed to determine free system memory, build will run without max memory restriction."
            )

        options = (
            " -e PLAT=manylinux2014_x86_64"
            " -e AUDITWHEEL_ARCH=86_64"
            " -e AUDITWHEEL_PLAT=manylinux2014_x86_64"
            " -e AUDITWHEEL_POLICY=manylinux2014"
            f" -e PYTORCH_MODULES_RELEASE_BUILD={os.environ['PYTORCH_MODULES_RELEASE_BUILD']}"
            f" -e PYTORCH_MODULES_DEBUG_BUILD={os.environ['PYTORCH_MODULES_DEBUG_BUILD']}"
            f" -e PYTORCH_MODULES_ROOT_PATH={os.environ['PYTORCH_MODULES_ROOT_PATH']}"
            f" -e PYTHONPATH={os.environ['PYTORCH_MODULES_ROOT_PATH']}/python"
            f" -e HABANA_SOFTWARE_STACK={os.environ['HABANA_SOFTWARE_STACK']}"
            f" -e BUILD_ROOT={os.environ['BUILD_ROOT']}"
            f" -e THIRD_PARTIES_ROOT={os.environ['THIRD_PARTIES_ROOT']}"
            f" -e SYNAPSE_ROOT={os.environ['SYNAPSE_ROOT']}"
            f" -e HCL_ROOT={os.environ['HCL_ROOT']}"
            f" -e MEDIA_ROOT={os.environ['MEDIA_ROOT']}"
            f" -e CODEC_ROOT={os.environ['CODEC_ROOT']}"
            f" -e SPECS_EXT_ROOT={os.environ['SPECS_EXT_ROOT']}"
            f" -e BUILD_ROOT_LATEST={os.environ['BUILD_ROOT_LATEST']}"
            f" -e BUILD_ROOT_RELEASE={os.environ['BUILD_ROOT_RELEASE']}"
            f" -e BUILD_ROOT_DEBUG={os.environ['BUILD_ROOT_DEBUG']}"
            f" -e HOST_USER={os.environ['USER']}"
            f" -e HOST_UID={os.getuid()}"
            f" -e HOST_GID={os.getgid()}"
            f" -e RELEASE_BUILD_NUMBER={release_build_number}"
            f"{proxy_keys}"
            f" -v ~/.ssh:/.ssh-host:ro"
            f" -v {os.environ['HABANA_SOFTWARE_STACK']}:{os.environ['HABANA_SOFTWARE_STACK']}"
            f" -v {os.environ['BUILD_ROOT']}:{os.environ['BUILD_ROOT']}"
            f" -v {manylinux_venvs_dir}:$HOME/.venvs"
            f" -v {manylinux_pip_cache_dir}:$HOME/.cache/pip"  # for faster venv restoration
            f" -v {ccache_dir}:$HOME/.ccache "
        )
        if self.with_icecc:
            options = (
                options + " --net=host"
                " -p ::10246/tcp -p ::8765/tcp -p ::8766/tcp -p ::8765/udp"
                " -e CCACHE_PREFIX=icecc"
            )
        command = (
            f"docker run --rm {options} {memory_limit} {self.image_name} {os.environ['PYTORCH_MODULES_ROOT_PATH']}/.devops/build.py "
            + " ".join(args)
            + "--cmake-flag -DMANYLINUX=ON"  # TODO: build_with_shim
        )
        log.debug(f"Running command: {command}")
        sp.check_call(command, shell=True)


def gather_wheel_targets(args, wheel_configs: List[WheelConfig]) -> Tuple[Set, List]:
    wheel_targets = set()
    selected_wheel_configs = []

    # TODO: should we take into account some build flags when determining this?
    platform = (
        "linux" if os.environ.get("AUDITWHEEL_POLICY", None) is None else "manylinux"
    )

    if not args.no_ext_build:
        selected_wheel_configs.extend(wheel_configs)
        for config in wheel_configs:
            wheel_targets.add(f"{config.target}/" + platform)
    return wheel_targets, selected_wheel_configs


def log_produced_wheels_and_dump_manifest(selected_wheel_configs: List[WheelConfig]):
    log.info("Produced wheels:")
    wheel_manifest = []
    for no, wheel_config in enumerate(selected_wheel_configs):
        wheel_list = glob.glob(wheel_config.file_path_pattern)
        if wheel_list:
            log.info(
                f" {no: 2}) Built {'optional ' if wheel_config.optional else ''}wheel {wheel_config.full_wheel_name} (pt_vers={wheel_config.pt_vers}, py_ver={wheel_config.py_ver}) in {wheel_list[0]}"
            )
            if len(wheel_list) > 1:
                log.error(
                    "More than one file matched wheel file path pattern - something went wrong"
                )
            else:
                wheel_manifest.append(
                    {
                        "package_name": wheel_config.full_wheel_name,
                        "python_ver": "cp" + str(wheel_config.py_ver).replace(".", ""),
                        "wheel_file": wheel_list[0],
                    }
                )
        else:
            log.info(
                f" {no: 2}) Failed building {'optional ' if wheel_config.optional else ''}wheel {wheel_config.full_wheel_name} (pt_vers={wheel_config.pt_vers}, py_ver={wheel_config.py_ver})"
            )
        with open(
            os.path.join(
                os.environ["PYTORCH_MODULES_RELEASE_BUILD"], "wheel_manifest.json"
            ),
            "w",
        ) as wheel_manifest_fd:
            json.dump(wheel_manifest, wheel_manifest_fd)


def list_wheel_specs_for_specific_pt_versions(versions: Set) -> List[WheelSpec]:
    return [
        WheelSpec(
            wheel_name="habana_torch_plugin",
            pt_versions=versions,
            wheel_src_dir="python_packages",
        ),
        WheelSpec(
            wheel_name="habana_torch_dataloader",
            pt_versions=versions,
            wheel_src_dir="pytorch_helpers/dataloader/habana_dataloader",
        ),
    ]


def prepare_wheel_specs(args, current_pt_version):
    if args.wheel_spec:
        wheel_specs = parse_wheel_spec(args.wheel_spec)
    else:
        if "current" in args.pt_versions:
            if current_pt_version is None:
                log.warning(
                    f"Requested building for 'current' PyTorch version, but no "
                    f"PyTorch is installed. Selecting {recommended_pt_version}."
                )
                current_pt_version = recommended_pt_version
            supported = get_supported_version(current_pt_version, supported_pt_versions)
            if not supported:
                log.fatal(
                    f"Requested current PT version ({current_pt_version}), "
                    f"which is not supported. Currently supported PT versions "
                    f"are {supported_pt_versions}."
                )
                sys.exit(1)
            wheel_specs = list_wheel_specs_for_specific_pt_versions({supported})
        elif "all" in args.pt_versions:
            wheel_specs = list_wheel_specs_for_specific_pt_versions(
                set(supported_pt_versions)
            )
        else:
            versions = set(
                ver if ver == "nightly" else Version(ver) for ver in args.pt_versions
            )
            wheel_specs = list_wheel_specs_for_specific_pt_versions(versions)
    return current_pt_version, wheel_specs


def locate_pt_sources():
    pt_source_dir = os.environ.get(
        "PYTORCH_MODULES_ROOT_PATH",
        os.path.abspath(os.path.join(os.path.dirname(build_py), "..")),
    )
    if os.path.isdir(pt_source_dir):
        log.debug(f"Will use sources at {pt_source_dir}")
    else:
        log.fatal(f"Invalid sources location: {pt_source_dir}")
        sys.exit(1)
    return pt_source_dir


def select_python_versions(args) -> Set[Version]:
    if "current" in args.python_versions:
        supported = get_supported_version(
            system_python_version, supported_python_versions
        )
        if not supported:
            log.fatal(
                f"Requested current python version "
                f"({system_python_version}), which is not supported"
            )
            sys.exit(1)
        return {supported}
    elif "all" in args.python_versions:
        return set(supported_python_versions)
    else:
        return set(Version(ver) for ver in args.python_versions)


def setup_logging(args) -> StringIO:
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)05s [%(filename)s:%(lineno)d] %(" "message)s",
        datefmt="%Y-%m-%d:%H:%M:%S",
    )
    warning_stream = StringIO()
    warnings_handler = logging.StreamHandler(warning_stream)
    warnings_handler.setLevel(logging.WARNING)
    log.addHandler(warnings_handler)
    return warning_stream


def reprint_warnings(warning_stream):
    warnings = warning_stream.getvalue()
    if warnings:
        log.warning("\033[33mWarnings:\033[39m")
        log.warning(warnings)


def print_build_summary(cmake_build_configs):
    log.info("Build summary:")
    for no, (config_dir, venv, optional) in enumerate(cmake_build_configs):
        if venv == os.environ.get("VIRTUAL_ENV", "."):
            log.info(f" {no: 2}) Built {config_dir} using current env.")
        else:
            log.info(
                f" {no: 2}) Built {'optional ' if optional else ''}{config_dir} using virtual env at {venv},\n\t to select it 'source {venv}/bin/activate'."
            )


def main():
    args, raw_args = parse_args()

    warning_stream = setup_logging(args)

    if args.use_icecc:
        ensure_icecc_setup()

    if args.manylinux:  # TODO
        raise NotImplemented("Manylinux builds not yet supported for PT")
        ManylinuxRunner(with_icecc=args.use_icecc).run(raw_args)
        exit()

    if args.get_pt_version:
        print_current_pt_version_and_exit()

    with elapsed_time_logger():
        log.debug(f"Build directory set to {build_dir}")

        pt_modules_root = locate_pt_sources()

        selected_python_versions = select_python_versions(args)
        log.debug(f"Selected Python versions: {selected_python_versions}")

        current_pt_version = get_current_pt_version()

        current_pt_version, wheel_specs = prepare_wheel_specs(args, current_pt_version)

        selected_pt_versions = set([item for sublist in wheel_specs for item in sublist.pt_versions])
        log.debug(
            f"Selected PyTorch versions: {selected_pt_versions}"
        )
        unsupported_pt_versions = selected_pt_versions.difference(supported_pt_versions)
        if unsupported_pt_versions:
            log.fatal(f"Selected unsupported PyTorch versions: {unsupported_pt_versions}")
            sys.exit(1)

        wheels_per_build_envs = prepare_build_envs(
            selected_python_versions,
            wheel_specs,
            pt_modules_root,
            current_python_version=system_python_version,
            current_pt_version=current_pt_version,
            recreate_venv=args.recreate_venv,
        )

        cmake_configurations = get_cmake_configurations(args)
        cmake_build_configs, wheel_configs = prepare_build_dirs(
            build_dir,
            wheels_per_build_envs,
            cmake_configurations,
            pt_modules_root,
            clean=args.configure,
        )

        wheel_targets, selected_wheel_configs = gather_wheel_targets(
            args, wheel_configs
        )

        build(
            build_dir,
            jobs=args.jobs,
            targets=wheel_targets,
            verbose=args.verbose,
            extra_make_flags=("--no-print-directory",),
            use_icecc=args.use_icecc,
        )
        if args.run_ctest:
            run_ctest_on_dirs(cmake_build_configs)

    print_build_summary(cmake_build_configs)

    if not args.no_ext_build:
        log_produced_wheels_and_dump_manifest(selected_wheel_configs)

    reprint_warnings(warning_stream)


if __name__ == "__main__":
    main()


###  TODO: which of the below we should add support for?

# local __install_ext="OFF";
# local __build_with_shim="ON";
# local __auditwheel="${PYTORCH_MODULES_ROOT_PATH}/.ci/scripts/pt_auditwheel.py";
# local __build_manylinux_whl="false";
# -n | --no-ext-build)
#     __build_ext="OFF";
#     __skip_ext_build=""
# ;;
#     local __whl_params="bdist_wheel";
# -i | --install-ext)
#     __install_ext="ON";
#     __whl_params="install"
# ;;
# --pt-version)
#     __pt_vers=$2
# ;;
# --py-version)
#     set_python_version $2;
#     __set_py_vers="true"
# ;;
# --no_shim)
#     __build_with_shim="OFF"
# ;;
# --manylinux)
#     __build_manylinux_whl="true";
#     __build_with_shim="ON"
# ;;
# *)
#     __argument=$1
# ;;
# # submodules...
# __def_vers=$(grep  -A3 $__pt_integ_vers $__ver_path | grep $__default_vers | cut -d':' -f 2);
# popd;
# # ...
# if [ -n "$__debug" ]; then
#     echo -e "Building in debug mode";
#     if [ ! -d $PYTORCH_MODULES_DEBUG_BUILD ]; then
#         __configure="yes";
#     fi;
#     if [ -n "$__configure" ]; then
#         if [ -d $PYTORCH_MODULES_DEBUG_BUILD ]; then
#             rm -rf $PYTORCH_MODULES_DEBUG_BUILD;
#         fi;
#         mkdir -p $PYTORCH_MODULES_DEBUG_BUILD/pkgs;
#         __pt_pkg_dir=$PYTORCH_MODULES_DEBUG_BUILD/pkgs;
#     fi;
#     _verify_exists_dir "$PYTORCH_MODULES_DEBUG_BUILD" $PYTORCH_MODULES_DEBUG_BUILD;
#     ( set -x;
# build happens here
#         mkdir -p $PYTORCH_MODULES_RELEASE_BUILD/pkgs;
#         __pt_pkg_dir=$PYTORCH_MODULES_RELEASE_BUILD/pkgs;
#     fi;
#     _verify_exists_dir "$PYTORCH_MODULES_RELEASE_BUILD" $PYTORCH_MODULES_RELEASE_BUILD;
#     ( set -x;

#    build happens here...

# TODO: auditwheeling, copying the wheel, cleaning up afterwards
# if [ "z${__build_manylinux_whl}" == "ztrue" ]; then
#     __install_auditwheel;
#     rm -rf $__pt_pkg_dir/wheelhouse;
#     for whlfile in $__pt_pkg_dir/*linux_x86_64.whl;
#     do
#         bash -c "$__python_cmd $__auditwheel repair $whlfile -w $__pt_pkg_dir/wheelhouse";
#         if [ $? -eq 0 ]; then
#             rm -f ${whlfile};
#         fi;
#     done;
#     cp -f $__pt_pkg_dir/wheelhouse/*.whl $__pt_pkg_dir;
#     rm -rf $__pt_pkg_dir/wheelhouse;
# fi;
