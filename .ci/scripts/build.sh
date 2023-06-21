#!/usr/bin/env bash
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

function pytorch_functions_help()
{
    echo -e "\n- The following is a list of available functions for PyTorch"
    echo -e "build_pytorch_fork             -   Build the habana pytorch fork"
    echo -e "build_pytorch_lightning_fork   -   Build the habana pytorch lightning fork"
    echo -e "build_pytorch_vision_fork      -   Build the habana pytorch vision fork"
    echo -e "build_pytorch_modules          -   Build habana pytorch intergation modules"
    echo -e "build_pytorch_dist             -   Build habana pytorch distrubuted modules"
    echo -e "build_pytorch_tb_plugin        -   Build habana pytorch tensorboard plugin"
    echo -e "build_habana_lightning_plugins -   Build habana lightning plugin"
    echo -e "build_lightning_habana_fork    -   Build lightning habana fork"
    echo -e "build_pytorch_data             -   Build habana pytorch data"
    echo -e "build_pytorch_text             -   Build habana pytorch text"
    echo -e "build_pytorch_audio            -   Build habana pytorch audio"
    echo -e "run_pytorch_qa_tests           -   Run pytorch QA tests"
    echo -e "run_pytorch_modules_tests      -   Run pytorch modules tests"
    echo -e "run_habana_lightning_tests     -   Run habana lightning plugin tests"
    echo -e "run_lightning_habana_fw_tests  -   Run Lightning Habana tests"
}

function pytorch_usage()
{
    if [ $1 == "build_pytorch_fork" ]; then
        echo -e "\n usage: $1 [options]\n"

        echo -e "options:\n"
        echo -e "  -j,  --jobs <val>           Max jobs used for compilation"
        echo -e "  -c,  --clean                clean up temporary files from 'build' command"
        echo -e "  -a,  --build-all            Build both debug and release build"
        echo -e "  -r,  --release              Build only release build"
        echo -e "       --recursive            Build all the pre-requisite modules in a recursive way"
        echo -e "  -d,  --debug                Build only debug build"
        echo -e "       --install              will install the package"
        echo -e "       --dist                 create a wheel distribution"
        echo -e "       --build-number         Extend whl version number by build number"
        echo -e "       --build-version        Build version used for whl creation"
        echo -e "       --pt-version           Build for given pytorch version"
        echo -e "       --py-version           Python version"
        echo -e "  -h,  --help                 Prints this help"
    fi

    if [ $1 == "build_pytorch_lightning_fork" ]; then
        echo -e "\n usage: $1 [options]\n"

        echo -e "options:\n"
        echo -e "  -j,  --jobs <val>           Max jobs used for compilation"
        echo -e "  -c,  --clean                clean up temporary files from 'build' command"
        echo -e "  -a,  --build-all            Python only code, option ignored"
        echo -e "  -r,  --release              Python only code, option ignored"
        echo -e "  -d,  --debug                Python only code, option ignored"
        echo -e "       --install              will install the package"
        echo -e "       --dist                 create a wheel distribution/default"
        echo -e "       --py-version           Python version"
        echo -e "  -h,  --help                 Prints this help"
    fi

    if [ $1 == "build_habana_lightning_plugins" ]; then
        echo -e "\n usage: $1 [options]\n"

        echo -e "options:\n"
        echo -e "  -j,  --jobs <val>           Max jobs used for compilation"
        echo -e "  -c,  --clean                clean up temporary files from 'build' command"
        echo -e "  -a,  --build-all            Python only code, option ignored"
        echo -e "  -r,  --release              Python only code, option ignored"
        echo -e "  -d,  --debug                Python only code, option ignored"
        echo -e "       --install              will install the package"
        echo -e "       --dist                 create a wheel distribution/default"
        echo -e "       --py-version           Python version"
        echo -e "  -h,  --help                 Prints this help"
    fi

    if [ $1 == "build_lightning_habana_fork" ]; then
        echo -e "\n usage: $1 [options]\n"

        echo -e "options:\n"
        echo -e "  -j,  --jobs <val>           Max jobs used for compilation"
        echo -e "  -c,  --clean                clean up temporary files from 'build' command"
        echo -e "  -a,  --build-all            Python only code, option ignored"
        echo -e "  -r,  --release              Python only code, option ignored"
        echo -e "  -d,  --debug                Python only code, option ignored"
        echo -e "       --install              will install the package"
        echo -e "       --dist                 create a wheel distribution/default"
        echo -e "       --py-version           Python version"
        echo -e "  -h,  --help                 Prints this help"
    fi

    if [ $1 == "build_pytorch_vision_fork" ]; then
        echo -e "\n usage: $1 [options]\n"

        echo -e "options:\n"
        echo -e "  -j,  --jobs <val>           Max jobs used for compilation"
        echo -e "  -c,  --clean                clean up temporary files from 'build' command"
        echo -e "  -a,  --build-all            Python only code, option ignored"
        echo -e "  -r,  --release              Python only code, option ignored"
        echo -e "  -d,  --debug                Python only code, option ignored"
        echo -e "       --install              will install the package"
        echo -e "       --dist                 create a wheel distribution/default"
        echo -e "       --py-version           Python version"
        echo -e "  -h,  --help                 Prints this help"
    fi

    if [ "$1" == "build_pytorch_modules" ]; then
        "${PYTORCH_MODULES_ROOT_PATH}"/.devops/build.py --help
        echo -e ""
        echo -e "Additionally:"
        echo -e "       --recursive            Build all NPU stack dependencies beforehand"
    fi

    if [ $1 == "build_pytorch_dist" ]; then
        echo -e "\n usage: $1 [options]\n"

        echo -e "options:\n"
        echo -e "  -j,  --jobs <val>           Overwrite number of jobs"
        echo -e "  -c   --configure            Configure before build"
        echo -e "  -a,  --build-all            Build both debug and release build"
        echo -e "  -r,  --release              Build only release build"
        echo -e "  -y,  --no-tidy              Skip running clang-tidy during build"
        echo -e "  -s,  --sanitize             Build with sanitize flags on"
        echo -e "  -v,  --verbose              Build with verbose"
        echo -e "  -h,  --help                 Prints this help"
    fi

   if [ $1 == "build_pytorch_tb_plugin" ]; then
      echo -e "\n usage: $1 [options]\n"

        echo -e "options:\n"
        echo -e "  -j,  --jobs <val>           Max jobs used for compilation"
        echo -e "  -c,  --clean                clean up temporary files from 'build' command"
        echo -e "  -a,  --build-all            Python only code, option ignored"
        echo -e "  -r,  --release              Python only code, option ignored"
        echo -e "  -d,  --debug                Python only code, option ignored"
        echo -e "       --install              will install the package"
        echo -e "       --dist                 create a wheel distribution/default"
        echo -e "       --py-version           Python version"
        echo -e "  -h,  --help                 Prints this help"
    fi

    if [ $1 == "run_pytorch_modules_tests" ]; then
        echo -e "\nusage: $1 [options]\n"
        echo -e "options:\n"
        echo -e "  -l,  --list-tests                   List the available tests"
        echo -e "  -s,  --specific-test TEST           Run TEST"
        echo -e "  -d,  --debug                        Use debug test binary"
        echo -e "  -m,  --maxfail NUM                  Stop after NUM failures"
        echo -e "  -p,  --pdb                          Run the app under pdb (python GDB)"
        echo -e "       --dut                          Choose gaudi or gaudi2 or greco. Default is gaudi"
        echo -e "  -x,  --xml PATH                     Output XML file to PATH - available in ST mode only"
        echo -e "  -a,  --marker                       Only run tests matching given mark expression. Example: -a 'mark1 and not mark2'"
        echo -e "  -t,  --suite-type TYPE              Run specific suite type [all, py_tests, cpp_tests]. Default: all"
        echo -e "  -hllog LOG_LEVEL                    0-TRACE, 1-DEBUG 2-INFO, 3-WARN, 4-ERR, 5-CRITICAL"
        echo -e "  -h,  --help                         Prints this help"
    fi

    if [ $1 == "run_pytorch_qa_tests" ]; then
        echo -e "\nusage: $1 [options]\n"
        echo -e "options:\n"
        echo -e "  -l,  --list-tests                   List the available tests"
        echo -e "  -s,  --specific-test TEST           Run TEST"
        echo -e "  -m,  --maxfail NUM                  Stop after NUM failures"
        echo -e "       --no-color                     Disable colors in output"
        echo -e "  -h,  --help                         Prints this help"
        echo -e "  -x,  --xml PATH                     Output XML file to PATH - available in ST mode only"
        echo -e "  -a,  --mark MARKER                  Run tests marked by MARKER"
        echo -e "  -t,  --suite-type TYPE              Run specific suite type [all, ops, perf, acc, topology_ci, distributed]. Default: all"
        echo -e "  -hllog LOG_LEVEL                    0-TRACE, 1-DEBUG 2-INFO, 3-WARN, 4-ERR, 5-CRITICAL"
    fi

    if [ $1 == "run_pytorch_lightning_qa_tests" ]; then
        echo -e "\nusage: $1 [options]\n"
        echo -e "options:\n"
        echo -e "  -l,  --list-tests                   List the available tests"
        echo -e "  -s,  --specific-test TEST           Run TEST"
        echo -e "  -m,  --maxfail NUM                  Stop after NUM failures"
        echo -e "       --no-color                     Disable colors in output"
        echo -e "  -h,  --help                         Prints this help"
        echo -e "  -x,  --xml PATH                     Output XML file to PATH - available in ST mode only"
        echo -e "  -a,  --mark MARKER                  Run tests marked by MARKER"
        echo -e "  -t,  --suite-type TYPE              Run specific suite type [all, ops, perf, acc, topology_ci, distributed]. Default: all"
        echo -e "  -hllog LOG_LEVEL                    0-TRACE, 1-DEBUG 2-INFO, 3-WARN, 4-ERR, 5-CRITICAL"

    fi

    if [ $1 == "run_habana_lightning_tests" ]; then
        echo -e "\nusage: $1 [options]\n"
        echo -e "options:\n"
        echo -e "  -l,  --list-tests                   List the available tests"
        echo -e "  -s,  --specific-test TEST           Run TEST"
        echo -e "  -m,  --maxfail NUM                  Stop after NUM failures"
        echo -e "  -p,  --pdb                          Run the app under pdb (python GDB)"
        echo -e "       --dut                          Choose gaudi or gaudi2 or greco. Default is gaudi"
        echo -e "  -x,  --xml PATH                     Output XML file to PATH - available in ST mode only"
        echo -e "  -a,  --marker                       Only run tests matching given mark expression. Example: -a 'mark1 and not mark2'"
        echo -e "  -t,  --suite-type TYPE              Run specific suite type [all, py_tests, cpp_tests]. Default: all"
        echo -e "  -hllog LOG_LEVEL                    0-TRACE, 1-DEBUG 2-INFO, 3-WARN, 4-ERR, 5-CRITICAL"
        echo -e "  -h,  --help                         Prints this help"
    fi

    if [ $1 == "run_lightning_habana_fw_tests" ]; then
        echo -e "\nusage: $1 [options]\n"
        echo -e "options:\n"
        echo -e "  -l,  --list-tests                   List the available tests"
        echo -e "  -s,  --specific-test TEST           Run TEST"
        echo -e "  -m,  --maxfail NUM                  Stop after NUM failures"
        echo -e "  -p,  --pdb                          Run the app under pdb (python GDB)"
        echo -e "       --dut                          Choose gaudi or gaudi2 or greco. Default is gaudi"
        echo -e "  -x,  --xml PATH                     Output XML file to PATH - available in ST mode only"
        echo -e "  -a,  --marker                       Only run tests matching given mark expression. Example: -a 'mark1 and not mark2'"
        echo -e "  -t,  --suite-type TYPE              Run specific suite type [all, py_tests, cpp_tests]. Default: all"
        echo -e "  -hllog LOG_LEVEL                    0-TRACE, 1-DEBUG 2-INFO, 3-WARN, 4-ERR, 5-CRITICAL"
        echo -e "  -h,  --help                         Prints this help"
    fi

    if [ $1 == "build_pytorch_data" ]; then
        echo -e "\n usage: $1 [options]\n"

        echo -e "options:\n"
        echo -e "  -j,  --jobs <val>           Max jobs used for compilation"
        echo -e "  -c,  --clean                clean up temporary files from 'build' command"
        echo -e "  -r,  --release              Python only code, option ignored"
        echo -e "  -d,  --debug                Python only code, option ignored"
        echo -e "       --install              will install the package"
        echo -e "       --dist                 create a wheel distribution/default"
        echo -e "       --py-version           Python version"
        echo -e "       --pt-data-version      PytorchData version"
        echo -e "  -h,  --help                 Prints this help"
    fi

    if [ $1 == "build_pytorch_text" ]; then
        echo -e "\n usage: $1 [options]\n"

        echo -e "options:\n"
        echo -e "  -j,  --jobs <val>           Max jobs used for compilation"
        echo -e "  -c,  --clean                clean up temporary files from 'build' command"
        echo -e "  -r,  --release              Python only code, option ignored"
        echo -e "  -d,  --debug                Python only code, option ignored"
        echo -e "       --install              will install the package"
        echo -e "       --dist                 create a wheel distribution/default"
        echo -e "       --py-version           Python version"
        echo -e "       --pt-text-version      PytorchText version"
        echo -e "  -h,  --help                 Prints this help"
    fi

    if [ $1 == "build_pytorch_audio" ]; then
        echo -e "\n usage: $1 [options]\n"

        echo -e "options:\n"
        echo -e "  -j,  --jobs <val>           Max jobs used for compilation"
        echo -e "  -c,  --clean                clean up temporary files from 'build' command"
        echo -e "  -r,  --release              Python only code, option ignored"
        echo -e "  -d,  --debug                Python only code, option ignored"
        echo -e "       --install              will install the package"
        echo -e "       --dist                 create a wheel distribution/default"
        echo -e "       --py-version           Python version"
        echo -e "       --pt-audio-version     PytorchAudio version"
        echo -e "  -h,  --help                 Prints this help"
    fi
}

build_pytorch_modules()
{
    SECONDS=0

    local __scriptname=$(__get_func_name)

    local __jobs=${NUMBER_OF_JOBS}
    local __all=""
    local __configure=""
    local __release=""
    local __build_res=0
    local __pytorch_module_name="pytorch_bridge"
    local __recursive=""
    local __result=""

    local __variables_to_build
    __variables_to_build=$(printf "%s\n" "$@" | sed s/--recursive// | sed s/--no-tidy//)

    # parameter while-loop
    while [ -n "$1" ];
    do
        case $1 in
        -a  | --build-all )
            __all="yes"
            ;;
        -j  | --jobs )
            shift
            __jobs=$1
            ;;
        -c  | --configure )
            __configure="yes"
            ;;
        -h  | --help )
            usage $__scriptname
            return 0
            ;;
        -r  | --release )
            __release="yes"
            ;;
        --recursive )
            __recursive="yes"
            ;;
        esac
        shift
    done

    if [ -n "$__configure" ]; then
        __check_mandatory_pkgs
        if [ $? -ne 0 ]; then
            restore_python_version
            return 1
        fi
    fi

    #CI job creates venv for every job. So we need to have python pkg install unconditionally
    install_pkg=($__pip_cmd install -r $PYTORCH_MODULES_ROOT_PATH/requirements.txt)
    if ! __running_in_venv; then
        install_pkg+=(--user)
    fi
    "${install_pkg[@]}"

    if [ -n "$KINETO_ROOT" ]; then
        echo "git submodule update for kineto"
        pushd $KINETO_ROOT
        __result=$?
        if [ $__result -ne 0 ]; then
            echo "Unable to cd into Kineto's root ($KINETO_ROOT)"
            return $__result
        fi
        git submodule update --init
        popd
    fi

    pushd $PYTORCH_MODULES_ROOT_PATH

    echo "git submodule update for pybind11"
    git submodule sync
    __result=$?
    if [ $__result -ne 0 ]; then
        echo "git submodule init failed!"
        popd
        restore_python_version
        return $__result
    fi

    git submodule update --init --recursive --force
    __result=$?
    if [ $__result -ne 0 ]; then
        echo "git submodule update failed!"
        popd
        restore_python_version
        return $__result
    fi

    if [ -n "$__recursive" ]; then
        local __release_par=""
        local __configure_par=""
        local __jobs_par=""

        if [ -n "$__configure" ]; then
            __configure_par="-c"
        fi

        if [ -n "$__release" ]; then
            __release_par="-r"
        fi

        if [ -n "$__all" ]; then
            __release_par="-a"
        fi

        __jobs_par="-j $__jobs"

        echo "Building pre-requisite packages for $__pytorch_module_name"

        __common_build_dependency -m $__pytorch_module_name $__configure_par $__release_par $__jobs_par
        __result=$?
        if [ $__result -ne 0 ]; then
            echo "Failed to build dependency packages $__pytorch_module_name"
            popd
            restore_python_version
            return $__result
        fi
    fi

    "${PYTORCH_MODULES_ROOT_PATH}"/.devops/build.py $__variables_to_build
    __result=$?
    if [ $__result -ne 0 ]; then
        echo "Failed to run build.py. Exit code: " $__result
        popd
        restore_python_version
        return $__result
    fi

    popd
    restore_python_version

    printf "\nElapsed time: %02u:%02u:%02u \n\n" $(($SECONDS / 3600)) $((($SECONDS / 60) % 60)) $(($SECONDS % 60))
}

build_pytorch_dist()
{
    SECONDS=0

    echo "Pytorch dist build skipped.  Support will be removed next release"
    return 0

    local __scriptname=$(__get_func_name)

    local __jobs=${NUMBER_OF_JOBS}
    local __all=""
    local __debug="yes"
    local __configure=""
    local __release=""
    local __no_tidy=""
    local __sanitize="OFF"
    local __verbose=""
    local __build_res=0

    # parameter while-loop
    while [ -n "$1" ];
    do
        case $1 in
        -a  | --build-all )
            __all="yes"
            ;;
        -j  | --jobs )
            shift
            __jobs=$1
            ;;
        -c  | --configure )
            __configure="yes"
            ;;
        -h  | --help )
            usage $__scriptname
            return 0
            ;;
        -r  | --release )
            __debug=""
            __release="yes"
            ;;
        -y  | --no-tidy )
            __no_tidy="yes"
            ;;
        -s  | --sanitize )
            __sanitize="ON"
            ;;
        -v  | --verbose )
            __verbose="VERBOSE=1"
            ;;
        *)
            __argument=$1
            ;;
        esac
        shift
    done

    pushd $PYTORCH_MODULES_ROOT_PATH
    echo "git submodule update for pybind11"
    git submodule sync
    __result=$?
    if [ $__result -ne 0 ]; then
        echo "git submodule init failed!"
        popd
        return $__result
    fi

    git submodule update --init --recursive --force
    __result=$?
    if [ $__result -ne 0 ]; then
        echo "git submodule update failed!"
        popd
        return $__result
    fi
    popd

    if [ -n "$__all" ]; then
        __debug="yes"
        __release="yes"
    fi

    CLANG_TIDY_DEFINE=""
    if [ ! -z "$__no_tidy" ]; then
        CLANG_TIDY_DEFINE="-DCLANG_TIDY="
    fi

    if [ -n "$__debug" ]; then
        echo -e "Building in debug mode"
        if [ ! -d $PYTORCH_DIST_DEBUG_BUILD ]; then
            __configure="yes"
        fi

        if [ -n "$__configure" ]; then
            if [ -d $PYTORCH_DIST_DEBUG_BUILD ]; then
                rm -rf $PYTORCH_DIST_DEBUG_BUILD
            fi
            mkdir -p $PYTORCH_DIST_DEBUG_BUILD
        fi

        _verify_exists_dir "$PYTORCH_DIST_DEBUG_BUILD" $PYTORCH_DIST_DEBUG_BUILD

        pushd $PYTORCH_DIST_DEBUG_BUILD
        (set -x; cmake \
            -DCMAKE_BUILD_TYPE=Debug \
            $CLANG_TIDY_DEFINE \
            -DSANITIZER=$__sanitize \
            $PYTORCH_DIST_ROOT_PATH)
         make $__verbose -j$__jobs
        __build_res=$?
        popd
        if [ $__build_res -ne 0 ]; then
            return $__build_res
        fi

        cp -fs $PYTORCH_DIST_DEBUG_BUILD/*.so $BUILD_ROOT_DEBUG
        if [ -z "$__all" ]; then
            cp -fs $PYTORCH_DIST_DEBUG_BUILD/*.so $BUILD_ROOT_LATEST
        fi
    fi

    if [ -n "$__release" ]; then
        echo "Building in release mode"
        if [ ! -d $PYTORCH_DIST_RELEASE_BUILD ]; then
            __configure="yes"
        fi

        if [ -n "$__configure" ]; then
            if [ -d $PYTORCH_DIST_RELEASE_BUILD ]; then
                rm -rf $PYTORCH_DIST_RELEASE_BUILD
            fi
            mkdir -p $PYTORCH_DIST_RELEASE_BUILD
        fi

        _verify_exists_dir "$PYTORCH_DIST_RELEASE_BUILD" $PYTORCH_DIST_RELEASE_BUILD
        pushd $PYTORCH_DIST_RELEASE_BUILD
        (set -x; cmake \
            -DCMAKE_BUILD_TYPE=Release \
            $CLANG_TIDY_DEFINE \
            -DSANITIZER=$__sanitize \
            $PYTORCH_DIST_ROOT_PATH)
        make $__verbose -j$__jobs
        __build_res=$?
        popd
        if [ $__build_res -ne 0 ]; then
            return $__build_res
        fi

        cp -fs $PYTORCH_DIST_RELEASE_BUILD/*.so $BUILD_ROOT_RELEASE
        cp -fs $PYTORCH_DIST_RELEASE_BUILD/*.so $BUILD_ROOT_LATEST
    fi

    printf "\nElapsed time: %02u:%02u:%02u \n\n" $(($SECONDS / 3600)) $((($SECONDS / 60) % 60)) $(($SECONDS % 60))
    return 0
}

__conda()
{
    if [ "z$__no_conda" == "ztrue" ]; then
        return
    else
        conda $*
    fi
}

__install_auditwheel()
{
    $__pip_cmd install auditwheel
}

build_pytorch_fork()
{
    SECONDS=0

    local __scriptname=$(__get_func_name)
    local __env_vars="DEBUG=1 USE_MPI=OFF"
    local __configure=""
    local __release=""
    local __debug="yes"
    local __whl_params=" bdist_wheel"
    local __pt_vers=""
    local __result
    local __pt_fork_tag="pytorch_fork_tags"
    local __pt_fork_vers="pytorch_fork_version"
    local __default_vers="default_vers"
    local __def_vers=""
    local __branch=""
    local __ver_path="${PYTORCH_MODULES_ROOT_PATH}/.ci/scripts/pt_version.json"
    local __build_manylinux_whl="false"
    local __auditwheel="${PYTORCH_MODULES_ROOT_PATH}/.ci/scripts/pt_auditwheel.py"
    local __set_py_vers="false"
    local __no_conda="false"

    # parameter while-loop
    while [ -n "$1" ];
    do
        case $1 in
        -j  | --jobs )
            __env_vars+=" MAX_JOBS=$2"
            ;;
        -c  | --configure )
             __configure="yes"
            ;;
        -r  | --release )
            __release="yes"
            __debug=""
            # Remove debug from env variable
            __env_vars=${__env_vars//"DEBUG=1"/}
            ;;
        -d  | --debug )
            __debug="yes"
            __release=""
            ;;
        --dist )
            __whl_params=" bdist_wheel"
            ;;
        --install )
            __whl_params=" install"
            ;;
        --build-number )
            __env_vars+=" PYTORCH_BUILD_NUMBER=$2"
            ;;
        --build-version )
            __env_vars+=" PYTORCH_BUILD_VERSION=$2"
            ;;
        --pt-version )
            __pt_vers=$2
            ;;
        --py-version )
             set_python_version $2
             __set_py_vers="true"
            ;;
        --no-conda )
            __no_conda="true"
            ;;
        --manylinux )
            __no_conda="true"
            __build_manylinux_whl="true"
            ;;
        -h  | --help )
            usage $__scriptname
            restore_python_version
            return 0
            ;;
        esac
        shift
    done

    if [ "z$__no_conda" != "ztrue" ]; then
        __install_anaconda
        __result=$?
        if [ $__result -ne 0 ]; then
            restore_python_version
            return $__result
        fi

        if [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
            source "$HOME/anaconda3/etc/profile.d/conda.sh"
        else
            export PATH="$HOME/anaconda3/bin:$PATH"
        fi

        local __venv=py_venv_$__python_ver
        __conda activate $__venv
        __result=$?
        if [ $__result -ne 0 ]; then
            echo "conda env $__venv activation failed"
            echo "Forgot to build with option -c ?"
            restore_python_version
            return $__result
        fi
        echo "Activated $__venv enviornment"
    else
        $__python_cmd -m pip install -r ${PYTORCH_MODULES_ROOT_PATH}/.ci/requirements/requirements-pytorch-python${__python_ver}_base.txt
    fi

    unset CMAKE_ROOT  # we're using CMake from requirements files

    pushd $PYTORCH_FORK_ROOT
    git submodule sync
    __result=$?
    if [ $__result -ne 0 ]; then
        echo "git submodule init failed!"
        __conda deactivate
        popd
        restore_python_version
        return $__result
    fi

    git submodule update --init --recursive --force
    __result=$?
    if [ $__result -ne 0 ]; then
        echo "git submodule update failed!"
        __conda deactivate
        popd
        restore_python_version
        return $__result
    fi

    __def_vers=$(grep  -A3 $__pt_fork_vers pt_version.json | grep $__default_vers | cut -d':' -f 2)
    if [ -n "$__pt_vers" ] && [ "$__def_vers" != "$__pt_vers" ]; then
        __branch=$(grep -A3 $__pt_fork_tag  __ver_path | grep $__pt_vers | awk -F $__pt_vers '{print $2}' | cut -d':' -f 2)
        if [ $__result -ne 0 ]; then
            echo "version $__pt_vers not found!"
            __conda deactivate
            popd
            restore_python_version
            return $__result
        fi
        echo "git checkout $__branch"
        git checkout $__branch
        __result=$?
        if [ $__result -ne 0 ]; then
            echo "git checkout $__branch failed!"
            __conda deactivate
            popd
            restore_python_version
            return $__result
        fi
    fi

    if [ "$__pt_vers" == "" ]; then
        echo "Default branch will be compiled"
    fi

    if [ -n "$__configure" ]; then
        $__python_cmd setup.py clean
        git clean -fd
        git submodule foreach --recursive git clean -xfd
    fi

    if [ -n "$__no_conda" ] && [ "$__no_conda" != "true" ]; then
        (set -x;export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"})
    else
        echo "CMAKE_BUILD=$CMAKE_BUILD"
        echo "CMAKE_ROOT=$CMAKE_ROOT"
    fi

    local __pkg_name="TORCH_PACKAGE_NAME=torch"
    if [ -n "$__debug" ]; then
       __pkg_name+="-debug"
       echo "Building torch in Debug mode"
    else
       echo "Building torch in Release mode"
    fi

    echo "Build parameters ${__whl_params}"
    if [ -n "$__env_vars" ]; then
        echo "Build Enviornment parameters ${__env_vars}"
    fi

    (set -x;eval ${__env_vars} ${__pkg_name} $__python_cmd setup.py ${__whl_params})
    __result=$?
    if [ $__result -ne 0 ]; then
        echo "Pytorch fork build failed!"
        __conda deactivate
        popd
        restore_python_version
        return $__result
    fi
    __conda deactivate

    if [ "z${__build_manylinux_whl}" == "ztrue" ];then
        __install_auditwheel
        bash -c "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PYTORCH_FORK_ROOT/torch/lib;$__python_cmd $__auditwheel repair $PYTORCH_FORK_ROOT/dist/torch*.whl"
        TORCH_WHL_PATH="$PYTORCH_FORK_ROOT/wheelhouse/"
    else
        TORCH_WHL_PATH="$PYTORCH_FORK_ROOT/dist/"
    fi
    if [ -n "$__debug" ]; then
       rm -rf $PYTORCH_FORK_DEBUG_BUILD/pkgs
       mkdir -p $PYTORCH_FORK_DEBUG_BUILD/pkgs
       cp -f ${TORCH_WHL_PATH}/torch*.whl $PYTORCH_FORK_DEBUG_BUILD/pkgs
    else
       rm -rf $PYTORCH_FORK_RELEASE_BUILD/pkgs
       mkdir -p $PYTORCH_FORK_RELEASE_BUILD/pkgs
       cp -f ${TORCH_WHL_PATH}/torch*.whl $PYTORCH_FORK_RELEASE_BUILD/pkgs
    fi
    popd
    printf "\nElapsed time: %02u:%02u:%02u \n\n" $(($SECONDS / 3600)) $((($SECONDS / 60) % 60)) $(($SECONDS % 60))
    restore_python_version
    return $__result
}


build_pytorch_tb_plugin()
{
    SECONDS=0
    local __scriptname=$(__get_func_name)
    local __env_vars=""
    local __configure=""
    local __whl_params=" sdist bdist_wheel"
    local __build_package="true"
    local __result
    local __set_py_vers="false"
    # parameter while-loop
    while [ -n "$1" ];
    do
        case $1 in
        -j  | --jobs )
            __env_vars+=" MAX_JOBS=$2"
            ;;
        -c  | --configure )
             __configure="yes"
            ;;
        -r  | --release )
            ;;
        -d  | --debug )
            ;;
        --dist )
            __whl_params=" sdist bdist_wheel"
            __build_package="true"
            ;;
        --install )
            __whl_params=" install"
            __build_package=""
            ;;
        --py-version )
            set_python_version $2
            __set_py_vers="true"
            ;;
        -h  | --help )
            usage $__scriptname
            restore_python_version
            return 0
            ;;
        esac
        shift
    done

    if [ -n "$__configure" ]; then
        __check_mandatory_pkgs
        if [ $? -ne 0 ]; then
            restore_python_version
            return 1
        fi
    fi

    #Initialize yarn dependencies for front-end
    #This is prerequisite before invoking setup.py
    pushd $KINETO_ROOT/tb_plugin/fe
    yarn install
    __result=$?
    if [ $__result -ne 0 ]; then
        echo "yarn install failed"
        popd
        restore_python_version
        return $__result
    fi
    popd

    install_cmd=($__pip_cmd install wheel)
    if ! __running_in_venv; then
        install_cmd+=(--user)
    fi
    "${install_cmd[@]}"
    __result=$?
    if [ $__result -ne 0 ]; then
        echo "pip install failed"
        restore_python_version
        return $__result
    fi

    pushd $KINETO_ROOT/tb_plugin

    if [ -n "$__configure" ]; then
        $__python_cmd setup.py clean
        git clean -fd
    fi

    echo "Build parameters ${__whl_params}"

    (set -x;eval ${__env_vars} $__python_cmd setup.py ${__whl_params})
    __result=$?
    if [ $__result -ne 0 ]; then
        echo "Pytorch tb plugin build failed!"
    fi

    popd
    if [ -n "$__build_package" ]; then
        PT_TB_WHL_PATH="$KINETO_ROOT/tb_plugin/dist/"
        rm -rf $PYTORCH_TB_PLUGIN_BUILD/pkgs
        mkdir -p $PYTORCH_TB_PLUGIN_BUILD/pkgs
        cp -f ${PT_TB_WHL_PATH}/*.whl $PYTORCH_TB_PLUGIN_BUILD/pkgs
    fi

    printf "\nElapsed time: %02u:%02u:%02u \n\n" $(($SECONDS / 3600)) $((($SECONDS / 60) % 60)) $(($SECONDS % 60))
    restore_python_version
    return $__result
}


build_pytorch_lightning_fork()
{
    SECONDS=0
    local __scriptname=$(__get_func_name)
    local __env_vars="PACKAGE_NAME=\"pytorch\""
    local __configure=""
    local __whl_params=" bdist_wheel"
    local __result
    local __set_py_vers="false"
    # parameter while-loop
    while [ -n "$1" ];
    do
        case $1 in
        -j  | --jobs )
            __env_vars+=" MAX_JOBS=$2"
            ;;
        -c  | --configure )
             __configure="yes"
            ;;
        -r  | --release )
            ;;
        -d  | --debug )
            ;;
        --dist )
            __whl_params=" bdist_wheel"
            ;;
        --install )
            __whl_params=" install"
            ;;
        --py-version )
            set_python_version $2
            __set_py_vers="true"
            ;;
        -h  | --help )
            usage $__scriptname
            restore_python_version
            return 0
            ;;
        esac
        shift
    done

    pushd $PYTORCH_LIGHTNING_FORK_ROOT

    if [ -n "$__configure" ]; then
        eval ${__env_vars} $__python_cmd setup.py clean
        git clean -fd
    fi

    echo "Build parameters ${__whl_params}"

    (set -x;eval ${__env_vars} $__python_cmd setup.py ${__whl_params})
    __result=$?
    if [ $__result -ne 0 ]; then
        echo "Pytorch lightning build failed!"
    fi

    popd
    if [[ "$__whl_params" = " bdist_wheel" ]]; then
        PTL_WHL_PATH="$PYTORCH_LIGHTNING_FORK_ROOT/dist/"
        rm -rf $PYTORCH_LIGHTNING_FORK_BUILD/pkgs
        mkdir -p $PYTORCH_LIGHTNING_FORK_BUILD/pkgs
        cp -f ${PTL_WHL_PATH}/*.whl $PYTORCH_LIGHTNING_FORK_BUILD/pkgs
    fi

    printf "\nElapsed time: %02u:%02u:%02u \n\n" $(($SECONDS / 3600)) $((($SECONDS / 60) % 60)) $(($SECONDS % 60))
    restore_python_version
    return $__result

}

build_habana_lightning_plugins()
{
    SECONDS=0
    local __scriptname=$(__get_func_name)
    local __env_vars=""
    local __configure=""
    local __whl_params=" bdist_wheel"
    local __result
    local __set_py_vers="false"
    # parameter while-loop
    while [ -n "$1" ];
    do
        case $1 in
        -j  | --jobs )
            __env_vars+=" MAX_JOBS=$2"
            ;;
        -c  | --configure )
             __configure="yes"
            ;;
        -r  | --release )
            ;;
        -d  | --debug )
            ;;
        --dist )
            __whl_params=" bdist_wheel"
            ;;
        --install )
            __whl_params=" install"
            ;;
        --py-version )
            set_python_version $2
            __set_py_vers="true"
            ;;
        -h  | --help )
            usage $__scriptname
            restore_python_version
            return 0
            ;;
        esac
        shift
    done

    pushd $HABANA_LIGHTNING_PLUGINS_ROOT

    if [ -n "$__configure" ]; then
        eval ${__env_vars} $__python_cmd setup.py clean
        git clean -fd
    fi

    echo "Build parameters for habana lightning plugins ${__whl_params}"

    (set -x;eval ${__env_vars} $__python_cmd setup.py ${__whl_params})
    __result=$?
    if [ $__result -ne 0 ]; then
        echo "Pytorch lightning build failed!"
    fi

    popd
    if [[ "$__whl_params" = " bdist_wheel" ]]; then
        PTL_WHL_PATH="$HABANA_LIGHTNING_PLUGINS_ROOT/dist/"
        rm -rf $HABANA_LIGHTNING_PLUGINS_BUILD/pkgs
        mkdir -p $HABANA_LIGHTNING_PLUGINS_BUILD/pkgs
        cp -f ${PTL_WHL_PATH}/*.whl $HABANA_LIGHTNING_PLUGINS_BUILD/pkgs
    fi

    printf "\nElapsed time: %02u:%02u:%02u \n\n" $(($SECONDS / 3600)) $((($SECONDS / 60) % 60)) $(($SECONDS % 60))
    restore_python_version
    return $__result
}

build_lightning_habana_fork()
{
    SECONDS=0
    local __scriptname=$(__get_func_name)
    local __env_vars=""
    local __configure=""
    local __whl_params=" bdist_wheel"
    local __result
    local __set_py_vers="false"
    # parameter while-loop
    while [ -n "$1" ];
    do
        case $1 in
        -j  | --jobs )
            __env_vars+=" MAX_JOBS=$2"
            ;;
        -c  | --configure )
             __configure="yes"
            ;;
        -r  | --release )
            ;;
        -d  | --debug )
            ;;
        --dist )
            __whl_params=" bdist_wheel"
            ;;
        --install )
            __whl_params=" install"
            ;;
        --py-version )
            set_python_version $2
            __set_py_vers="true"
            ;;
        -h  | --help )
            usage $__scriptname
            restore_python_version
            return 0
            ;;
        esac
        shift
    done

    pushd $LIGHTNING_HABANA_FORK_ROOT

    if [ -n "$__configure" ]; then
        eval ${__env_vars} $__python_cmd setup.py clean
        git clean -fd
    fi

    echo "Build parameters for lightning habana ${__whl_params}"

    (set -x;eval ${__env_vars} $__python_cmd setup.py ${__whl_params})
    __result=$?
    if [ $__result -ne 0 ]; then
        echo "Lightning habana build failed!"
    fi

    popd
    if [[ "$__whl_params" = " bdist_wheel" ]]; then
        PTL_WHL_PATH="$LIGHTNING_HABANA_FORK_ROOT/dist/"
        rm -rf $LIGHTNING_HABANA_FORK_BUILD/pkgs
        mkdir -p $LIGHTNING_HABANA_FORK_BUILD/pkgs
        cp -f ${PTL_WHL_PATH}/*.whl $LIGHTNING_HABANA_FORK_BUILD/pkgs
    fi

    printf "\nElapsed time: %02u:%02u:%02u \n\n" $(($SECONDS / 3600)) $((($SECONDS / 60) % 60)) $(($SECONDS % 60))
    restore_python_version
    return $__result
}

build_pytorch_vision_fork()
{
    SECONDS=0
    local __scriptname=$(__get_func_name)
    local __env_vars=""
    local __configure=""
    local __whl_params=" bdist_wheel"
    local __result
    local __build_manylinux_whl="false"
    local __auditwheel="${PYTORCH_MODULES_ROOT_PATH}/.ci/scripts/pt_auditwheel.py"
    local __set_py_vers="false"
    # parameter while-loop
    while [ -n "$1" ];
    do
        case $1 in
        -j  | --jobs )
            __env_vars+=" MAX_JOBS=$2"
            ;;
        -c  | --configure )
             __configure="yes"
            ;;
        -r  | --release )
            ;;
        -d  | --debug )
            ;;
        --dist )
            __whl_params=" bdist_wheel"
            ;;
        --install )
            __whl_params=" install"
            ;;
        --manylinux )
            __build_manylinux_whl="true"
            ;;
        --py-version )
            set_python_version $2
            __set_py_vers="true"
            ;;
        -h  | --help )
            usage $__scriptname
            restore_python_version
            return 0
            ;;
        esac
        shift
    done

    pushd $PYTORCH_VISION_FORK_ROOT

    if [ -n "$__configure" ]; then
        $__python_cmd setup.py clean
        git clean -fd
    fi

    echo "Build parameters ${__whl_params}"

    (set -x;eval ${__env_vars} $__python_cmd setup.py ${__whl_params})
    __result=$?
    if [ $__result -ne 0 ]; then
        echo "Pytorch torchvision build failed!"
    fi
    if [ "z${__build_manylinux_whl}" == "ztrue" ];then
        __install_auditwheel
        bash -c "$__python_cmd $__auditwheel repair $PYTORCH_VISION_FORK_ROOT/dist/*.whl"
        PTV_WHL_PATH="$PYTORCH_VISION_FORK_ROOT/wheelhouse/"
    else
        PTV_WHL_PATH="$PYTORCH_VISION_FORK_ROOT/dist/"
    fi

    popd
    if [[ "$__whl_params" = " bdist_wheel" ]]; then
        rm -rf $PYTORCH_VISION_FORK_BUILD/pkgs
        mkdir -p $PYTORCH_VISION_FORK_BUILD/pkgs
        cp -f ${PTV_WHL_PATH}/*.whl $PYTORCH_VISION_FORK_BUILD/pkgs
    fi

    printf "\nElapsed time: %02u:%02u:%02u \n\n" $(($SECONDS / 3600)) $((($SECONDS / 60) % 60)) $(($SECONDS % 60))
    restore_python_version
    return $__result

}

run_pytorch_modules_tests()
{
    local __pytorch_modules_tests_exe="python -m pytest"
    local __cpp_tests_exe="$PYTORCH_MODULES_RELEASE_BUILD/test_pt_integration"
    local __scriptname=$(__get_func_name)
    local __xml=""
    local __ld_lib="$BUILD_ROOT_RELEASE"
    local __print_tests=""
    local __py_filter=""
    local __cpp_filter=""
    local __failures=""
    local __marker=""
    local __verbose=""
    local __test_status=0
    local __suite_type="all"
    local __dut="gaudi"
    local __hllog=3

    # parameter while-loop
    while [ -n "$1" ];
    do
        case $1 in
        -d  | --debug )
            __cpp_tests_exe="$PYTORCH_MODULES_DEBUG_BUILD/test_pt_integration"
            ;;
        -l  | --list-tests )
            __print_tests="yes"
            ;;
        -s  | --specific-test )
            shift
            __py_filter="-k $1"
            __cpp_filter="--gtest_filter=$1"
            ;;
        -m  | --maxfail )
            shift
            __failures="--maxfail=$1"
            ;;
        -p  | --pdb )
            shift
            __pdb="--pdb"
            ;;
        -t | --suite-type )
            shift
            __suite_type="$1"
            ;;
	     --dut )
            shift
            __dut="$1"
            ;;
        -hllog )
            shift
            __hllog=$1
            ;;
        -x  | --xml )
            shift
            __xml="$1"
            ;;
        -a | --marker )
            shift
            __marker="-m \"$1\""
            ;;
        -h  | --help )
            usage $__scriptname
            return 0
            ;;
        *)
            echo "The parameter $1 is not allowed"
            usage $__scriptname
            return 1 # error
            ;;
        esac
        shift
    done

    case $__suite_type in
    all)
        echo "List or Run 'all' tests"
        ;;
    py_tests)
        __test_type="py_tests"
        ;;
    cpp_tests)
        __test_type="cpp_tests"
        ;;
    *)
        echo "Test suite type \"$__suite_type\" is not allowed"
        usage $__scriptname
        return 1 # error
        ;;
    esac

    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${__ld_lib}
    if [ "$__suite_type" == "all" ] || [ "$__suite_type" == "cpp_tests" ]; then
    	if [ "$__dut" == "gaudi" ]; then
        (set -x; eval LOG_LEVEL_ALL=${__hllog} $__cpp_tests_exe --gtest_output=xml:$__xml $__cpp_filter)
        	__test_status=$?
        elif [ "$__dut" == "gaudi2" ]; then
        echo "Running tests on Gaudi2"
	(set -x; eval LOG_LEVEL_ALL=${__hllog} $__cpp_tests_exe --gtest_output=xml:$__xml --gtest_filter=-HpuOpTest.nll_loss2d_fwd_out_bf16:LazyInferencePassTest.linear $__cpp_filter)
    		__test_status=$?
	elif [ "$__dut" == "gaudi3" ]; then
	echo "Running tests on Gaudi3"
                (set -x; eval LOG_LEVEL_ALL=${__hllog} $__cpp_tests_exe --gtest_output=xml:$__xml --gtest_filter=-EagerKernelTest.MatMulTest:EagerKernelTest.*:LazyCustomKernelTest.OptSgdMomentumCustomOp_*:LazyDynamicComputeOutputShapesTest.*:LazyDynamicShapesTest*:LazyIndexKernelTest*:GenOps*:LazyFillKernelTest.ExecuteFillGraph*:LazyLinearKernelTest.MatmulTest:LazyFillKernelTest.ExecuteZerosGraph:LazyTensorShapeKernelTest*:LazyLinearKernelTest*:SBS/SBSWithParamsTest*:LazyBinaryKernelTest*:UniqueDimTest*:HpuOpComputeShapeTest*:LazyInferencePassTest*:LazyBasicKernelTest*:LazyDynamicFallbackTest*:LazyDynamicShapesBucketRefineTest*:LazyDynamicShapesSerializtionTest*:LazyMaskKernelTest*:LazyNormKernelTest*:LazyReductionKernelTest*:LazyUnaryKernelTest*:LazyUpsampleKernelTest*:SifTest*:TypePromotionTests*:TestStream*:norm/NormHpuOpTest*:TypePromotion*:HpuOpTest*)
		__test_status=$?
        elif [ "$__dut" == "greco" ]; then
        echo "Running greco tests"
		(set -x; eval LOG_LEVEL_ALL=${__hllog} PT_HPU_INFERENCE_MODE=true $__cpp_tests_exe --gtest_output=xml:$__xml --gtest_filter=HpuOpTest*addmm*:HpuOpTest*addbmm*:*LayerNormForwardExecute*:*LazyConvKernel*Pool* $__cpp_filter)
    		__test_status=$?
	fi
    fi

    if [ -n "$__print_tests" ]; then
        pushd $HABANA_SOFTWARE_STACK/pytorch-integration/tests/
        echo "python tests:"
        ${__pytorch_modules_tests_exe} --collect-only

        echo "cpp tests:"
        ${__cpp_tests_exe} --gtest_list_tests
        __test_status=$?
        popd
        return $__test_status
    fi

    if [[ "$__suite_type" = "all" || "$__suite_type" = "py_tests" ]] ; then
        pushd $HABANA_SOFTWARE_STACK/pytorch-integration/tests/
        (set -x; eval ${__pytorch_modules_tests_exe} pytest_working/ -v $__failures $__py_filter --junit-xml="${__xml}tests.xml" ${__marker})
        __test_status=$__test_status || $?
        popd
    fi

    # return error code of the tests
    return ${__test_status}
}

run_pytorch_qa_tests()
{
    local __pytorch_qa_test_path="$HABANA_PYTORCH_QA_ROOT"
    local __scriptname=$(__get_func_name)
    local __print_tests=""
    local __filter=" "
    local __xml=""
    local __failures=""
    local __color=""
    local __pytest_marks=""
    local __hllog="3"
    local __suite_type="all"
    local __test_status=0
    local config_file="${__pytorch_qa_test_path}/config/test_order_config.txt"
    local _not_set_testpath=0
    local __aurora_path="${HABANA_PYTORCH_QA_ROOT}/../aurora"
    local __dut="gaudi"

    # By default the tox venv installs all the python modules from the external world instead of cached data.
    # This is because there is no config file which pip can use to get this info.
    # So populate the PIP env variable to point to the same pip.conf as in CI env. This file is custom made
    # and points to habana artifactory cache.
    export PIP_CONFIG_FILE="${VIRTUAL_ENV}/pip.conf"

    # parameter while-loop
    while [ -n "$1" ];
    do
        case $1 in
        -s  | --specific-test )
            shift
            __filter="-k $1"
            ;;
        -m  | --maxfail )
            shift
            __failures="--maxfail=$1"
            ;;
        -x  | --xml )
            shift
            IFS=. read __xml sfx <<< $1
            ;;
        -a  | --mark )
            shift
            __pytest_marks="-m=$1"
            ;;
        -hllog )
            shift
            __hllog=$1
            ;;
        -t | --suite-type )
            shift
            __suite_type="$1"
            ;;
        --dut )
            shift
            __dut="$1"
            ;;
        -l  | --list-tests )
            __print_tests="yes"
            ;;
        --no-color )
            __color="--color=no"
            ;;
        -h  | --help )
            usage $__scriptname
            return 0
            ;;
        *)
            echo "The parameter $1 is not allowed"
            usage $__scriptname
            return 1 # error
            ;;
        esac
        shift
    done

    case $__suite_type in
    all)
        echo "List or Run 'all' tests"
        ;;
    ops)
        _not_set_testpath=1
        __pytorch_qa_test_path+="/../torch_feature_val/single_op/"
        ;;
    dtypes_val)
        _not_set_testpath=1
        __pytorch_qa_test_path+="/../torch_feature_val/misc/dtypes_validation/"
        ;;
    topology)
        _not_set_testpath=1
        __pytorch_qa_test_path+="/topologies_tests/CI_tests/"
        ;;
    perf)
        _not_set_testpath=1
        __pytorch_qa_test_path+="/topologies_tests/perf_tests/"
        ;;
    acc)
        _not_set_testpath=1
        __pytorch_qa_test_path+="/topologies_tests/accuracy_tests/"
        ;;
    distributed)
        _not_set_testpath=1
        __pytorch_qa_test_path+="/distributed_tests/"
        ;;
    topology_ci)
        #set the default habanaqa path; the path is set in the code
        ;;
    subgraph)
        _not_set_testpath=1
        __pytorch_qa_test_path+="/../torch_feature_val/subgraph/"
        ;;
    *)
        echo "Test suite type \"$__suite_type\" is not allowed"
        usage $__scriptname
        return 1 # error
        ;;
    esac

    if [ -n "$__print_tests" ]; then
        pushd ${__pytorch_qa_test_path}
        (set -x; $__python_cmd -m pytest -v ${__pytest_marks} --collect-only)
        __test_status=$?
        popd
        return $__test_status
    fi
    pushd ${__pytorch_qa_test_path}

    opts="$__python_cmd -m pytest -v -o junit_logging=all ${__failures} ${__filter} ${__color} "${__pytest_marks}""
    #Adding a separate new variable only for single op params
    opts_single_op="$__python_cmd -m pytest -v -o junit_logging=all ${__failures} ${__filter} ${__color} "${__pytest_marks}" -n 1 --dut ${__dut} "
    test_path=""

    if [ "$__pytest_marks" == "-m=smoke" ] && [ "$__suite_type" == "ops" ] && [ "${__dut}" == "gaudi" ]; then
        #run pytorch single_op tests with suite_type = ops
        # run in eager mode
       (set -x; LOCK_GAUDI_SYNAPSE_API=1 ENABLE_CONSOLE=true PYTHONPATH="$PYTORCH_TESTS_ROOT" $opts_single_op ${__pytorch_qa_test_path} "--junit-xml=${__xml}_"single_op.xml" ")
        __test_status_1=$?
       (set -x; LOCK_GAUDI_SYNAPSE_API=1 ENABLE_CONSOLE=true PYTHONPATH="$PYTORCH_TESTS_ROOT" $opts_single_op ${__pytorch_qa_test_path} "--junit-xml=${__xml}_"strided_lazy_single_op.xml"" --mode lazy --strided)
        __test_status_2=$?
        __test_status=$((__test_status_1 | __test_status_2))
    elif [ "$__pytest_marks" == "-m=drs_dynamic_smoke" ] && [ "$__suite_type" == "ops" ]; then
       (set -x; LOCK_GAUDI_SYNAPSE_API=1 ENABLE_CONSOLE=true PYTHONPATH="$PYTORCH_TESTS_ROOT" $opts_single_op ${__pytorch_qa_test_path} "--junit-xml=${__xml}_"single_op_drs_dynamic.xml" " --mode lazy --drs 3 --dynamic)
        __test_status=$?
    elif [ "$__pytest_marks" == "-m=smoke" ] && [ "$__suite_type" == "ops" ] && [ "${__dut}" == "gaudi2" ]; then
       (set -x; LOCK_GAUDI_SYNAPSE_API=1 ENABLE_CONSOLE=true PYTHONPATH="$PYTORCH_TESTS_ROOT" $opts_single_op ${__pytorch_qa_test_path} "--junit-xml=${__xml}_"gc_eager_single_op.xml"" --mode gc_eager)
        __test_status_1=$?
       (set -x; LOCK_GAUDI_SYNAPSE_API=1 ENABLE_CONSOLE=true PYTHONPATH="$PYTORCH_TESTS_ROOT" $opts_single_op ${__pytorch_qa_test_path} "--junit-xml=${__xml}_"lazy_single_op.xml"" --mode lazy )
        __test_status_2=$?
        __test_status=$((__test_status_1 | __test_status_2))
    elif [ "$__pytest_marks" == "-m=smoke" ] && [ "$__suite_type" == "ops" ] && [ "${__dut}" == "greco" ]; then
       (set -x; LOCK_GAUDI_SYNAPSE_API=1 ENABLE_CONSOLE=true PYTHONPATH="$PYTORCH_TESTS_ROOT" $opts_single_op ${__pytorch_qa_test_path} "--junit-xml=${__xml}_"lazy_single_op.xml"" --mode lazy )
        __test_status=$?
    elif [ "$__pytest_marks" == "-m=dsd_subgraph" ] && [ "$__suite_type" == "subgraph" ]; then
       (set -x; LOCK_GAUDI_SYNAPSE_API=1 ENABLE_CONSOLE=true PYTHONPATH="$PYTORCH_TESTS_ROOT" $opts_single_op ${__pytorch_qa_test_path} "--junit-xml=${__xml}_"dynamic_subgraph.xml" " --mode lazy --dynamic)
        __test_status=$?
    elif [ "$__pytest_marks" == "-m=smoke" ] && [ "$__suite_type" == "dtypes_val" ]; then
       (set -x; LOCK_GAUDI_SYNAPSE_API=1 ENABLE_CONSOLE=true PYTHONPATH="$PYTORCH_TESTS_ROOT" $opts_single_op ${__pytorch_qa_test_path} "--junit-xml=${__xml}_"dtypes_val.xml"" )
        __test_status=$?
    else
       if [ "$__pytest_marks" == "-m=smoke" ] && [ "$__suite_type" == "all" ]; then
            #run single_op and topologies smoke tests
            (set -x; LOCK_GAUDI_SYNAPSE_API=1 ENABLE_CONSOLE=true PYTHONPATH="$PYTORCH_TESTS_ROOT" $opts_single_op ${__pytorch_qa_test_path}"/../torch_feature_val/single_op" "--junit-xml=${__xml}_"single_op.xml" ")
            test_path="topologies_tests"

       elif [ "$__suite_type" == "topology_ci" ]; then
            #run topology smoke tests  with suite_type topology
            test_path="topologies_tests"
       elif [ "$__pytest_marks" == "-m=smoke_dist" ] || [ "$__pytest_marks" == "-m=smoke_dist_gaudi2" ] || [ "$__pytest_marks" == "-m=smoke_dist_gaudi3" ] && [ "$__suite_type" == "distributed" ]; then
          #run distributed tests other than topology
            (set -x; LOCK_GAUDI_SYNAPSE_API=1 $opts ${__pytorch_qa_test_path}/dist_operations "--junit-xml=${__xml}_"distributed_ci.xml"")
            __test_status=$?
            return ${__test_status}
       fi

       # Add support for pytest's record_property
       opts="$opts -o junit_family=xunit1"

       if [[ $_not_set_testpath -ne 1 ]]; then
          test_path="topologies_tests"
       fi

       # Seperate common pytest command into forked and non-forked versions
       opts_forked="$opts --forked"

       $__python_cmd  $PYTORCH_TESTS_ROOT/tests/torch_training_tests/utils/tox_env_collector.py --marker "'${__pytest_marks}'" --opts "'${opts_forked}'" --log-level $__hllog --test-path ${__pytorch_qa_test_path}${test_path} --test-suite $__suite_type
       __test_status=$?
    fi
    popd

    # Don't cleas up the requirement python packages
    # for pytest in case of reproduction environment.
    # Otherwise, clean since the ifference in package versions can
    # cause dependencies between different SW versions
    if [ "$REPRODUCTION_ENV" != "yes" ]
    then
        __clean_pytest_dev_py_deps
    fi

    return ${__test_status}
}

install_requirements_pytorch()
{
    $__pip_cmd uninstall -y wrapt requests gast
    $__sudo -H $__pip_cmd uninstall -y wrapt requests gast
    cmd=($__pip_cmd install -r ${PYTORCH_MODULES_ROOT_PATH}/.ci/requirements/requirements-pytorch-${__python_cmd}_base.txt)
    if ! __running_in_venv; then
        cmd+=(--user)
        install_cmd+=(--user)
    fi
    "${install_cmd[@]}"
    "${cmd[@]}"
}

run_habana_lightning_tests()
{
    local __habana_lightning_tests_exe="python -m pytest"
    local __scriptname=$(__get_func_name)
    local __xml=""
    local __ld_lib="$BUILD_ROOT_RELEASE"
    local __print_tests=""
    local __py_filter=""
    local __failures=""
    local __marker=""
    local __verbose=""
    local __test_status=0
    local __suite_type="all"
    local __dut="gaudi"
    local __hllog=3

    # parameter while-loop
    while [ -n "$1" ];
    do
        case $1 in
        -l  | --list-tests )
            __print_tests="yes"
            ;;
        -s  | --specific-test )
            shift
            __py_filter="-k $1"
            ;;
        -m  | --maxfail )
            shift
            __failures="--maxfail=$1"
            ;;
        -p  | --pdb )
            shift
            __pdb="--pdb"
            ;;
        -t | --suite-type )
            shift
            __suite_type="$1"
            ;;
	     --dut )
            shift
            __dut="$1"
            ;;
        -hllog )
            shift
            __hllog=$1
            ;;
        -x  | --xml )
            shift
            __xml="$1"
            ;;
        -a | --marker )
            shift
            __marker="-m \"$1\""
            ;;
        -h  | --help )
            usage $__scriptname
            return 0
            ;;
        *)
            echo "The parameter $1 is not allowed"
            usage $__scriptname
            return 1 # error
            ;;
        esac
        shift
    done

    if [ -n "$__print_tests" ]; then
        pushd $HABANA_LIGHTNING_PLUGINS_ROOT/tests/
        echo "python tests:"
        ${__habana_lightning_tests_exe} --collectonly
        __test_status=$?
        popd
        return $__test_status
    fi

    if [[ "$__suite_type" = "all" || "$__suite_type" = "py_tests" ]] ; then
        pushd $HABANA_LIGHTNING_PLUGINS_ROOT/tests/
        (set -x; eval ${__habana_lightning_tests_exe} -v $__failures $__py_filter --junit-xml="${__xml}ptl_plugin_uts.xml" ${__marker})
        __test_status=$?
        popd
    fi

    # return error code of the tests
    return ${__test_status}
}


run_pytorch_lightning_qa_tests()
{
    local __pytorch_lightning_qa_tests_exe="python -m pytest"
    local __scriptname=$(__get_func_name)
    local __xml=""
    local __ld_lib="$BUILD_ROOT_RELEASE"
    local __print_tests=""
    local __py_filter=""
    local __failures=""
    local __marker=""
    local __verbose=""
    local __test_status=0
    local __suite_type="all"
    local __dut="gaudi"
    local __hllog=3

    # parameter while-loop
    while [ -n "$1" ];
    do
        case $1 in
        -l  | --list-tests )
            __print_tests="yes"
            ;;
        -s  | --specific-test )
            shift
            __py_filter="-k $1"
            ;;
        -m  | --maxfail )
            shift
            __failures="--maxfail=$1"
            ;;
        -p  | --pdb )
            shift
            __pdb="--pdb"
            ;;
        -t | --suite-type )
            shift
            __suite_type="$1"
            ;;
	     --dut )
            shift
            __dut="$1"
            ;;
        -hllog )
            shift
            __hllog=$1
            ;;
        -x  | --xml )
            shift
            __xml="$1"
            ;;
        -a | --marker )
            shift
            __marker="-m \"$1\""
            ;;
        -h  | --help )
            usage $__scriptname
            return 0
            ;;
        *)
            echo "The parameter $1 is not allowed"
            usage $__scriptname
            return 1 # error
            ;;
        esac
        shift
    done

    if [ -n "$__print_tests" ]; then
        pushd $PYTORCH_LIGHTNING_FORK_ROOT/tests/
        echo "python tests:"
        ${__pytorch_lightning_qa_tests_exe} --collectonly tests_pytorch/accelerators/test_hpu.py  tests_pytorch/plugins/precision/hpu/test_hpu.py
        __test_status=$?
        popd
        return $__test_status
    fi

    if [[ "$__suite_type" = "all" || "$__suite_type" = "py_tests" ]] ; then
        pushd $PYTORCH_LIGHTNING_FORK_ROOT/tests/

        echo "Executing Single card HPU test"
        (set -x; eval ${__pytorch_lightning_qa_tests_exe} -v $__failures $__py_filter tests_pytorch/accelerators/test_hpu.py --forked --junit-xml="${__xml}ptl_fw_uts.xml" ${__marker})
        ((__test_status=__test_status || $?))

        echo "Executing Multi card HPU test"
        (set -x; eval ${__pytorch_lightning_qa_tests_exe} -v $__failures $__py_filter tests_pytorch/accelerators/test_hpu.py --forked --hpus 8 --junit-xml="${__xml}ptl_fw_uts_8.xml" ${__marker})
        ((__test_status=__test_status || $?))

        echo "Executing HPU Precision test"
        (set -x; eval ${__pytorch_lightning_qa_tests_exe} -v $__failures $__py_filter tests_pytorch/plugins/precision/hpu/test_hpu.py --hmp-bf16 \
                'tests_pytorch/plugins/precision/hpu/ops_bf16.txt' --hmp-fp32 \
                'tests_pytorch/plugins/precision/hpu/ops_fp32.txt' --forked \
                "${__xml}ptl_fw_uts_precision.xml" ${__marker})
        ((__test_status=__test_status || $?))

        popd
    fi

    # return error code of the tests
    return ${__test_status}
}

run_lightning_habana_fw_tests()
{
    local __pytorch_lightning_qa_tests_exe="python -m pytest"
    local __scriptname=$(__get_func_name)
    local __xml=""
    local __ld_lib="$BUILD_ROOT_RELEASE"
    local __print_tests=""
    local __py_filter=""
    local __failures=""
    local __marker=""
    local __verbose=""
    local __test_status=0
    local __suite_type="all"
    local __dut="gaudi"
    local __hllog=3

    # parameter while-loop
    while [ -n "$1" ];
    do
        case $1 in
        -l  | --list-tests )
            __print_tests="yes"
            ;;
        -s  | --specific-test )
            shift
            __py_filter="-k $1"
            ;;
        -m  | --maxfail )
            shift
            __failures="--maxfail=$1"
            ;;
        -p  | --pdb )
            shift
            __pdb="--pdb"
            ;;
        -t | --suite-type )
            shift
            __suite_type="$1"
            ;;
	     --dut )
            shift
            __dut="$1"
            ;;
        -hllog )
            shift
            __hllog=$1
            ;;
        -x  | --xml )
            shift
            __xml="$1"
            ;;
        -a | --marker )
            shift
            __marker="-m \"$1\""
            ;;
        -h  | --help )
            usage $__scriptname
            return 0
            ;;
        *)
            echo "The parameter $1 is not allowed"
            usage $__scriptname
            return 1 # error
            ;;
        esac
        shift
    done

    if [ -n "$__print_tests" ]; then
        pushd $LIGHTNING_HABANA_FORK_ROOT/tests/
        echo "Fabric tests:"
        ${__pytorch_lightning_qa_tests_exe} --collectonly tests_fabric/
        __test_status=$?

        echo "Lightning HPU tests:"
        ${__pytorch_lightning_qa_tests_exe} --collectonly tests_pytorch/
        __test_status=$?

        popd
        return $__test_status
    fi

    if [[ "$__suite_type" = "all" || "$__suite_type" = "py_tests" ]] ; then
        pushd $LIGHTNING_HABANA_FORK_ROOT/tests/

        echo "Executing Lightning fabric tests on HPU"
        (set -x; eval ${__pytorch_lightning_qa_tests_exe} -v $__failures $__py_filter tests_fabric/ --forked --junit-xml="${__xml}fabric_fw_uts.xml" ${__marker})
        ((__test_status=__test_status || $?))

        echo "Executing Lightning Habana tests"
        (set -x; eval ${__pytorch_lightning_qa_tests_exe} -v $__failures $__py_filter tests_pytorch \
         --hmp-bf16 'tests_pytorch/plugins/precision/hpu/ops_bf16.txt'\
         --hmp-fp32 'tests_pytorch/plugins/precision/hpu/ops_fp32.txt' --forked --hpus 8 --junit-xml="${__xml}lightning_fw_uts_8.xml" ${__marker})
        ((__test_status=__test_status || $?))

        popd
    fi

    # return error code of the tests
    return ${__test_status}
}

install_lightning_plugin()
{
    $__pip_cmd install habana-lightning-plugins --extra-index-url https://artifactory-kfs.habana-labs.com/artifactory/api/pypi/habana_pypi/simple
}

install_requirements_pytest()
{
    $__pip_cmd uninstall -y wrapt requests gast
    $__sudo -H $__pip_cmd uninstall -y wrapt requests gast
    cmd=($__pip_cmd install -r ${PYTORCH_MODULES_ROOT_PATH}/.ci/requirements/requirements-pytest-$__python_cmd.txt)
    if ! __running_in_venv; then
        cmd+=(--user)
    fi
    "${cmd[@]}"
}

uninstall_requirements_pytest()
{
    cmd=($__pip_cmd uninstall -r ${PYTORCH_MODULES_ROOT_PATH}/.ci/requirements/requirements-pytest-$__python_cmd.txt -y)
    "${cmd[@]}"
}

clean_pytorch_pkgs()
{
    $__pip_cmd uninstall -y hb-torch torch hmp gather2d-cpp HabanaEmbeddingBag-cpp habanaOptimizerSparseSgd-cpp preproc-cpp habanaOptimizerSparseAdagrad-cpp habana-torch-dataloader habana-torch
    $__sudo -H $__pip_cmd uninstall -y hb-torch torch hmp gather2d-cpp HabanaEmbeddingBag-cpp habanaOptimizerSparseSgd-cpp preproc-cpp habanaOptimizerSparseAdagrad-cpp habana-torch-dataloader habana-torch
}

__check_pytorch_dev_py_deps()
{
    install_requirements_pytorch
}

__check_pytest_dev_py_deps()
{
    install_requirements_pytest
}

__check_lightning_plugin_py_deps()
{
    install_lightning_plugin
}

__clean_pytorch_dev_py_deps()
{
    clean_pytorch_pkgs
}

__clean_pytest_dev_py_deps()
{
    uninstall_requirements_pytest
}
__install_anaconda()
{
    local __conda_res

    #Check if conda env exist
    if $HOME/anaconda3/bin/conda list > /dev/null 2>&1; then
        echo "Found existing conda installation"
    else
        echo "Conda installation not found in default path, Installing..."

        local __conda_installer=Anaconda3-2020.02-Linux-x86_64.sh
        (set -x;wget https://repo.anaconda.com/archive/$__conda_installer 2> /dev/null)
        __conda_res=$?
        if [ $__conda_res -ne 0 ]; then
            echo "Conda download failed!"
            return $__conda_res
        fi
        (set -x; bash  $__conda_installer -b -f)
        __conda_res=$?
        rm -rf $__conda_installer
        if [ $__conda_res -ne 0 ]; then
            echo "Conda installation failed!"
            return $__conda_res
        fi
    fi

    if [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
        source "$HOME/anaconda3/etc/profile.d/conda.sh"
    else
        export PATH="$HOME/anaconda3/bin:$PATH"
    fi

    local __conda_venv=py_venv_$__python_ver
    if $(conda env list | awk '{print $1}'| grep -x $__conda_venv) > /dev/null 2>&1; then
        echo "Found Conda so 1st deactivating it and then deleting"
        __conda deactivate
        echo "Removing Conda env $__conda_venv"
        __conda env remove --name $__conda_venv
    fi

    # Create conda env with specific version of python
    echo "Creating conda venv with Python ver=$__python_ver"
    (set -x;__conda create --name $__conda_venv python=$__python_ver -y)
    __conda_res=$?
    if [ $__conda_res -ne 0 ]; then
        echo "Conda env creation failed!"
        return $__conda_res
    fi

    __conda activate $__conda_venv
    __conda_res=$?
    if [ $__conda_res -ne 0 ]; then
        echo "Conda env activation failed!"
        return $__conda_res
    fi
    echo "Activated  conda venv $__conda_venv"
    local __pip_conf=$VIRTUAL_ENV/pip.conf
    if [ -f "$__pip_conf"  ]; then
        export PIP_CONFIG_FILE=$__pip_conf
    fi

    $__python_cmd -m pip install -r ${PYTORCH_MODULES_ROOT_PATH}/.ci/requirements/requirements-pytorch-python${__python_ver}_base.txt
    __conda_res=$?
    if [ $__conda_res -ne 0 ]; then
        echo "Conda package installation failed!"
        __conda deactivate
        return $__conda_res
    fi
    __conda deactivate
    printf "Installation of conda packages done\n"
    return $__conda_res
}

# SW-40601 Workaround to uninstall torchvision and install habana-torchvision in Pytorch CI
install_habana_torchvision()
{
    $__pip_cmd uninstall -y torchvision
    cmd=($__pip_cmd install habana-torchvision==0.10.0)
    if ! __running_in_venv; then
        cmd+=(--user)
    fi
    "${cmd[@]}"
    $__pip_cmd uninstall -y torch
    $__pip_cmd uninstall -y pillow
    $__pip_cmd uninstall -y pillow-simd
    cmd=($__pip_cmd install pillow-simd==7.0.0.post3)
    if ! __running_in_venv; then
        cmd+=(--user)
    fi
    "${cmd[@]}"
}

# Method to install pillow-simd which is required for performance
# pillow package gets pulled along with installation of torchvision
# So it is required to uninstall pillow and install pillow-simd
# after torchvision installation
install_pillow_simd()
{
    $__pip_cmd uninstall -y pillow
    $__pip_cmd uninstall -y pillow-simd
    cmd=($__pip_cmd install pillow-simd==7.0.0.post3)
    if ! __running_in_venv; then
        cmd+=(--user)
    fi
    "${cmd[@]}"
}

# set_python_version to set envs related to python version during build
set_python_version()
{
    case $1 in
    "3.6" | "3.7" | "3.8" | "3.10")
        echo "version $1"
        ;;
    *)
        echo "Usage: $0 <3.7/3.8/3.10>"
        return
        ;;
    esac
    export __old_python_ver=$__python_ver
    export __old_python_cmd=$__python_cmd
    export __old_pip_cmd=$__pip_cmd
    export __python_ver=$1
    export __python_cmd="python${__python_ver}"
    export __pip_cmd="${__python_cmd} -m pip"
    echo "__python_ver = ${__python_ver}"
    echo "__python_cmd = ${__python_cmd}"
    echo "__pip_cmd    = ${__pip_cmd}"
}

restore_python_version()
{
    if [ "z$__set_py_vers" == "ztrue" ]; then
        [ "z$__old_python_ver" != "z" ] && export __python_ver=$__old_python_ver
        [ "z$__old_pip_cmd" != "z" ] && export __pip_cmd=$__old_pip_cmd
        [ "z$__old_python_cmd" != "z" ] && export __python_cmd=$__old_python_cmd
    fi
}

build_pytorch_text()
{
    SECONDS=0
    local __scriptname=$(__get_func_name)
    local __env_vars=""
    local __configure=""
    local __whl_params=" bdist_wheel"
    local __result
    local __set_py_vers="false"
    local __profile_getter_path="${PYTORCH_MODULES_ROOT_PATH}/.devops/profile_getter.py"
    local __pt_text_version
    # parameter while-loop
    while [ -n "$1" ];
    do
        case $1 in
        -j  | --jobs )
            __env_vars+=" MAX_JOBS=$2"
            shift
            ;;
        -c  | --configure )
             __configure="yes"
            ;;
        -r  | --release )
            ;;
        -d  | --debug )
            ;;
        -a  | --build-all )
            ;;
        --dist )
            __whl_params=" bdist_wheel"
            ;;
        --install )
            __whl_params=" install"
            ;;
        --py-version )
            set_python_version $2
            __set_py_vers="true"
            shift
            ;;
        --pt-text-version )
            __pt_text_version="$2"
            shift
            ;;
        -h  | --help )
            usage $__scriptname
            restore_python_version
            return 0
            ;;
        esac
        shift
    done

    rm -rf $PYTORCH_TEXT_ROOT
    mkdir -p $PYTORCH_TEXT_ROOT
    pushd $PYTORCH_TEXT_ROOT

    # checkout github torchaudio repo
    if [ -z ${__pt_text_version} ]; then
        __pt_text_version=$($__profile_getter_path --get-extras-version torchtext current)
    fi
    echo "get torchtext from github (tag: $__pt_text_version)"
    git clone https://github.com/pytorch/text --branch v$__pt_text_version --single-branch --depth 1 .
    git submodule update --init --recursive

    if [ -n "$__configure" ]; then
        $__python_cmd setup.py clean
        git clean -fd
    fi

    echo "Build parameters ${__whl_params}"

    (set -x;eval ${__env_vars} $__python_cmd setup.py ${__whl_params})
    __result=$?
    if [ $__result -ne 0 ]; then
        echo "Pytorch torchtext build failed!"
    fi
    PTT_WHL_PATH="$PYTORCH_TEXT_ROOT/dist/"

    popd
    if [[ "$__whl_params" = " bdist_wheel" ]]; then
        rm -rf $PYTORCH_TEXT_BUILD/pkgs
        mkdir -p $PYTORCH_TEXT_BUILD/pkgs
        cp -f ${PTT_WHL_PATH}/*.whl $PYTORCH_TEXT_BUILD/pkgs
    fi

    printf "\nElapsed time: %02u:%02u:%02u \n\n" $(($SECONDS / 3600)) $((($SECONDS / 60) % 60)) $(($SECONDS % 60))
    restore_python_version
    return $__result
}

build_pytorch_data()
{
    SECONDS=0
    local __scriptname=$(__get_func_name)
    local __env_vars=""
    local __configure=""
    local __whl_params=" bdist_wheel"
    local __result
    local __set_py_vers="false"
    local __profile_getter_path="${PYTORCH_MODULES_ROOT_PATH}/.devops/profile_getter.py"
    local __pt_data_version
    # parameter while-loop
    while [ -n "$1" ];
    do
        case $1 in
        -j  | --jobs )
            __env_vars+=" MAX_JOBS=$2"
            shift
            ;;
        -c  | --configure )
             __configure="yes"
            ;;
        -r  | --release )
            ;;
        -d  | --debug )
            ;;
        --dist )
            __whl_params=" bdist_wheel"
            ;;
        --install )
            __whl_params=" install"
            ;;
        --py-version )
            set_python_version $2
            __set_py_vers="true"
            shift
            ;;
        --pt-data-version )
            __pt_data_version="$2"
            shift
            ;;
        -h  | --help )
            usage $__scriptname
            restore_python_version
            return 0
            ;;
        esac
        shift
    done

    rm -rf $PYTORCH_DATA_ROOT
    mkdir -p $PYTORCH_DATA_ROOT
    pushd $PYTORCH_DATA_ROOT

    # checkout github torchaudio repo
    if [ -z ${__pt_data_version} ]; then
        __pt_data_version=$($__profile_getter_path --get-extras-version torchdata current)
    fi
    echo "get torchdata from github (tag: $__pt_data_version)"
    git clone https://github.com/pytorch/data --branch v$__pt_data_version --single-branch --depth 1 .
    git submodule update --init --recursive

    if [ -n "$__configure" ]; then
        $__python_cmd setup.py clean
        git clean -fd
    fi

    echo "Build parameters ${__whl_params}"

    (set -x;eval ${__env_vars} $__python_cmd setup.py ${__whl_params})
    __result=$?
    if [ $__result -ne 0 ]; then
        echo "Pytorch torchdata build failed!"
    fi
    PTD_WHL_PATH="$PYTORCH_DATA_ROOT/dist/"

    popd
    if [[ "$__whl_params" = " bdist_wheel" ]]; then
        rm -rf $PYTORCH_DATA_BUILD/pkgs
        mkdir -p $PYTORCH_DATA_BUILD/pkgs
        cp -f ${PTD_WHL_PATH}/*.whl $PYTORCH_DATA_BUILD/pkgs
    fi

    printf "\nElapsed time: %02u:%02u:%02u \n\n" $(($SECONDS / 3600)) $((($SECONDS / 60) % 60)) $(($SECONDS % 60))
    restore_python_version
    return $__result
}

build_pytorch_audio()
{
    SECONDS=0
    local __scriptname=$(__get_func_name)
    local __env_vars="PATH=/opt/bin:$PATH USE_CUDA=0 BUILD_RNNT=0"
    local __configure=""
    local __whl_params=" bdist_wheel"
    local __result
    local __set_py_vers="false"
    local __profile_getter_path="${PYTORCH_MODULES_ROOT_PATH}/.devops/profile_getter.py"
    local __pt_audio_version
    # parameter while-loop
    while [ -n "$1" ];
    do
        case $1 in
        -j  | --jobs )
            __env_vars+=" MAX_JOBS=$2"
            shift
            ;;
        -c  | --configure )
             __configure="yes"
            ;;
        -r  | --release )
            ;;
        -d  | --debug )
            ;;
        --dist )
            __whl_params=" bdist_wheel"
            ;;
        --install )
            __whl_params=" install"
            ;;
        --py-version )
            set_python_version $2
            __set_py_vers="true"
            shift
            ;;
        --pt-audio-version )
            __pt_audio_version="$2"
            shift
            ;;
        -h  | --help )
            usage $__scriptname
            restore_python_version
            return 0
            ;;
        esac
        shift
    done

    rm -rf $PYTORCH_AUDIO_ROOT
    mkdir -p $PYTORCH_AUDIO_ROOT
    pushd $PYTORCH_AUDIO_ROOT

    # checkout github torchaudio repo
    if [ -z ${__pt_audio_version} ]; then
        __pt_audio_version=$($__profile_getter_path --get-extras-version torchaudio current)
    fi
    echo "get torchaudio from github (tag: $__pt_audio_version)"
    git clone https://github.com/pytorch/audio --branch v$__pt_audio_version --single-branch --depth 1 .
    git submodule update --init --recursive

    if [ -n "$__configure" ]; then
        $__python_cmd setup.py clean
        git clean -fd
    fi

    echo "Build parameters ${__whl_params}"

    (set -x;eval ${__env_vars} $__python_cmd setup.py ${__whl_params})
    __result=$?
    if [ $__result -ne 0 ]; then
        echo "Pytorch torchaudio build failed!"
    fi
    PTA_WHL_PATH="$PYTORCH_AUDIO_ROOT/dist/"

    popd
    if [[ "$__whl_params" = " bdist_wheel" ]]; then
        rm -rf $PYTORCH_AUDIO_BUILD/pkgs
        mkdir -p $PYTORCH_AUDIO_BUILD/pkgs
        cp -f ${PTA_WHL_PATH}/*.whl $PYTORCH_AUDIO_BUILD/pkgs
    fi

    printf "\nElapsed time: %02u:%02u:%02u \n\n" $(($SECONDS / 3600)) $((($SECONDS / 60) % 60)) $(($SECONDS % 60))
    restore_python_version
    return $__result
}