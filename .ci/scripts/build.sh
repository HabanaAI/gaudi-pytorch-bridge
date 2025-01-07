#!/usr/bin/env bash
###############################################################################
#
#  Copyright (c) 2021-2025 Intel Corporation
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
###############################################################################

function pytorch_functions_help()
{
    echo -e "\n- The following is a list of available functions for PyTorch"
    echo -e "build_pytorch_fork             -   Build the habana pytorch fork"
    echo -e "build_pytorch_vision_fork      -   Build the habana pytorch vision fork"
    echo -e "build_pytorch_modules          -   Build habana pytorch intergation modules"
    echo -e "build_pytorch_dist             -   Build habana pytorch distrubuted modules"
    echo -e "build_pytorch_tb_plugin        -   Build habana pytorch tensorboard plugin"
    echo -e "build_lightning_habana_fork    -   Build lightning habana fork"
    echo -e "build_pytorch_data             -   Build habana pytorch data"
    echo -e "build_pytorch_text             -   Build habana pytorch text"
    echo -e "build_pytorch_audio            -   Build habana pytorch audio"
    echo -e "run_pytorch_qa_tests           -   Run pytorch QA tests"
    echo -e "run_pytorch_modules_tests      -   Run pytorch modules tests"
    echo -e "run_habana_lightning_tests     -   Run habana lightning plugin tests"
    echo -e "run_lightning_habana_fw_tests  -   Run Lightning Habana tests"
    echo -e "build_pytorch_vision           -   Build the habana pytorch vision"
}

function pytorch_usage()
{
    if [ $1 == "build_pytorch_fork" ]; then
        echo -e "\n usage: $1 [options]\n"

        echo -e "options:\n"
        echo -e "  -j,  --jobs <val>           Max jobs used for compilation"
        echo -e "  -c,  --clean                clean up temporary files from 'build' command"
        echo -e "  -r,  --release              Build only release build"
        echo -e "       --recursive            Build all the pre-requisite modules in a recursive way"
        echo -e "  -s,  --sanitize             Build with sanitize flags on"
        echo -e "  -d,  --debug                Build only debug build"
        echo -e "       --install              will install the package"
        echo -e "       --dist                 create a wheel distribution"
        echo -e "       --build-number         Extend whl version number by build number"
        echo -e "       --build-version        Build version used for whl creation"
        echo -e "       --pytorch-next         Build pytorch-next instead of pytorch-fork"
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
        echo -e "       --no-fe                Disable front-end build"
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
        echo -e "  -t,  --suite-type TYPE              Run specific suite type [all, py_tests, cpp_tests, cpp_lazy, cpp_eager]. Default: all"
        echo -e "  -c,  --test-case JUNITID            Run specific test based on JUnit ID"
        echo -e "       --pytest-mode                  Run specific pytest suite mode: [all, lazy, compile, eager]. Default: all"
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
    if [ $1 == "build_pytorch_vision" ]; then
        echo -e "\n usage: $1 [options]\n"

        echo -e "options:\n"
        echo -e "  -j,  --jobs <val>           Max jobs used for compilation"
        echo -e "  -c,  --clean                clean up temporary files from 'build' command"
        echo -e "  -r,  --release              Python only code, option ignored"
        echo -e "  -d,  --debug                Python only code, option ignored"
        echo -e "       --install              will install the package"
        echo -e "       --dist                 create a wheel distribution/default"
        echo -e "       --py-version           Python version"
        echo -e "       --pt-vision-version    Pytorch Vision version"
        echo -e "  -h,  --help                 Prints this help"
    fi
}

__error() {
    echo "ERROR:" "$@" >&2
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

build_pytorch_fork()
{
    SECONDS=0

    local __scriptname=$(__get_func_name)
    local __env_vars="DEBUG=1 USE_MPI=OFF BUILD_TEST=0 INSTALL_TEST=0"
    local __configure=""
    local __release=""
    local __debug="yes"
    local __whl_params=" bdist_wheel"
    local __pt_vers=""
    local __result
    local __pt_fork_tag="pytorch_fork_tags"
    local __pt_fork_vers="pytorch_fork_version"
    local __default_vers="default_vers"
    local __branch=""
    local __build_manylinux_whl="false"
    local __auditwheel="${PYTORCH_MODULES_ROOT_PATH}/.ci/scripts/pt_auditwheel.py"
    local __set_py_vers="false"
    local __pytorch_next="false"

    # parameter while-loop
    while [ -n "$1" ];
    do
        case $1 in
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
            shift
            ;;
        --build-version )
            __env_vars+=" PYTORCH_BUILD_VERSION=$2"
            shift
            ;;
        --py-version )
             set_python_version $2
             __set_py_vers="true"
             shift
            ;;
        --recursive )
            # No-op. Fork has no dependencies.
            ;;
        -s  | --sanitize )
            __env_vars+=" USE_ASAN=ON"
            ;;
        --manylinux )
            __build_manylinux_whl="true"
            ;;
        --pytorch-next )
            __pytorch_next="true"
            ;;
        -h  | --help )
            usage $__scriptname
            restore_python_version
            return 0
            ;;
        -j  | --jobs )
            __env_vars+=" MAX_JOBS=$2"
            shift
            ;;
        -j* )
            __env_vars+=" MAX_JOBS=${1:2}"  # drop the first two characters
            ;;
        *)
            echo Invalid argument: $1
            usage $__scriptname
            restore_python_version
            return 1
        esac
        shift
    done

    unset CMAKE_ROOT  # we're using CMake from requirements files

    __provide_mkl || exit $?

    local __pytorch_root=${PYTORCH_FORK_ROOT}

    if [[ "z$__pytorch_next" == "ztrue" ]]; then
        __pytorch_root=${PYTORCH_NEXT_ROOT}
    fi
    pushd $__pytorch_root
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

    if [ -n "$__configure" ]; then
        $__python_cmd setup.py clean
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
        popd
        restore_python_version
        return $__result
    fi

    if [ "z${__build_manylinux_whl}" == "ztrue" ];then
        bash -c "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$__pytorch_root/torch/lib;$__python_cmd $__auditwheel repair $__pytorch/dist/torch*.whl"
        TORCH_WHL_PATH="$__pytorch_root/wheelhouse/"
    else
        TORCH_WHL_PATH="$__pytorch_root/dist/"
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
    local __build_frontend="true"
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
        --no-fe )
            __build_frontend=""
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

    if [ -n "$__build_frontend" ]; then
        ./scripts/setup.sh
        __result=$?
        if [ $__result -ne 0 ]; then
            echo "front end setup failed"
            popd
            restore_python_version
            return $__result
        fi

        ./scripts/build.sh
        __result=$?
        if [ $__result -ne 0 ]; then
            echo "front end build failed"
            popd
            restore_python_version
            return $__result
        fi
    fi

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

    __provide_mkl || exit $?

    if [ -n "$__configure" ]; then
        $__python_cmd setup.py clean
        rm -rf ${LIGHTNING_HABANA_FORK_ROOT}/dist/*.whl
    fi

    echo "Build parameters ${__whl_params}"

    (set -x;eval ${__env_vars} $__python_cmd setup.py ${__whl_params})
    __result=$?
    if [ $__result -ne 0 ]; then
        echo "Pytorch torchvision build failed!"
    fi
    if [ "z${__build_manylinux_whl}" == "ztrue" ]; then
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
    local __cpp_tests_exe_eager="$PYTORCH_MODULES_RELEASE_BUILD/test_pt2_integration"
    local __pt_major_version=$(python -m pip list | grep  "^torch\s" | tr -s '[:space:]' | cut -d ' ' -f 2 | cut -d '.' -f 1)
    local __scriptname=$(__get_func_name)
    local __xml="test_detail.xml"
    local __ld_lib="$BUILD_ROOT_RELEASE"
    local __print_tests=""
    local __py_filter=""
    local __cpp_filter=""
    local __cpp_filter_eager=""
    local __failures=""
    local __marker=""
    local __verbose=""
    local __test_status=0
    local __suite_type="all"
    local __dut="gaudi"
    local __hllog=3
    local __test_case=""
    local __pytest_mode="all"

    source ${PYTORCH_MODULES_ROOT_PATH}/.ci/scripts/disabled_tests.sh
    local __disable_failing_eager_tests="--gtest_filter=-"`echo ${FAILING_EAGER_TESTS[@]} | tr ' ' ':'`

    # parameter while-loop
    while [ -n "$1" ];
    do
        case $1 in
        -d  | --debug )
            __cpp_tests_exe="$PYTORCH_MODULES_DEBUG_BUILD/test_pt_integration"
            __cpp_tests_exe_eager="$PYTORCH_MODULES_DEBUG_BUILD/test_pt2_integration"
            ;;
        -l  | --list-tests )
            __print_tests="yes"
            ;;
        -s  | --specific-test )
            shift
            __py_filter="-k $1"
            __cpp_filter="--gtest_filter=$1"
            __cpp_filter_eager="--gtest_filter=$1"
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
        -c  | --test-case )
            shift
            if [[ "$1" =~ \  ]]; then
               echo "Test case can't contain white space and only one test case can be provided."
               usage $__scriptname
               return 1
            fi
            __test_case="$1"
            ;;
        -pm | --pytest-mode )
            shift
            __pytest_mode="$1"
            if [[ "${__pytest_mode}" != "lazy" && "${__pytest_mode}" != "compile" && "${__pytest_mode}" != "eager" && "${__pytest_mode}" != "all" ]]; then
                echo "Pytest mode \"$__pytest_mode\" is not allowed"
                usage $__scriptname
                return 1 # error
            fi
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
    cpp_lazy)
        __test_type="cpp_lazy"
        ;;
    cpp_eager)
        __test_type="cpp_eager"
        ;;
    cpp_tests)
        __test_type="cpp_tests"
        ;;
    infra)
        __test_type="infra"
        ;;
    *)
        echo "Test suite type \"$__suite_type\" is not allowed"
        usage $__scriptname
        return 1 # error
        ;;
    esac

    #JUnitID for pytes tests has format
    #Pytest{Eager | Compile | Lazy}.pytest_working.{compile | eager | lazy | any_mode}.<test file>.<test name>[parameters]
    #JUnitID for gtest tests has format
    #Cpp{Eager | Lazy}.<test suite>.<test case>[parameters]
    if [ $__test_case ]; then
        local __test_string_regex="^([a-zA-Z0-9\._-]*)(\[.*\])?$"
        local __test_string=""
        local __test_parameters=""
        if ! [[ $__test_case =~ $__test_string_regex ]]; then
            echo "Incorrect JUnit ID provided"
            return 1
        fi
        __test_string=${BASH_REMATCH[1]}

        if [ ${BASH_REMATCH[2]} ]; then
            __test_parameters=${BASH_REMATCH[2]}
        fi
        __test_string=(${__test_string//./ })

        case ${__test_string[0]} in
        CppEager)
            echo "Specified test belongs to suite CppEager. Changing suite_type to cpp_eager"
            __suite_type="cpp_eager"
            __cpp_filter=${__test_string[@]:1}
            __cpp_filter="--gtest_filter=${__cpp_filter// /.}"
            echo $__cpp_filter
            ;;
        CppLazy)
            echo "Specified test belongs to suite CppLazy. Changing suite_type to cpp_lazy"
            __suite_type="cpp_lazy"
            __cpp_filter=${__test_string[@]:1}
            __cpp_filter="--gtest_filter=${__cpp_filter// /.}"
            ;;
        PytestLazy)
            echo "Specified test belongs to suite PytestLazy. Changing suite_type to py_test with mode lazy"
            __suite_type="py_tests"
            __pytest_mode="lazy"
            __py_filter="${__test_string[@]:4}${__test_parameters}"
            __py_filter="-k \"${__py_filter// /:}\""
            ;;
        PytestEager)
            echo "Specified test belongs to suite PytestEager. Changing suite_type to py_test with mode eager"
            __suite_type="py_tests"
            __pytest_mode="eager"
            __py_filter="${__test_string[@]:4}${__test_parameters}"
            __py_filter="-k \"${__py_filter// /:}\""
            ;;
        PytestCompile)
            echo "Specified test belongs to suite PytestCompile. Changing suite_type to py_test with mode compile"
            __suite_type="py_tests"
            __pytest_mode=compile
            __py_filter="${__test_string[@]:4}${__test_parameters}"
            __py_filter="-k \"${__py_filter// /:}\""
            ;;
        esac
    fi


    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${__ld_lib}
    if [ ! -n "$__print_tests" ]; then
        if [ "$__dut" == "gaudi" ]; then
            if [ "$__suite_type" == "all" ] || [ "$__suite_type" == "cpp_tests" ] || [ "$__suite_type" == "cpp_lazy" ]; then
                echo "Running tests on Gaudi"
                (set -x; eval LOG_LEVEL_ALL=${__hllog} $__cpp_tests_exe --gtest_output=xml:$__xml $__cpp_filter)
                __test_status=$?
            fi
            if [ $__pt_major_version -eq 2 ]; then
                if [ "$__suite_type" == "all" ] || [ "$__suite_type" == "cpp_tests" ] || [ "$__suite_type" == "cpp_eager" ]; then
                    (LOG_LEVEL_ALL=${__hllog} PT_HPU_LAZY_MODE=0 $__cpp_tests_exe_eager --gtest_output=xml:$__xml $__disable_failing_eager_tests $__cpp_filter)
                    __test_status=$((__test_status | $?))
                fi
            fi
        elif [ "$__dut" == "gaudi2" ]; then
            if [ "$__suite_type" == "all" ] || [ "$__suite_type" == "cpp_tests" ] || [ "$__suite_type" == "cpp_lazy" ]; then
                echo "Running tests on Gaudi2"
                (set -x; eval LOG_LEVEL_ALL=${__hllog} $__cpp_tests_exe --gtest_output=xml:$__xml $__cpp_filter)
                __test_status=$?
            fi
            if [ $__pt_major_version -eq 2 ]; then
                if [ "$__suite_type" == "all" ] || [ "$__suite_type" == "cpp_tests" ] || [ "$__suite_type" == "cpp_eager" ]; then
                    (LOG_LEVEL_ALL=${__hllog} PT_HPU_LAZY_MODE=0 $__cpp_tests_exe_eager --gtest_output=xml:$__xml $__disable_failing_eager_tests $__cpp_filter)
                    __test_status=$((__test_status | $?))
                fi
            fi
        elif [ "$__dut" == "gaudi3" ]; then
            if [ "$__suite_type" == "all" ] || [ "$__suite_type" == "cpp_tests" ] || [ "$__suite_type" == "cpp_lazy" ]; then
                echo "Running tests on Gaudi3"
                (set -x; eval LOG_LEVEL_ALL=${__hllog} $__cpp_tests_exe --gtest_output=xml:$__xml $__cpp_filter)
                __test_status=$?
            fi
            if [ $__pt_major_version -eq 2 ]; then
                if [ "$__suite_type" == "all" ] || [ "$__suite_type" == "cpp_tests" ] || [ "$__suite_type" == "cpp_eager" ]; then
                    (LOG_LEVEL_ALL=${__hllog} PT_HPU_LAZY_MODE=0 $__cpp_tests_exe_eager --gtest_output=xml:$__xml $__disable_failing_eager_tests $__cpp_filter)
                    __test_status=$((__test_status | $?))
                fi
            fi
        elif [ "$__dut" == "greco" ]; then
            echo "Running greco tests"
            (set -x; eval LOG_LEVEL_ALL=${__hllog} PT_HPU_INFERENCE_MODE=true $__cpp_tests_exe --gtest_output=xml:$__xml --gtest_filter=HpuOpTest*addmm*:HpuOpTest*addbmm*:*LayerNormForwardExecute*:*LazyConvKernel*Pool* $__cpp_filter)
                __test_status=$?
        fi
    fi

    if [ "$__suite_type" != "py_tests" ]; then
        if [ -f "$__xml/test_pt2_integration.xml" ]; then
            # Workaround for jenkins skipping duplicated test names within "AllTests" scope
            sed -i -E 's/classname="(.+)"/classname="CppEager.\1"/g' $__xml/test_pt2_integration.xml
        fi

        if [ -f "$__xml/test_pt_integration.xml" ]; then
            # Workaround for jenkins skipping duplicated test names within "AllTests" scope
            sed -i -E 's/classname="(.+)"/classname="CppLazy.\1"/g' $__xml/test_pt_integration.xml
        fi
    fi

    if [ -n "$__print_tests" ]; then
        pushd $HABANA_SOFTWARE_STACK/pytorch-integration/tests/
        echo "python tests:"
        ${__pytorch_modules_tests_exe} --collect-only
        __test_status=$((__test_status | $?))

        echo "cpp tests:"
        ${__cpp_tests_exe} --gtest_list_tests
        __test_status=$((__test_status | $?))

        if [ $__pt_major_version -eq 2 ]; then
            echo "cpp eager tests:"
            ${__cpp_tests_exe_eager} --gtest_list_tests
            __test_status=$((__test_status | $?))
        fi
        popd

        echo "infra tests:"
        pushd $PYTORCH_MODULES_ROOT_PATH/.devops/
        ${__pytorch_modules_tests_exe} tests/ --collect-only
        __test_status=$((__test_status | $?))
        popd
        pushd $PYTORCH_MODULES_ROOT_PATH/scripts/
        ${__pytorch_modules_tests_exe} tests/ --collect-only
        __test_status=$((__test_status | $?))
        popd
        pushd $PYTORCH_MODULES_ROOT_PATH/tests/user_custom_op/
        ${__pytorch_modules_tests_exe} . --collect-only
        __test_status=$((__test_status | $?))
        popd

        return $__test_status
    fi

    if [[ "$__suite_type" = "all" || "$__suite_type" = "py_tests" ]] ; then
        pushd $HABANA_SOFTWARE_STACK/pytorch-integration/tests/
        if [[ "$__pytest_mode" = "lazy" || "$__pytest_mode" = "all" ]] ; then
            (set -x; eval ${__pytorch_modules_tests_exe} pytest_working/ -v $__failures $__py_filter --junit-xml="${__xml}_lazy_pytest.xml" --mode="lazy" --dut="${__dut}" --junit-prefix="PytestLazy" ${__marker})
            __test_status=$((__test_status | $?))
            (set -x; eval PT_HPU_AUTOLOAD=1 DO_NOT_IMPORT_HABANA_TORCH=1 ${__pytorch_modules_tests_exe} pytest_working/test_autoload.py -v $__failures $__py_filter --junit-xml="${__xml}_lazy_pytest_autoload.xml" --mode="lazy" --dut="${__dut}" --junit-prefix="PytestLazy" ${__marker})
            __test_status=$((__test_status | $?))
        fi
        if [[ "$__pytest_mode" = "compile" || "$__pytest_mode" = "all" ]] ; then
            (set -x; eval ${__pytorch_modules_tests_exe} pytest_working/ -v $__failures $__py_filter --junit-xml="${__xml}_compile_pytest.xml" --mode="compile" --dut="${__dut}" --junit-prefix="PytestCompile" ${__marker})
            __test_status=$((__test_status | $?))
            (set -x; eval PT_HPU_AUTOLOAD=1 DO_NOT_IMPORT_HABANA_TORCH=1 ${__pytorch_modules_tests_exe} pytest_working/test_autoload.py -v $__failures $__py_filter --junit-xml="${__xml}_compile_pytest_autoload.xml" --mode="compile" --dut="${__dut}" --junit-prefix="PytestCompile" ${__marker})
            __test_status=$((__test_status | $?))
        fi
        if [[ "$__pytest_mode" = "eager" || "$__pytest_mode" = "all" ]] ; then
            (set -x; eval ${__pytorch_modules_tests_exe} pytest_working/ -v $__failures $__py_filter --junit-xml="${__xml}_eager_pytest.xml" --mode="eager" --dut="${__dut}" --junit-prefix="PytestEager" ${__marker})
            __test_status=$((__test_status | $?))
            (set -x; eval PT_HPU_AUTOLOAD=1 DO_NOT_IMPORT_HABANA_TORCH=1 ${__pytorch_modules_tests_exe} pytest_working/test_autoload.py -v $__failures $__py_filter --junit-xml="${__xml}_eager_pytest_autoload.xml" --mode="eager" --dut="${__dut}" --junit-prefix="PytestEager" ${__marker})
            __test_status=$((__test_status | $?))
        fi
        popd
    fi

    if [[ "$__suite_type" = "all" || "$__suite_type" = "py_tests" || "$__suite_type" = "infra" ]]; then
      pushd $PYTORCH_MODULES_ROOT_PATH/.devops/
      (set -x; eval ${__pytorch_modules_tests_exe} tests/ -v $__failures $__py_filter --junit-xml="${__xml}_infra_pytest.xml" --junit-prefix="Infra." ${__marker})
      __test_status=$((__test_status | $?))
      popd
      pushd $PYTORCH_MODULES_ROOT_PATH/scripts/
      (set -x; eval ${__pytorch_modules_tests_exe} tests/ -v $__failures $__py_filter --junit-xml="${__xml}_infra_scripts_pytest.xml" --junit-prefix="InfraScripts." ${__marker})
      __test_status=$((__test_status | $?))
      popd
      if [[ "$__pytest_mode" = "eager" ]] ; then
        pushd $PYTORCH_MODULES_ROOT_PATH/tests/user_custom_op/
        $__python_cmd setup.py install
        (set -x; eval PT_HPU_LAZY_MODE=0 ${__pytorch_modules_tests_exe} test_hpu_custom_op.py -v $__failures $__py_filter --junit-xml="${__xml}_infra_custom_op_pytest.xml" --junit-prefix="InfraCustomOp." ${__marker})
        __test_status=$((__test_status | $?))
        (set -x; eval ${__pytorch_modules_tests_exe} test_hpu_custom_op.py -v $__failures $__py_filter --junit-xml="${__xml}_infra_custom_op_pytest.xml" --junit-prefix="InfraCustomOp." ${__marker})
        __test_status=$((__test_status | $?))
        (set -x; eval ${__pytorch_modules_tests_exe} test_hpu_legacy_custom_op.py -v $__failures $__py_filter --junit-xml="${__xml}_infra_custom_op_pytest.xml" --junit-prefix="InfraCustomOp." ${__marker})
        __test_status=$((__test_status | $?))
        popd
      fi
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
    rn50_eager_1c|rn50_graph_1c)
        __pytorch_qa_test_path="${PYTORCH_TESTS_ROOT}/tests/gdn_tests/topologies_tests"
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
       elif [ "$__suite_type" == "rn50_eager_1c" ]; then
            install_requirements_event_plugin
            (set -x; python3 -m pytest -sv ${__pytorch_qa_test_path}/test_resnet.py -k resnet_lars_1epoch_1xcard_bf16_eager_mode_gaudi2 "--junit-xml=${__xml}_"rn50_eager_ci_functional.xml"")
            __test_status=$?
            return ${__test_status}
        elif  [ "$__suite_type" == "rn50_graph_1c" ]; then
            install_requirements_event_plugin
            (set -x; python3 -m pytest -sv ${__pytorch_qa_test_path}/test_resnet.py -k resnet_lars_1epoch_1xcard_bf16_graph_mode_gaudi2_100_steps "--junit-xml=${__xml}_"rn50_graph_ci_functional.xml"")
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
    # Otherwise, clean since the difference in package versions can
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
    sudo -H $__pip_cmd uninstall -y wrapt requests gast
    cmd=($__pip_cmd install -r ${PYTORCH_MODULES_ROOT_PATH}/.ci/requirements/requirements-pytorch.txt)
    if ! __running_in_venv; then
        cmd+=(--user)
    fi
    "${cmd[@]}"
}

install_requirements_event_plugin()
{
    cmd=($__pip_cmd install -r ${EVENT_TESTS_PLUGIN_ROOT}/.ci/requirements/requirements-prod.txt)
    if ! __running_in_venv; then
        cmd+=(--user)
    fi
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
        if [[ "$__dut" = "sim" ]]; then
            export ENABLE_EXEUTION_ON_GAUDI_SIM=1
        fi
        echo "Fabric tests:"
        ${__pytorch_lightning_qa_tests_exe} --collectonly test_fabric/
        __test_status=$?

        echo "Lightning HPU tests:"
        ${__pytorch_lightning_qa_tests_exe} --collectonly test_pytorch/
        __test_status=$?

        popd
        return $__test_status
    fi

    if [[ "$__suite_type" = "all" || "$__suite_type" = "py_tests" ]] ; then
        pushd $LIGHTNING_HABANA_FORK_ROOT/internal_ci/scripts

        echo "Executing Lightning Habana CI tests on HPU"
        (set -x; bash run_ci.sh -d ${__dut})
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
    sudo -H $__pip_cmd uninstall -y wrapt requests gast
    cmd=($__pip_cmd install -r ${PYTORCH_MODULES_ROOT_PATH}/.ci/requirements/requirements-test.txt)
    if ! __running_in_venv; then
        cmd+=(--user)
    fi
    "${cmd[@]}"
}

uninstall_requirements_pytest()
{
    $__pip_cmd uninstall -r ${PYTORCH_MODULES_ROOT_PATH}/.ci/requirements/requirements-test.txt -y
}

clean_pytorch_pkgs()
{
    echo "-> Removing PyTorch-related packages"
    $__pip_cmd uninstall -y hb-torch torch hmp gather2d-cpp HabanaEmbeddingBag-cpp habanaOptimizerSparseSgd-cpp preproc-cpp habanaOptimizerSparseAdagrad-cpp habana-torch-dataloader habana-torch habana-torch-plugin
    sudo -H $__pip_cmd uninstall -y hb-torch torch hmp gather2d-cpp HabanaEmbeddingBag-cpp habanaOptimizerSparseSgd-cpp preproc-cpp habanaOptimizerSparseAdagrad-cpp habana-torch-dataloader habana-torch habana-torch-plugin
}

# Returns an error code 1 if any nvidia-related packages are installed.
check_no_unwanted_packages_installed() {
    echo "-> Verifying if no unwanted packages are installed"
    local -a faulty_packages
    faulty_packages=$(! "$__pip_cmd" freeze | grep 'triton\|nvidia')
    local -r retcode=$?
    readonly faulty_packages
    if [ "$retcode" -ne 0 ]; then
        printf 'ERROR: the following unwanted packages are installed:\n'
        printf '%s\n' "${faulty_packages[@]}"
        printf 'Have you installed CUDA torch at some point?\n'
        return 1
    fi
}

# Verifies the reported torch.__version__ matches the specified profile.
# Args: $1 - profile name as in build_profiles.json, e.g. current, next
check_proper_pt_version_installed() {
    if [ $# -ne 1 ]; then
        __error "check_proper_pt_version_installed requires a profile name as an argument, e.g. current or next"
        return 2
    fi

    echo "-> Verifying if the proper PT version is installed ($1)"
    pushd "${PYTORCH_MODULES_ROOT_PATH}"/.devops || return 1

    $__python_cmd - <<EOF "$1"
import sys
from build_profiles.profiles import get_version_literal_and_source as get_profile
from build_profiles.version import Version

try:
    import torch
except:
    print("ERROR: Torch is not installed in the current env", file=sys.stderr)
    sys.exit(1)

installed_version=torch.__version__
expected_version=get_profile(sys.argv[1]).version

if Version(expected_version).major_minor_match(Version(installed_version)):
    sys.exit()

print(f"ERROR: Installed torch version {installed_version} does not resemble the expected {expected_version}",
      file=sys.stderr)
sys.exit(1)
EOF
    local -r retcode=$?

    popd || return 1

    return $retcode
}

# Installs packages required for PT Fork and Modules build pipelines to pass
install_pytorch_build_requirements() (
    set -e

    echo "-> Installing PT build requirements"
    $__pip_cmd install -r "${PYTORCH_MODULES_ROOT_PATH}"/requirements.txt

    check_no_unwanted_packages_installed
)

# Installs packages required for PT Fork, Modules, Lightning and other artifacts to work after deployment
install_pytorch_deploy_requirements() (
    set -e

    echo "-> Installing PT deployment requirements"
    $__pip_cmd install -r "${PYTORCH_MODULES_ROOT_PATH}"/.ci/requirements/requirements-pytorch.txt

    check_no_unwanted_packages_installed
)

# Installs packages required to run CI/CD test pipelines on PT artifacts
install_pytorch_test_requirements() (
    set -e

    echo "-> Installing PT test requirements"
    $__pip_cmd install -r "${PYTORCH_MODULES_ROOT_PATH}"/.ci/requirements/requirements-test.txt

    check_no_unwanted_packages_installed
)

# Args:
#   $1 - profile name, e.g. current
__print_torch_version_for_profile() {
    pushd "${PYTORCH_MODULES_ROOT_PATH}"/.devops >/dev/null || return 1

    $__python_cmd - <<EOF "$1"
import sys
from build_profiles.profiles import get_version_literal_and_source as get_profile
print(get_profile(sys.argv[1]).version)
EOF
    local -r retcode=$?

    popd >/dev/null || return 1

    return $retcode
}

# Args: $1 - profile name as in build_profiles.json, e.g. current, next
__print_matching_torch_wheel_path() (
    set -e

    if [ $# -ne 1 ]; then
        __error "__print_matching_torch_wheel_path requires a profile name as 1st argument, e.g. current or next"
        return 2
    fi

    local -r profile_name="$1"; shift

    # The wheels should reside in the fork build directory - even if they're from PT Next
    local wheels
    wheels=$(find "${PYTORCH_FORK_RELEASE_BUILD}"/pkgs -type f -name "*.whl")
    readonly wheels

    local -r wheel_count=$(echo "$wheels" | wc -l)

    if [ "$wheel_count" -lt 1 ]; then
        __error "did not find any torch wheels"
        return 1
    elif [ "$wheel_count" -eq 1 ]; then
        # No way to check commit hash as in CI/Promote we don't have pytorch-{fork,next} repos available.
        # Just verify the version against the profile.
        local expected_version
        expected_version=$(__print_torch_version_for_profile "$profile_name")
        readonly expected_version
        case $wheels in
            *$expected_version*)
                echo "$wheels"
                return
                ;;
            *)
                __error "the found wheel: $wheels does not match the expected version: $expected_version"
                return 1
                ;;
        esac
    fi

    # Multiple wheels found - match a single one

    if [ "$profile_name" = "current" ]; then
        local -r torch_root="${PYTORCH_FORK_ROOT}"
    else
        if [ "$profile_name" != "next" ]; then
            __error "only current and next profiles are supported"
            return 2
        fi
        local -r torch_root="${PYTORCH_NEXT_ROOT}"
    fi

    local torch_revision
    torch_revision=$(cd "${torch_root}" && git rev-parse HEAD)
    readonly torch_revision

    local matching_wheels
    matching_wheels=$(echo "$wheels" | grep "${torch_revision:0:7}")
    readonly matching_wheels

    local -r matching_wheel_count=$(echo "$matching_wheels" | wc -l)
    if [ "$matching_wheel_count" -ne 1 ]; then
        __error "did not find exactly one matching torch wheel. Found: $matching_wheels"
        return 1
    fi
    echo "${matching_wheels}"
)

# Args:
# * $1 - profile name as in build_profiles.json, e.g. current, next
# * $@ - additional arguments to pass to the bulk pip install command
__set_up_pytorch_artifacts_impl() (
    set -e

    if [ $# -lt 1 ] || [ "$1" != "current" ] && [ "$1" != "next" ]; then
        __error "__set_up_pytorch_artifacts_impl requires a profile name as 1st argument, e.g. current or next"
        return 2
    fi

    local -r profile_name="$1"; shift

    echo "-> Looking for a proper torch wheel to install"
    local pt_fork_wheel_path
    pt_fork_wheel_path=$(__print_matching_torch_wheel_path "$profile_name")
    readonly pt_fork_wheel_path
    echo "  -> Will use this torch wheel: $pt_fork_wheel_path"

    uninstall_pytorch_artifacts

    local -ar pip_install_args=(
        -r "${PYTORCH_MODULES_ROOT_PATH}"/.ci/requirements/requirements-pytorch.txt
        "$@"
        "$pt_fork_wheel_path"
        "${PYTORCH_VISION_BUILD}"/pkgs/*.whl
        "${PYTORCH_MODULES_RELEASE_BUILD}"/pkgs/*.whl
    )

    echo "-> Installing PT requirements and artifacts: " "${pip_install_args[@]}"

    # Installing in one go should prevent issues with CUDA torch begin pulled in by accident
    $__pip_cmd install -U "${pip_install_args[@]}"


    if [ "${GERRIT_PROJECT}" = "lightning-habana-fork" ]; then
        echo "-> Installing ligtning-habana-fork wheels"
        $__pip_cmd install -U "${LIGHTNING_HABANA_FORK_BUILD}"/pkgs/*.whl --force-reinstall --no-deps
    fi

    __install_habana_transformer_engine

    check_no_unwanted_packages_installed
)


__install_habana_transformer_engine() {
    echo "-> Looking for Habana Transformer Engine wheel to install"
    hte_whls=$(ls ${TRANSFORMER_ENGINE_FORK_BUILD}/pkgs/*.whl 2>/dev/null | wc -l || true)
    if [ ${hte_whls} -gt 0 ]; then
        echo "  -> Habana Transformer Engine wheel found"
        $__pip_cmd install -U "${TRANSFORMER_ENGINE_FORK_BUILD}"/pkgs/*.whl --force-reinstall --no-deps
        echo "  -> Habana Transformer Engine installed"
    else
        echo "  -> Habana Transformer Engine wheel not found"
    fi
}

# Args: $1 - profile name as in build_profiles.json, e.g. current, next
__set_up_pytorch_artifacts_for_testing_impl() {
    if [ $# -ne 1 ]; then
        __error "__set_up_pytorch_artifacts_for_testing_impl requires a profile name as an argument, e.g. current or next"
        return 2
    fi

    __set_up_pytorch_artifacts_impl "$1" -r "${PYTORCH_MODULES_ROOT_PATH}"/.ci/requirements/requirements-test.txt
}

__move_future_pytorch_version_artifacts_to_current_dirs() {
    echo "-> Preparing to install future PT version artifacts"
    (
        set -e

        rm -fv "$PYTORCH_FORK_RELEASE_BUILD"/pkgs/torch-*.whl
        rm -fv "$PYTORCH_MODULES_RELEASE_BUILD"/pkgs/*.whl
        rm -fv "$PYTORCH_VISION_FORK_BUILD"/pkgs/*.whl
        rm -fv "$PYTORCH_VISION_BUILD"/pkgs/*.whl

        if [ -d "/dependencies" ]; then
            local -r find_root="/dependencies"
        else
            local -r find_root="./dependencies"
        fi
        local -r pt_next_dir=$(find $find_root -name pt_next_deps)

        cp -fv "$pt_next_dir"/whl_pyfork/*torch*.whl "${PYTORCH_FORK_RELEASE_BUILD}"/pkgs/
        cp -fv "$pt_next_dir"/whl_pytorch_vision/*torch*.whl "${PYTORCH_VISION_BUILD}"/pkgs/
        cp -fv "$pt_next_dir"/whl_pyint/*.whl "${PYTORCH_MODULES_RELEASE_BUILD}"/pkgs/
        cp -fv "$pt_next_dir"/{test_pt_integration,test_pt2_integration} "${PYTORCH_MODULES_RELEASE_BUILD}"/
    )
    local -r retcode=$?
    if [ $retcode -ne 0 ]; then
        __error "Unable to set up PT Next artifacts"
    fi
    return $retcode
}

# Args: $1 - profile name as in build_profiles.json, e.g. current, next
set_up_pytorch_artifacts() (
    set -e

    if [ $# -ne 1 ]; then
        __error "set_up_pytorch_artifacts requires a profile name as an argument, e.g. current or next"
        return 2
    fi

    if [ "$1" != "current" ]; then
        __move_future_pytorch_version_artifacts_to_current_dirs
    fi

    __set_up_pytorch_artifacts_impl "$1"

    check_proper_pt_version_installed "$1"
)

# Args: $1 - profile name as in build_profiles.json, e.g. current, next
set_up_pytorch_artifacts_for_testing() (
    set -e

    if [ $# -ne 1 ]; then
        __error "set_up_pytorch_artifacts_for_testing requires a profile name as an argument, e.g. current or next"
        return 2
    fi

    if [ "$1" != "current" ]; then
        __move_future_pytorch_version_artifacts_to_current_dirs
    fi

    __set_up_pytorch_artifacts_for_testing_impl "$1"

    check_proper_pt_version_installed "$1"
)

uninstall_pytorch_artifacts() {
    echo "-> Uninstalling PT artifacts"
    $__pip_cmd uninstall -y torch torch-debug habana-torch-dataloader habana-torch-plugin torch_tb_profiler torchaudio torchdata torchtext torchvision
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

# Installs MKL include files and static libraries if needed and points CMake at them.
# This is a very similar approach to what upstream uses.
# By installing static libs we don't require MKL to be installed at runtime, either from pip or from the system package
# manager.
#
# Inspired by https://github.com/pytorch/builder/blob/main/common/install_mkl.sh
__provide_mkl()
{
  local -r __mkl_version=2024.2.0

  # choose the location depending on the user's sudo permissions
  local -r __mkl_root="$(sudo -v &>/dev/null && echo /opt/intel || echo ~/.local/opt/intel)"

  (
    set -e

    # install MKL if not installed yet
    if compgen -G "${__mkl_root}"/lib/*mkl* >/dev/null; then
      echo Will use MKL from "${__mkl_root}"
    else
      echo Installing MKL at "${__mkl_root}"

      sudo mkdir -p "${__mkl_root}"
      sudo chown -R "$(whoami)" "${__mkl_root}"

      mkdir /tmp/mkl
      pushd /tmp/mkl

      python3 -mpip install wheel
      python3 -mpip download -d . mkl-static==${__mkl_version} mkl-include==${__mkl_version}

      python3 -m wheel unpack mkl_static-${__mkl_version}-py2.py3-none-manylinux1_x86_64.whl
      mv mkl_static-${__mkl_version}/mkl_static-${__mkl_version}.data/data/lib "${__mkl_root}"

      python3 -m wheel unpack mkl_include-${__mkl_version}-py2.py3-none-manylinux1_x86_64.whl
      mv mkl_include-${__mkl_version}/mkl_include-${__mkl_version}.data/data/include "${__mkl_root}"

      popd
    fi
  )
  local __result=$?
  if [ $__result -ne 0 ]; then
    echo Error: unable to provide MKL
    return $__result
  fi

  export CMAKE_LIBRARY_PATH=${__mkl_root}/lib:$CMAKE_LIBRARY_PATH
  export CMAKE_INCLUDE_PATH=${__mkl_root}/include:$CMAKE_INCLUDE_PATH
}

# Method to install pillow-simd which is required for performance
# pillow package gets pulled along with installation of torchvision
# So it is required to uninstall pillow and install pillow-simd
# after torchvision installation
install_pillow_simd()
{
    $__pip_cmd uninstall -y pillow
    $__pip_cmd uninstall -y pillow-simd
    cmd=($__pip_cmd install -U --force-reinstall git+https://github.com/aostrowski-hbn/pillow-simd.git@simd/9.5.x)
    if ! __running_in_venv; then
        cmd+=(--user)
    fi
    CC="cc -mavx2" "${cmd[@]}"
}

# set_python_version to set envs related to python version during build
set_python_version()
{
    case $1 in
    "3.8" | "3.10" | "3.11" | "3.12")
        echo "version $1"
        ;;
    *)
        echo "Usage: $0 <3.8/3.10/3.11/3.12>"
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

get_github_repo()
{
  # clone github repo and checkout required branch/tag/sha-id
  if [[ "$#" -ne 2 ]]; then
    # Check if the correct number of parameters is provided
    echo -e "\n Usage: get_github_repo repository revision \n"
    echo -e "repository    -   Name of the repository for Ex:pytorch/data"
    echo -e "revision      -   provide branch/tag/sha-id for Ex: main;v0.16.2;53ca583fd5e5d53004b7a73654ba7ac5afcb715b"
    exit 1
  fi
  # assign parameters to variables
  local __repository="${1}"
  local __revision="${2}"
  # For branch fast path can be chosen, for revision need to fetch whole repo
  if [[ ${__revision} =~ ^[0-9a-f]{7,40}$ ]]; then
    # Enable --not-checkout to clone without downloading working-tree.
    git clone --no-checkout "https://github.com/${__repository}" .
    # Checkout required sha-id if found.
    git checkout ${__revision} || { echo "Error: Checkout Failed! Provided revision: $__revision is not valid"; exit 1 ;}
  else
    # Clone only last state of given branch
    git clone --depth 1 --branch ${__revision} "https://github.com/${__repository}" . || { echo "Error: Checkout Failed! Provided revision: $__revision is not valid"; exit 1 ;}
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

    # checkout github torchtext repo
    if [ -z ${__pt_text_version} ]; then
        __pt_text_version=$($__profile_getter_path --get-extras-version torchtext current)
    fi
    echo "get torchtext from github (tag: $__pt_text_version)"
    get_github_repo "pytorch/text" "${__pt_text_version}"
    git submodule update --init --recursive

    if [ -n "$__configure" ]; then
        $__python_cmd setup.py clean
    fi

    # Replace tcmalloc_minimal with tcmalloc to avoid Segmentation fault during torchtext importing.
    # For details look here: SW-201537
    sed -i 's/tcmalloc_minimal/tcmalloc/g' third_party/sentencepiece/src/CMakeLists.txt

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

    # checkout github torchdata repo
    if [ -z ${__pt_data_version} ]; then
        __pt_data_version=$($__profile_getter_path --get-extras-version torchdata current)
    fi
    echo "get torchdata from github (tag: $__pt_data_version)"
    get_github_repo "pytorch/data" "${__pt_data_version}"
    git submodule update --init --recursive

    if [ -n "$__configure" ]; then
        $__python_cmd setup.py clean
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
    get_github_repo "pytorch/audio" "${__pt_audio_version}"
    git submodule update --init --recursive

    if [ -n "$__configure" ]; then
        $__python_cmd setup.py clean
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

build_pytorch_vision()
{
    SECONDS=0
    local __scriptname=$(__get_func_name)
    local __env_vars=""
    local __configure=""
    local __whl_params=" bdist_wheel"
    local __result
    local __set_py_vers="false"
    local __profile_getter_path="${PYTORCH_MODULES_ROOT_PATH}/.devops/profile_getter.py"
    local __pt_vision_version
    local __next_version=false
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
        --pt-vision-version )
            __pt_vision_version="$2"
            shift
            ;;
        --next )
            __next_version=true
            ;;
        -h  | --help )
            usage $__scriptname
            restore_python_version
            return 0
            ;;
        esac
        shift
    done

    rm -rf $PYTORCH_VISION_ROOT
    mkdir -p $PYTORCH_VISION_ROOT
    pushd $PYTORCH_VISION_ROOT

    # checkout github torch vision repo
    if [ -z ${__pt_vision_version} ]; then
        if [ ${__next_version} = true ]; then
            __pt_vision_version=$($__profile_getter_path --get-extras-version torchvision next)
        else
            __pt_vision_version=$($__profile_getter_path --get-extras-version torchvision current)
        fi
    fi
    echo "get torch vision from github (tag: $__pt_vision_version)"
    get_github_repo "pytorch/vision" "${__pt_vision_version}"
    git submodule update --init --recursive

    if [ -n "$__configure" ]; then
        $__python_cmd setup.py clean
    fi

    echo "Build parameters ${__whl_params}"

    (set -x;eval ${__env_vars} $__python_cmd setup.py ${__whl_params})
    __result=$?
    if [ $__result -ne 0 ]; then
        echo "Pytorch torch vision build failed!"
    fi
    PTV_WHL_PATH="$PYTORCH_VISION_ROOT/dist/"

    popd
    if [[ "$__whl_params" = " bdist_wheel" ]]; then
        rm -rf $PYTORCH_VISION_BUILD/pkgs
        mkdir -p $PYTORCH_VISION_BUILD/pkgs
        cp -f ${PTV_WHL_PATH}/*.whl $PYTORCH_VISION_BUILD/pkgs
    fi

    printf "\nElapsed time: %02u:%02u:%02u \n\n" $(($SECONDS / 3600)) $((($SECONDS / 60) % 60)) $(($SECONDS % 60))
    restore_python_version
    return $__result
}

# Installs pt fork, pt vision, lightning if required & pt modules
# called from the ci/promote flow
install_pytorch_whls() {
    __clean_pytorch_dev_py_deps
    #temporary workaround for pytorch-fork migration to separate component SW-162985
    echo "Starting install_pytorch_whls..."
    whls_count=$(find ${PYTORCH_FORK_RELEASE_BUILD}/pkgs -type f -name "*.whl" | wc -l)
    if [ $whls_count -gt 1 ]; then
        pyfork_revision=$(cd ${PYTORCH_FORK_ROOT} && git rev-parse HEAD)
        echo "installing pyfork version ${pyfork_revision:0:7}"
        $__pip_cmd install -U ${PYTORCH_FORK_RELEASE_BUILD}/pkgs/*${pyfork_revision:0:7}*.whl
    elif [ $whls_count == 1 ]; then
        printf "Whl detected in ${PYTORCH_FORK_RELEASE_BUILD}/pkgs directory"
        $__pip_cmd install -U ${PYTORCH_FORK_RELEASE_BUILD}/pkgs/*.whl
    else
        echo "Didn't find pytorch_fork whl file"
        exit 1
    fi
    $__pip_cmd install -U ${PYTORCH_VISION_BUILD}/pkgs/*.whl
    install_pillow_simd
    if [ "${GERRIT_PROJECT}" = "lightning-habana-fork" ];  then
        $__pip_cmd install -U ${LIGHTNING_HABANA_FORK_BUILD}/pkgs/*.whl --force-reinstall --no-deps
    fi
    $__pip_cmd install -U ${PYTORCH_MODULES_RELEASE_BUILD}/pkgs/*.whl
    __install_habana_transformer_engine
}

install_pytorch_whls_future() {
    rm -fv $PYTORCH_FORK_RELEASE_BUILD/pkgs/torch-*.whl
    rm -fv $PYTORCH_MODULES_RELEASE_BUILD/pkgs/*.whl
    rm -fv $PYTORCH_VISION_FORK_BUILD/pkgs/*.whl
    rm -fv $PYTORCH_VISION_BUILD/pkgs/*.
    if [ -d "/dependencies" ]; then
        find_root="/dependencies"
    else
        find_root="./dependencies"
    fi
    pt_next_dir=$(find $find_root -name pt_next_deps)
    cp -fv $pt_next_dir/whl_pyfork/*torch*.whl ${PYTORCH_FORK_RELEASE_BUILD}/pkgs/
    cp -fv $pt_next_dir/whl_pytorch_vision/*torch*.whl ${PYTORCH_VISION_BUILD}/pkgs/
    cp -fv $pt_next_dir/whl_pyint/*.whl ${PYTORCH_MODULES_RELEASE_BUILD}/pkgs/
    cp -fv $pt_next_dir/{test_pt_integration,test_pt2_integration} ${PYTORCH_MODULES_RELEASE_BUILD}/
    install_pytorch_whls
}

dsa_debugger()
{
    if [ -z "$PYTORCH_MODULES_ROOT_PATH" ]
    then
        echo "PYTORCH_MODULES_ROOT_PATH path is not defined"
        return 1
    fi

    local __dsa_debugger_py="$__python_cmd $PYTORCH_MODULES_ROOT_PATH/python_packages/habana_frameworks/torch/utils/debug/dsa_debugger.py"
    ${__dsa_debugger_py} "$@"

    return $?
}
