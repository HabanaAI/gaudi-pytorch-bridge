#!/usr/bin/env bash
###############################################################################
#
#  Copyright (c) 2025 Intel Corporation
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

function help_run_pytorch_integration_tests() {
    echo -e "\nFunction run_pytorch_integration_tests usage: [options]\n"
    echo -e "options:\n"
    echo -e "  -l,  --list-tests                   List the available tests"
    echo -e "  -s,  --specific-test TEST           Run TEST"
    echo -e "  -d,  --debug                        Use debug test binary"
    echo -e "  -m,  --maxfail NUM                  Stop after NUM failures"
    echo -e "  -x,  --xml PATH                     Output XML file to PATH - available in ST mode only"
    echo -e "  -a,  --marker                       Only run tests matching given mark expression. Example: -a 'mark1 and not mark2'"
    echo -e "  -t,  --suite-type TYPE              Run specific suite type [all, py_tests, cpp_tests, cpp_lazy, cpp_eager]. Default: all"
    echo -e "  -c,  --test-case JUNITID            Run specific test based on JUnit ID"
    echo -e "       --pytest-mode                  Run specific pytest suite mode: [all, lazy, compile, eager]. Default: all"
    echo -e "  -hllog LOG_LEVEL                    0-TRACE, 1-DEBUG 2-INFO, 3-WARN, 4-ERR, 5-CRITICAL"
    echo -e "  -r, --rerun-failures                Rerun tests on failure"
    echo -e "  -h,  --help                         Prints this help"
}

run_pytorch_integration_tests()
{
    local __pytorch_modules_dir=$(cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )/../../
    local __pytorch_modules_tests_exe="python3 -m pytest"
    local __xml="test_detail"
    local __print_tests=""
    local __failures=""
    local __marker=""
    local __test_status=0
    local __dut="gaudi3"
    local __hllog=3
    local __pytest_mode="eager"
    local __suite_type="py_tests"
    local __py_rerun_fail="--reruns 1"

    # parameter while-loop
    while [ -n "$1" ];
    do
        case $1 in
        -l  | --list-tests )
            __print_tests="yes"
            ;;
        -m  | --maxfail )
            shift
            __failures="--maxfail=$1"
            ;;
        -t | --suite-type )
            shift
            __suite_type="$1"
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
        -pm | --pytest-mode )
            shift
            __pytest_mode="$1"
            if [[ "${__pytest_mode}" != "compile" && "${__pytest_mode}" != "eager" ]]; then
                echo "Pytest mode \"$__pytest_mode\" is not allowed"
                help_run_pytorch_integration_tests
                return 1 # error
            fi
            ;;
        -h  | --help )
            help_run_pytorch_integration_tests
            return 0
            ;;
        *)
            echo "The parameter $1 is not allowed"
            help_run_pytorch_integration_tests
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
    *)
        echo "Test suite type \"$__suite_type\" is not allowed"
        help_run_pytorch_integration_tests
        return 1 # error
        ;;
    esac

    python3 -m pip install -r ${__pytorch_modules_dir}/.ci/requirements/requirements-test.txt

    if [ -n "$__print_tests" ]; then
        pushd ${__pytorch_modules_dir}/tests/
        echo "python tests:"
        ${__pytorch_modules_tests_exe} --collect-only --vendor
        __test_status=$((__test_status | $?))
        popd
        return $__test_status
    fi

    if [[ "$__suite_type" = "all" || "$__suite_type" = "py_tests" ]] ; then
        pushd ${__pytorch_modules_dir}/tests/
        if [[ "$__pytest_mode" = "compile" || "$__pytest_mode" = "all" ]] ; then
            (set -x; eval ${__pytorch_modules_tests_exe} pytest_working/ -v $__failures $__py_rerun_fail --junit-xml="${__xml}_compile_pytest.xml" --vendor --mode="compile" --dut="${__dut}" --junit-prefix="PytestCompile" ${__marker})
            __test_status=$((__test_status | $?))
        fi
        if [[ "$__pytest_mode" = "eager" || "$__pytest_mode" = "all" ]] ; then
            (set -x; eval ${__pytorch_modules_tests_exe} pytest_working/ -v $__failures $__py_rerun_fail --junit-xml="${__xml}_eager_pytest.xml" --vendor --mode="eager" --dut="${__dut}" --junit-prefix="PytestEager" ${__marker})
            __test_status=$((__test_status | $?))
            (set -x; eval PT_HPU_AUTOLOAD=1 DO_NOT_IMPORT_HABANA_TORCH=1 ${__pytorch_modules_tests_exe} pytest_working/test_autoload.py -v $__failures $__py_rerun_fail --junit-xml="${__xml}_eager_pytest_autoload.xml" --mode="eager" --dut="${__dut}" --junit-prefix="PytestEager" ${__marker})
            __test_status=$((__test_status | $?))
        fi
        popd
    fi

    # return error code of the tests
    return ${__test_status}
}
