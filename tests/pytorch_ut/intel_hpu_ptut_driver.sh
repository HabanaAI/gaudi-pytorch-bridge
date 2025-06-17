#!/bin/bash
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

echo "Detecting pytorch version..."
pt_ver=$(python3 -c 'import torch as pt; print(pt.__version__.split("+")[0])')
echo "Detected pytorch version is $pt_ver"
pt_ver="v${pt_ver}"
echo "Detected pytorch version is $pt_ver"
# Detect number of ASICs in the VM
asic_count=$(lspci | grep -c acc)
simulator_count=$(pgrep -c coral)
if [ $asic_count != 0 ]; then
    echo "Total number of ASICs detected are : $asic_count"
    export DOCKER_HOME=/root
    ls -l $DOCKER_HOME/
    ls -l /usr/lib/habanalabs/
    ls -l $DOCKER_HOME/.local/lib/
    ls -l /usr/local/lib/
    # Set permissions
    chmod -R 777 $HABANA_SOFTWARE_STACK/pytorch-training-tests/tests/pytorch/vendorci
    pushd $HABANA_SOFTWARE_STACK/pytorch-training-tests/tests/pytorch/vendorci
    elif [ $simulator_count != 0 ]; then
    echo "Total number of ASIC Simulators detected are : $simulator_count"
    export DOCKER_HOME=/
    export HABANA_SOFTWARE_STACK=$HABANA_NPU_STACK_PATH
fi
# Needed packages for Habana QA PyTorch tests
OS_Version=$(lsb_release -d)
# Test
LOCK_GAUDI_SYNAPSE_API=1
__out_dir=""
__test_type=""
__test_file=""
__test_filter=""
__lazy_mode=""
__devop_xmldir=""
usage()
{
    echo "  -h  | --help        Prints this help"
    echo "  -o  | --out_dir     Output folder to place test results"
    echo "  -t  | --test_type   Tests type (all/core/ops/slow) to execute tests"
    echo "  -f  | --test_file   Files to be tested"
    echo "  -k  | --test_filter Test filter to be passed to -k option of pytest"
    echo "  -cl | --clone       Clone the tests from Pytorch's upstream repository and run the tests"
    echo "  -d, | --device-id   Asic ID. Default value is gaudi"
    echo "  -m  | --lazy_mode   PT_HPU_LAZY_MODE. Execution mode.  1 Lazy mode and 0 Eager mode."
    echo "  -g  | --gen_analysis Generatate analysis_report"
    echo ""
    return 0
}
set_env()
{
    # Set environment variables
    export LOG_LEVEL_ALL=4
    export PT_HPU_LAZY_MODE=0
    export PT_HPU_PLACE_ON_CPU=none
    export PT_ENABLE_INT64_SUPPORT=1
    export PYTORCH_TEST_WITH_SLOW=0 # Do not run slow tests, set it to 1 to run slow tests
    export PYTORCH_TEST_WITH_SLOW_GRADCHECK=0
    export PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES=0
    export PATH=$PATH:$DOCKER_HOME/.local/lib/:/usr/local/lib/:$DOCKER_HOME/.local/bin
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$DOCKER_HOME/.local/lib/:/usr/lib/habanalabs/:$DOCKER_HOME/:/usr/local/lib/:$DOCKER_HOME/.local/bin
    export SOFTWARE_DATA=$DOCKER_HOME/software/data
    export SOFTWARE_UTILS=$DOCKER_HOME/software/utils
    export PYTORCH_TESTS_ROOT=$HABANA_SOFTWARE_STACK/pytorch-training-tests
    export PYTORCH_MODULES_ROOT_PATH=$HABANA_SOFTWARE_STACK/pytorch-integration
    export PYTHONPATH=$PYTHONPATH:$DOCKER_HOME/pytorch-training-tests:$HABANA_PYTORCH_FWUT_ROOT:$PYTORCH_MODULES_ROOT_PATH/topologies:/usr/lib/habanalabs:$DOCKER_HOME/$DISTRIBUTED_HPU_PIPELINE_PATH:$DOCKER_HOME/repos/$DISTRIBUTED_HPU_PIPELINE_PATH
    export PYTORCH_MODULES_RELEASE_BUILD=/usr/lib/habanalabs/
    export MODEL_GARDEN_ROOT=$HABANA_SOFTWARE_STACK/model_garden
    export MODEL_GARDEN_PYTORCH_PATH=$HABANA_SOFTWARE_STACK/model_garden/PyTorch
    export PYTORCH_TEST_DIR=$HABANA_PYTORCH_FWUT_FORK_ROOT/test/
    export DEVICE_UNDER_TEST=$deivce_id
    #export HABANA_LOGS="/root/logs/habana_logs"
    [[ -z "${QNPU_PATH}" ]] && export HABANA_LOGS="/root/logs/habana_logs" || export HABANA_LOGS=$PWD/logs/habana_logs
    mkdir -p ${HABANA_LOGS}
    rm -rf /var/log/habana_logs
    ln -s $HABANA_LOGS /var/log/habana_logs
    pip install --upgrade pip
    pip install -r $PWD/requirements.txt --progress-bar off --user --no-warn-script-location --quiet
    echo "Installation of dependencies is completed"
    python -V
    pip -V
    python -um pip list
}
run_gen_analysis()
{
    out_dir=$1
    echo "Running gen_analysis_report"
    Test_Cmd="python -u gen_analysis_report.py --pytest_report_path $out_dir/ --read_console_log --gen"
    echo $Test_Cmd
    (set -x; eval $Test_Cmd)
    test_return_code=$?
    Test_Cmd="python -u gen_analysis_report.py --pytest_report_path $out_dir/ --read_console_log --sum_fail"
    echo $Test_Cmd
    (set -x; eval $Test_Cmd)
    test_return_code=$?
    Test_Cmd="python -u gen_analysis_report.py --pytest_report_path $out_dir/ --read_console_log --sum_fallback"
    echo $Test_Cmd
    (set -x; eval $Test_Cmd)
    test_return_code=$?
    #
    if [  -f analysis_report_with_console_error.csv ]; then
        cp analysis_report_with_console_error.csv $out_dir/
    fi
    if [  -f analysis_report_with_console_error_failure_analysis.csv ]; then
        cp analysis_report_with_console_error_failure_analysis.csv $out_dir/
    fi
    if [  -f analysis_report_with_console_error_fallback_analysis.csv ]; then
        cp analysis_report_with_console_error_fallback_analysis.csv $out_dir/
    fi
}
run_core_tests()
{
    echo "Running only core tests"
    Test_Cmd="python3 run_test_hpu.py -v -k 'hpu' -dir ${PYTORCH_TEST_DIR} --outdir=$out_dir -fp 8 -core"
    echo $Test_Cmd
    (set -x; eval $Test_Cmd)
    return $?
}
run_all_tests()
{
    echo "Running all test cases"
   Test_Cmd="python3 run_test_hpu.py -v -k 'hpu' -dir ${PYTORCH_TEST_DIR} --outdir=$out_dir -fp 8"
   echo $Test_Cmd
   (set -x; eval $Test_Cmd)
   return $?
}
test_core_ops1_tests()
{
    echo "Running core ops list 1 test cases"
    Test_Cmd="python3 run_test_hpu.py -v -k 'hpu' -dir ${PYTORCH_TEST_DIR} --outdir=$out_dir -fp 8 --ops1"
    echo $Test_Cmd
    (set -x; eval $Test_Cmd)
    return $?
}
test_core_ops2_tests()
{
    echo "Running core ops list 2 test cases"
    Test_Cmd="python3 run_test_hpu.py -v -k 'hpu' -dir ${PYTORCH_TEST_DIR} --outdir=$out_dir -fp 8 --ops2"
    echo $Test_Cmd
    (set -x; eval $Test_Cmd)
    return $?
}
ci_tests()
{
    echo "Running ci test cases"
    Test_Cmd="python3 run_test_hpu.py -v -k 'hpu' -dir $pwd/pytorch/test --outdir=$out_dir  --ci_tests"
    echo $Test_Cmd
    (set -x; eval $Test_Cmd)
    return $?
}
test_slow_tests()
{
    echo "Running slow test cases"
    Test_Cmd="python3 run_test_hpu.py -v -k 'hpu' -dir ${PYTORCH_TEST_DIR} --outdir=$out_dir -fp 8 --slow"
    echo $Test_Cmd
    (set -x; eval $Test_Cmd)
    return $?
}
#
test_distributed_hpu_pipeline_tests()
{
    apt update
    apt install etcd etcd-server -y
    echo "Running distributed_hpu repo test cases"
    Test_Cmd="python3 run_test_hpu.py -v  -dir ${PYTORCH_TEST_DIR} --outdir=$out_dir -fp 8 --distributed_hpu_pipeline"
    echo $Test_Cmd
    (set -x; eval $Test_Cmd)
    return $?
}
#
test_distributed_hpu_tests()
{
    apt update
    apt install etcd etcd-server -y
    echo "Running distributed_hpu repo test cases"
    Test_Cmd="python3 run_test_hpu.py -v  -dir ${PYTORCH_TEST_DIR} --outdir=$out_dir -fp 8 --distributed_hpu"
    echo $Test_Cmd
    (set -x; eval $Test_Cmd)
    return $?
}
test_distributed_hpu_fsdp_tests()
{
    echo "Running distributed_hpu repo test cases"
    Test_Cmd="python3 run_test_hpu.py -v  -dir ${PYTORCH_TEST_DIR} --outdir=$out_dir -fp 8 --distributed_hpu_fsdp"
    echo $Test_Cmd
    (set -x; eval $Test_Cmd)
    return $?
}
#
test_distributed_hpu_fsdp2_tests()
{
    echo "Running distributed_hpu repo test cases"
    Test_Cmd="python3 run_test_hpu.py -v  -dir ${PYTORCH_TEST_DIR} --outdir=$out_dir -fp 8 --distributed_hpu_fsdp2"
    echo $Test_Cmd
    (set -x; eval $Test_Cmd)
    return $?
}
test_dynamo_hpu_tests()
{
    echo "Running dynamo_hpu repo test cases"
    Test_Cmd="python3 run_test_hpu.py -v -k 'hpu' -dir ${PYTORCH_TEST_DIR} --outdir=$out_dir -fp 8 --dynamo_hpu"
    echo $Test_Cmd
    (set -x; eval $Test_Cmd)
    return $?
}
test_gpu_migration_hpu_tests()
{
    echo "Running gpu_migration_hpu repo test cases"
    Test_Cmd="python3 run_test_hpu.py -v -dir ${PYTORCH_TEST_DIR} --outdir=$out_dir -fp 8 --gpu_migration_hpu"
    echo $Test_Cmd
    (set -x; eval $Test_Cmd)
    return $?
}
run_test_fallback_disabled()
{
    echo "Running only test_fallaback_disabled test cases"
    cp $HABANA_PYTORCH_FWUT_ROOT/conftest.py $HABANA_PYTORCH_FWUT_FORK_ROOT/test/
    cp $HABANA_PYTORCH_FWUT_ROOT/passing_tests.csv $HABANA_PYTORCH_FWUT_FORK_ROOT/test/
    Test_Cmd="python3  ${HABANA_PYTORCH_FWUT_ROOT}/run_test_hpu.py -v -k '_hpu' -m passing -dir ${PYTORCH_TEST_DIR} --outdir=$out_dir -fp 8"
    echo $Test_Cmd
    (set -x; eval $Test_Cmd)
    return $?
}
test_run()
{
    echo "Running only test with file/filter $test_file  $test_filter"
    if [ -z "$test_file" ] && [ -z "$test_filter" ] ; then
        Test_Cmd="python3  run_test_hpu.py -v -m 'not skip' -dir ${PYTORCH_TEST_DIR} --outdir=$out_dir -n 8"
        elif [ -z "$test_file" ] && [ ! -z "$test_filter" ]; then
        Test_Cmd="python3  run_test_hpu.py -v -k \"$test_filter\" -m 'not skip' -dir ${PYTORCH_TEST_DIR} --outdir=$out_dir -n 8"
        elif [ ! -z "$test_file" ] && [ -z "$test_filter" ]; then
        Test_Cmd="python3  run_test_hpu.py -v -i \"$test_file\" -m 'not skip' -dir ${PYTORCH_TEST_DIR} --outdir=$out_dir -n 8"
        elif [ ! -z "$test_file" ] && [ ! -z "$test_filter" ]; then
        Test_Cmd="python3  run_test_hpu.py -v -i \"$test_file\" -k \"$test_filter\" -m 'not skip' -dir ${PYTORCH_TEST_DIR} --outdir=$out_dir -n 8"
    fi
    echo $Test_Cmd
    (set -x; eval $Test_Cmd)
    return $?
}
run_test()
{
    out_dir=$1
    test_type=$2
    test_file="$3"
    test_filter="$4"
    if [[ -z $out_dir ]] && [[ -z $test_type ]]; then
        echo "Its mandatory to pass -o & -t option please see help option"
        usage
        exit 1 #error
    else
        case "$test_type" in
            "core")
                run_core_tests
            ;;
            "all")
                run_all_tests
            ;;
            "test_fallback_disabled")
                run_test_fallback_disabled
            ;;
            "test")
                test_run
            ;;
            "ops1")
                test_core_ops1_tests
            ;;
            "ops2")
                test_core_ops2_tests
            ;;
            "ci")
                ci_tests
            ;;
            "slow")
                test_slow_tests
            ;;
            "distributed_hpu")
                test_distributed_hpu_tests
            ;;
            "distributed_hpu_fsdp")
                test_distributed_hpu_fsdp_tests
            ;;
            "distributed_hpu_fsdp2")
                test_distributed_hpu_fsdp2_tests
            ;;
            "distributed_hpu_pipeline")
                test_distributed_hpu_pipeline_tests
            ;;
            "dynamo_hpu")
                test_dynamo_hpu_tests
            ;;
            "gpu_migration_hpu")
                test_gpu_migration_hpu_tests
            ;;
            *)
                echo "Incorrect test suite"
            ;;
        esac
    fi
}
deivce_id="gaudi"
while [ -n "$1" ]; do
    case "$1" in
        -o|--out_dir)
            shift
            __out_dir=$1
        ;;
        --junit_xml)
            shift
            __devop_xmldir=$1
        ;;
        -t|--test_type)
            shift
            __test_type=$1
        ;;
        -f|--test_file)
            shift
            __test_file="$1"
        ;;
        -k|--test_filter)
            shift
            __test_filter="$1"
        ;;
        -d|--deivce_id )
            shift
            deivce_id="$1"
        ;;
        -cl|--run-clone )
            run_clone="true"
        ;;
        -m|--lazy_mode )
            shift
            __lazy_mode="$1"
        ;;
        -g|--gen_analysis )
            gen_analysis="true"
        ;;
        -h|--help)
            usage
            exit 0
        ;;
        *)
            echo "The parameter $1 is not allowed"
            usage
            exit 1 #error
        ;;
    esac
    shift
done
if [ "$run_clone" == "true" ]; then
    rm -rf pytorch
    echo "Cloning pytorch from github ..."
    git clone --branch ${pt_ver} https://github.com/pytorch/pytorch.git
    retVal=$?
    if [ $retVal == 0 ]; then
        cd pytorch
        git checkout ${pt_ver}
        cd ..
        echo "Enabling the tests decorated with @onlyNativeDeviceTypes for HPU"
        sed -i 's/@onlyNativeDeviceTypes/#@onlyNativeDeviceTypes/g' pytorch/test/test_*.py
        cp -r pytorch/pytest.ini ../
        rm -rf pytorch/pytest.ini
    fi
fi
echo $PWD
set_env $__lazy_mode
echo $PWD
cp -r run_test_hpu.py $PWD/pytorch/test
cp -r test_infra_hpu.py $PWD/pytorch/test


cd $PWD/pytorch//test
run_test $__out_dir $__test_type "$__test_file" "$__test_filter"
cd ../../
cat ${pt_ver}/*.regressions > $__out_dir/console_logs/all_regressions.log 2>/dev/null
# xml to csv
# non forked csv
Test_Cmd="python -u report_generator.py -x $__out_dir/ -o $__out_dir/ --con $__out_dir/console_logs/no_fork/ -vv"
echo $Test_Cmd
(set -x; eval $Test_Cmd)
test_return_code=$?
Test_Cmd="python parse_pytorch_framework_test_xml.py $out_dir/"
echo $Test_Cmd
(set -x; eval $Test_Cmd)
test_return_code=$?
cp pytorch_framework_test_report.csv $out_dir/
if [ "$gen_analysis" == "true" ]; then
    run_gen_analysis $__out_dir
fi
cp -r $out_dir/* ${HABANA_LOGS}/
if [ "z${__devop_xmldir}" != "z" ]; then
    echo "copying ${HABANA_LOGS} to ${__devop_xmldir}"
    cp -r ${HABANA_LOGS}/* ${__devop_xmldir}/ 2>&1 > /dev/null
fi
echo "PyTorch Tests Completed"
exit $test_return_code
