###############################################################################
#
#  Copyright (c) 2021-2024 Intel Corporation
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

import os

import habana_frameworks.torch.hpu as hpu
import pytest
import torch

TRAIN_CMD = f"python {os.path.abspath(__file__)}"
DEBUG_MODE = os.getenv("ENABLE_DIVERGENCE_ANALYSER_TESTS_DEBUG")

if DEBUG_MODE == "1":
    train_cmd = [TRAIN_CMD, ""]
    do_train = [0, 1]
    do_compare = [0, 1]
    to_csv = [0, 1]
    do_split = [0, 1]
    enable_parallel = [0, 1]
    outdir = ["/tmp/", ""]
    exit_on_first_mismatch = [0, 1]
else:
    train_cmd = [TRAIN_CMD]
    do_train = [1]
    do_compare = [1]
    to_csv = [1]
    do_split = [1]
    enable_parallel = [0, 1]
    outdir = ["/tmp/"]
    exit_on_first_mismatch = [0, 1]


def _test_hpu_ops():
    def run(shape):
        a = torch.randn(shape).to("hpu")
        b = torch.randn(shape).to("hpu")
        out = torch.add(a, b)
        out = torch.mul(out, a)
        out = torch.relu(out)
        return out.cpu()

    n_channels = torch.randint(4, 64, (10,))
    for c in n_channels:
        out = run(shape=(4, c, 3))


def predict_status(train_cmd, do_train, do_compare, to_csv, do_split, enable_parallel, outdir, exit_on_first_mismatch):
    if outdir == "":
        return 512  # error: argument --outdir: expected one argument
    elif do_train == 1 and train_cmd == "":
        return 256  # AssertionError
    elif do_train == 1 and enable_parallel == 1 and do_split == 0:
        return 256  # AssertionError
    elif do_compare == 0 and to_csv == 1:
        return 256  # AssertionError
    else:
        return 0


def get_cmd(train_cmd, do_train, do_compare, to_csv, do_split, enable_parallel, outdir, exit_on_first_mismatch):
    cmd = f"python $PYTORCH_MODULES_ROOT_PATH/python_packages/habana_frameworks/torch/utils/debug/divergence_analyzer.py \
        --train_cmd '{train_cmd}' \
        --do_train {do_train} \
        --do_compare {do_compare} \
        --do_split {do_split} \
        --to_csv {to_csv} \
        --enable_parallel {enable_parallel} \
        --outdir {outdir} \
        --exit_on_first_mismatch {exit_on_first_mismatch}"
    return cmd


@pytest.mark.parametrize("train_cmd", train_cmd)
@pytest.mark.parametrize("do_train", do_train)
@pytest.mark.parametrize("do_compare", do_compare)
@pytest.mark.parametrize("to_csv", to_csv)
@pytest.mark.parametrize("do_split", do_split)
@pytest.mark.parametrize("enable_parallel", enable_parallel)
@pytest.mark.parametrize("outdir", outdir)
@pytest.mark.parametrize("exit_on_first_mismatch", exit_on_first_mismatch)
def test_divergence_tool(
    train_cmd, do_train, do_compare, to_csv, do_split, enable_parallel, outdir, exit_on_first_mismatch
):
    n_devices = hpu.device_count()
    if enable_parallel == 1 and n_devices < 2:
        return

    expected_status = predict_status(
        train_cmd, do_train, do_compare, to_csv, do_split, enable_parallel, outdir, exit_on_first_mismatch
    )

    if do_train == 0 and do_compare == 1:
        cmd_train = get_cmd(
            train_cmd=TRAIN_CMD,
            do_train=1,
            do_compare=0,
            to_csv=0,
            do_split=do_split,
            enable_parallel=do_split,
            outdir=outdir,
            exit_on_first_mismatch=exit_on_first_mismatch,
        )
        os.system(cmd_train)

    cmd = get_cmd(train_cmd, do_train, do_compare, to_csv, do_split, enable_parallel, outdir, exit_on_first_mismatch)
    test_status = os.system(cmd)

    assert expected_status == test_status
    assert os.path.exists(os.path.join(outdir, "dumps", "mismatch.txt")) == False


if __name__ == "__main__":
    torch.manual_seed(10)
    _test_hpu_ops()
