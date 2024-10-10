###############################################################################
# Copyright (C) 2023 Habana Labs, Ltd. an Intel Company
# All Rights Reserved.
#
# Unauthorized copying of this file or any element(s) within it, via any medium
# is strictly prohibited.
# This file contains Habana Labs, Ltd. proprietary and confidential information
# and is subject to the confidentiality and license agreements under which it
# was provided.
#
###############################################################################

import numpy as np
import pytest
import torch
from numpy import testing
from test_utils import cpu, hpu

# N - batch
# H - input height
# W - input width
# C - input channels
# min - min value of uniform random gen output, corresponds to mean for normal distribution
# max - max value of uniform random gen output, corresponds to stddev for normal distribution

test_case_list = [
    # N, H, W, C, min, max, seed
    (2, 2, 2, 2, 0, 5, 1245),
    (4, 3, 2, 1, -1, 2, 3235),
]


@pytest.mark.parametrize("N, H, W, C, min, max, seed", test_case_list)
def test_hpu_rand_gen_uniform_fwd(N, H, W, C, min, max, seed):
    # CPU and HPU uses different algorithm for RNG. Hence they are not compared
    # Instead basic sanity like range and staleness are checked
    a = float(min)
    b = float(max)
    g = torch.Generator()

    input1 = torch.empty(N, C, H, W, dtype=torch.float)
    in1_hpu = input1.to(hpu)
    g.manual_seed(seed)
    output1 = in1_hpu.uniform_(a, b, generator=g)
    output1_hpu = output1.to(cpu).detach().numpy()

    input2 = torch.empty(N, C, H, W, dtype=torch.float)
    in2_hpu = input2.to(hpu)
    g.manual_seed(seed)
    output2 = in2_hpu.uniform_(a, b, generator=g)
    output2_hpu = output2.to(cpu).detach().numpy()

    # verify if the two tensors are same for same seed
    testing.assert_equal(output1_hpu, output2_hpu)

    # verify if the output values exceeds the range
    testing.assert_equal((np.min(output2_hpu) >= a) and (np.max(output2_hpu) <= b), True)

    # verify if two tensors are different for different seeds
    input3 = torch.empty(N, C, H, W, dtype=torch.float)
    in3_hpu = input3.to(hpu)
    output3 = in3_hpu.uniform_(a, b)
    output3_hpu = output3.to(cpu).detach().numpy()
    testing.assert_equal(np.any(np.not_equal(output2_hpu, output3_hpu)), True)


@pytest.mark.parametrize("N, H, W, C, mean, std, seed", test_case_list)
def test_hpu_rand_gen_normal_fwd(N, H, W, C, mean, std, seed):
    # CPU and HPU uses different algorithm for RNG. Hence they are not compared
    # Instead basic sanity like range and staleness are checked
    g = torch.Generator()

    input1 = torch.empty(N, C, H, W, dtype=torch.float)
    in1_hpu = input1.to(hpu)
    g.manual_seed(seed)
    output1 = in1_hpu.normal_(mean=mean, std=std, generator=g)
    output1_hpu = output1.to(cpu).detach().numpy()

    input2 = torch.empty(N, C, H, W, dtype=torch.float)
    in2_hpu = input2.to(hpu)
    g.manual_seed(seed)
    output2 = in2_hpu.normal_(mean=mean, std=std, generator=g)
    output2_hpu = output2.to(cpu).detach().numpy()

    # verify if the two tensors are same for same seed
    testing.assert_equal(output1_hpu, output2_hpu)

    # verify if the output values exceeds the range (used 5 sigma range)
    testing.assert_equal(
        (np.min(output2_hpu) >= (mean - 5 * std)) and (np.max(output2_hpu) <= (mean + 5 * std)),
        True,
    )

    # verify if two tensors are different for different seeds
    input3 = torch.empty(N, C, H, W, dtype=torch.float)
    in3_hpu = input3.to(hpu)
    output3 = in3_hpu.normal_(mean=mean, std=std)
    output3_hpu = output3.to(cpu).detach().numpy()

    testing.assert_equal(np.any(np.not_equal(output2_hpu, output3_hpu)), True)

    torch.manual_seed(seed)
    # verify if two tensors are same across runs for the same manual seed
    output4 = in3_hpu.normal_(mean=mean, std=std)
    output4_hpu = output4.to(cpu).detach().numpy()

    torch.manual_seed(seed)
    output5 = in3_hpu.normal_(mean=mean, std=std)
    output5_hpu = output5.to(cpu).detach().numpy()

    testing.assert_equal(output4_hpu, output5_hpu)


@pytest.mark.parametrize("N, H, W, C, mean, std, seed", test_case_list)
def test_hpu_rand_gen_log_normal_fwd(N, H, W, C, mean, std, seed):
    # CPU and HPU uses different algorithm for RNG (log_normal op is based on random_normal kernel).
    # Hence they are not compared. Instead basic sanity like range and staleness are checked
    g = torch.Generator()

    input1 = torch.empty(N, C, H, W, dtype=torch.float)
    in1_hpu = input1.to(hpu)
    g.manual_seed(seed)
    output1 = torch.log(in1_hpu.log_normal_(mean=mean, std=std, generator=g))
    output1_hpu = output1.to(cpu).detach().numpy()

    input2 = torch.empty(N, C, H, W, dtype=torch.float)
    in2_hpu = input2.to(hpu)
    g.manual_seed(seed)
    output2 = torch.log(in2_hpu.log_normal_(mean=mean, std=std, generator=g))
    output2_hpu = output2.to(cpu).detach().numpy()

    # verify if the two tensors are same for same seed
    testing.assert_equal(output1_hpu, output2_hpu)

    # verify if the output values exceeds the range (used 5 sigma range)
    testing.assert_equal(
        (np.min(output2_hpu) >= (mean - 5 * std)) and (np.max(output2_hpu) <= (mean + 5 * std)),
        True,
    )

    # verify if two tensors are different for different seeds
    input3 = torch.empty(N, C, H, W, dtype=torch.float)
    in3_hpu = input3.to(hpu)
    output3 = torch.log(in3_hpu.log_normal_(mean=mean, std=std))
    output3_hpu = output3.to(cpu).detach().numpy()

    testing.assert_equal(np.any(np.not_equal(output2_hpu, output3_hpu)), True)

    torch.manual_seed(seed)
    # verify if two tensors are same across runs for the same manual seed
    output4 = torch.log(in3_hpu.log_normal_(mean=mean, std=std))
    output4_hpu = output4.to(cpu).detach().numpy()

    torch.manual_seed(seed)
    output5 = torch.log(in3_hpu.log_normal_(mean=mean, std=std))
    output5_hpu = output5.to(cpu).detach().numpy()

    testing.assert_equal(output4_hpu, output5_hpu)


@pytest.mark.skip(reason="synNodeCreateWithId failed for node: broadcast with synStatus 1 [Invalid argument].")
@pytest.mark.parametrize("N, H, W, C, min, max, seed", test_case_list)
def test_hpu_rand_gen_bernoulli_fwd_scalar(N, H, W, C, min, max, seed):
    # CPU and HPU uses different algorithm for RNG. Hence they are not compared
    # Instead basic sanity like range and staleness are checked

    input = torch.empty(N, C, H, W, dtype=torch.float)
    in_hpu = input.uniform_(0, 1).to(hpu)
    torch.manual_seed(seed)
    output1 = torch.bernoulli(in_hpu)
    output1_hpu = output1.to(cpu).detach().numpy()

    torch.manual_seed(seed)
    output2 = torch.bernoulli(in_hpu)
    output2_hpu = output2.to(cpu).detach().numpy()

    # verify if the two tensors are same for same seed
    testing.assert_equal(output1_hpu, output2_hpu)

    # verify if the output values exceeds the range
    testing.assert_equal((np.min(output2_hpu) >= 0) and (np.max(output2_hpu) <= 1), True)

    # Test bernoulli._float
    torch.manual_seed(seed)
    p = 0.5
    input_i32 = torch.empty((N, C, H, W), dtype=torch.int32)
    input_i32_hpu = input_i32.to(hpu)
    output1 = input_i32_hpu.bernoulli(p, generator=g)
    output1_hpu = output1.to(cpu).detach().numpy()

    torch.manual_seed(seed)
    input_f32 = torch.empty((N, C, H, W), dtype=torch.float32)
    input_f32_hpu = input_f32.to(hpu)
    output2 = input_f32_hpu.bernoulli(p)
    output2_hpu = output2.to(cpu).detach().numpy()

    # verify if the two tensors are same for same seed
    testing.assert_equal(output1_hpu, output2_hpu)


@pytest.mark.parametrize("N, H, W, C, min, max, seed", test_case_list)
def test_hpu_rand_gen_bernoulli_fwd(N, H, W, C, min, max, seed):
    # CPU and HPU uses different algorithm for RNG. Hence they are not compared
    # Instead basic sanity like range and staleness are checked

    input = torch.empty(N, C, H, W, dtype=torch.float)
    in_hpu = input.uniform_(0, 1).to(hpu)
    torch.manual_seed(seed)
    output1 = torch.bernoulli(in_hpu)
    output1_hpu = output1.to(cpu).detach().numpy()

    torch.manual_seed(seed)
    output2 = torch.bernoulli(in_hpu)
    output2_hpu = output2.to(cpu).detach().numpy()

    # verify if the two tensors are same for same seed
    testing.assert_equal(output1_hpu, output2_hpu)

    # verify if the output values exceeds the range
    testing.assert_equal((np.min(output2_hpu) >= 0) and (np.max(output2_hpu) <= 1), True)


if __name__ == "__main__":
    test_hpu_rand_gen_bernoulli_fwd_scalar(*test_case_list[0])
