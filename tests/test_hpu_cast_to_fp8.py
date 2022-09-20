import os
import torch
import numpy as np
import pytest
from habana_frameworks.torch.hpex.kernels.CastToFp8 import cast_to_fp8

# Test data
input_and_expected = [(2.0, 2.0),
                      (-2.0, -2.0),
                      (3.0, 3.0),
                      (-3.0, -3.0),
                      (3.2, 3.0),
                      (-3.2, -3.0),
                      (3.26, 3.5),
                      (-3.26, -3.5),
                      (3.25, 3.0),
                      (-3.25, -3.0),
                      (25697.81367237185, 24576),
                      (-25697.81367237185, -24576),
                      (31518.35309567908, 32768),
                      (-31518.35309567908, -32768),
                      (64275.36250533416, np.inf),
                      (66000, np.inf),
                      (90000, np.inf),
                      (-90000, -np.inf),
                      (1.0, 1.0),
                      (-1.0, -1.0),
                      (0.5, 0.5),
                      (-0.5, -0.5),
                      (0.9, 0.875),
                      (-0.9, -0.875),
                      (0.46875, 0.5),
                      (-0.46875, -0.5),
                      (0.48, 0.5),
                      (-0.48, -0.5),
                      (0.46, 0.4375),
                      (-0.46, -0.4375),
                      (0.00000667572021484375, 0),
                      (-0.00000667572021484375, -0), ]

data = []
expected_data = []
for input, expected in input_and_expected:
    data.append(input)
    expected_data.append(expected)

@pytest.mark.parametrize("device", [torch.device("hpu:0")])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("size", [2, 4, 8, 16])
@pytest.mark.parametrize("batched", [True, False])
def test_matmul_fp8(device, dtype, size, batched):
    # calculate cpu reference
    quantized_data = torch.tensor(expected_data, dtype=dtype, device=torch.device("cpu"))
    t1 = quantized_data.reshape([-1, size])
    t2 = quantized_data.reshape([size, -1])
    if batched:
        t1 =torch.unsqueeze(t1, 0)
        t2 = torch.unsqueeze(t2, 0)
        t1 = t1.expand(2, -1, size)
        t2 = t2.expand(2, size, -1)

    out = torch.matmul(t1, t2)
    out = out.to(torch.float)

    # quantize and calculate hpu result
    input_data = torch.tensor(data, dtype=dtype, device=device)
    input_data1 = input_data.reshape([-1, size])
    input_data2 = input_data.reshape([size, -1])
    if batched:
        input_data1 = torch.unsqueeze(input_data1, 0)
        input_data2 = torch.unsqueeze(input_data2, 0)
        input_data1 = input_data1.expand(2, -1, size)
        input_data2 = input_data2.expand(2, size, -1)

    # should be equivalent to .to(torch.fp8r152)
    t1_h = cast_to_fp8(input_data1)
    t2_h = cast_to_fp8(input_data2)

    out_h = torch.matmul(t1_h, t2_h).to(dtype)
    out_h = out_h.to(torch.float).cpu()
    assert np.array_equal(out_h, out, equal_nan=True), f"Data mismatch"

@pytest.mark.parametrize("device", [torch.device("hpu:0")])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("stochastic_rounding", [True, False])
@pytest.mark.parametrize("seed", [0, 12342])
def test_cast_with_stochastic_rounding(device, dtype, stochastic_rounding, seed):
    os.environ["ENABLE_CONTIGUOUS_CAST_REMOVAL"] = "false"
    os.environ["ENABLE_EXPERIMENTAL_FLAGS"] = "1"
    input_value = 18.5
    input_data = torch.tensor([input_value] * 1000, dtype=dtype, device=device)
    casted = cast_to_fp8(input_data, stochastic_rounding=stochastic_rounding, seed=seed)
    upcasted = casted.to(dtype)
    mean = torch.mean(upcasted).cpu()
    # When stochastic rounding is turned off, 18.5 will be rounded to 20.0 with default rounding mode
    # (or 16.0 when rounded down). With stochastic rounding, it rounds up or down with the probability
    # dependent on the distance between original value to the closest fp8 numbers, so the mean result
    # should be close to the input value.
    if stochastic_rounding:
        assert mean < 19.5
        assert mean > 17.5
    else:
        assert mean == 20.0
