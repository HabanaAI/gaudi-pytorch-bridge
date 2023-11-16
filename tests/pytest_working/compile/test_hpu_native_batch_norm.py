import torch
import pytest
import habana_frameworks.torch.dynamo.compile_backend

@pytest.mark.parametrize("dtype", [torch.float, torch.bfloat16])
@pytest.mark.parametrize("params", [
    (
        {
            "dims" : (2, 3, 4, 5),
            "momentum" : 0.999,
            "eps" : 1e-5
        }
    )
])
def test_hpu_native_batch_norm_legit_no_training(dtype, params):
    def fn(input, weight, bias, running_mean, running_var, momentum, eps):
        return torch._native_batch_norm_legit_no_training(input, weight, bias, running_mean, running_var, momentum, eps)

    torch._dynamo.reset()
    inductor_compiled_fn = torch.compile(fn)
    aot_hpu_compiled_fn = torch.compile(fn, backend="aot_hpu_training_backend")

    input = torch.randn(*params["dims"], dtype=dtype)
    weight = torch.randn(params["dims"][1])
    bias = torch.randn(params["dims"][1])
    running_mean = torch.randn(params["dims"][1])
    running_var = torch.randn(params["dims"][1])

    cpu_out = inductor_compiled_fn(input, weight, bias, running_mean, running_var, params["momentum"], params["eps"])
    hpu_out = aot_hpu_compiled_fn(input.to("hpu"), weight.to("hpu"), bias.to("hpu"), running_mean.to("hpu"), running_var.to("hpu"), params["momentum"], params["eps"])

    assert torch.allclose(cpu_out[0], hpu_out[0].to("cpu"), equal_nan=True)
