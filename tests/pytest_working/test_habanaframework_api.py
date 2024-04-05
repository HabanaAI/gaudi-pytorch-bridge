import os

import habana_frameworks.torch as htorch
import pytest
import torch
from test_utils import env_var_in_scope, hpu


# Use torch_hpu APIs, equivalent to torch.cuda APIs
@pytest.mark.skip(reason="libhlml.so: cannot open shared object file: No such file or directory")
def test_basic_apis():
    print("hpu available", htorch.hpu.is_available())
    print("hpu device count", htorch.hpu.device_count())
    print("hpu device name", htorch.hpu.get_device_name())
    print("hpu current device", htorch.hpu.current_device())
    print("hpu synchronize", htorch.hpu.synchronize())
    print("hpu memory_usage", htorch.hpu.memory_usage())
    print("hpu utilization", htorch.hpu.utilization())
    print("hpu is_bf16_supported", htorch.hpu.is_bf16_supported())

    print("hpu get_device_capability", htorch.hpu.get_device_capability(hpu))
    print("hpu get_device_properties", htorch.hpu.get_device_properties(hpu))
    print("hpu get_arch_list", htorch.hpu.get_arch_list())
    print("hpu get_gencode_flags", htorch.hpu.get_gencode_flags())
    if htorch.hpu.device_count() >= 2:
        print("hpu can_device_access_peer", htorch.hpu.can_device_access_peer(0, 1))
        with env_var_in_scope({"HLS_MODULE_ID": "1"}):
            htorch.hpu.set_device(1)
            print(os.getenv("HLS_MODULE_ID"))
        with env_var_in_scope({"HLS_MODULE_ID": "0"}):
            with htorch.hpu.device(0):
                print(os.getenv("HLS_MODULE_ID"))

    htorch.core.mark_step()

    print("dist init", htorch.distributed.hccl.initialize_distributed_hpu())


def test_device_synchronize_api():
    torch.zeros(10, 2).to("hpu")
    tB_h = torch.full((1000,), 1, device="hpu")  # noqa
    htorch.hpu.synchronize()  # Need verify with the log


def test_get_device_index_api():
    try:
        htorch.hpu._get_device_index("hpu0", optional=True)
    except Exception as err:
        assert err != "Invalid device string"
        pass

    # with self.assertRaisesRegex(ValueError, "Expected a hpu device"):
    try:
        cpu_device = torch.device("cpu")
        htorch.hpu._get_device_index(cpu_device, optional=True)
    except Exception as err:
        assert err != "Expected a hpu device"

    index = htorch.hpu._get_device_index("hpu:1")
    assert index == 1
    index = htorch.hpu._get_device_index(torch.device("hpu:2"))
    assert index == 2
