<<<<<<< HEAD:tests/test_habanaframework_api.py
import torch
import habana_frameworks.torch as htorch
=======
>>>>>>> 1c8e3092c... [SW-140881] python tests for PT, part 6:tests/pytest_working/test_habanaframework_api.py
import os

import habana_frameworks.torch as htorch
import pytest
import torch
from test_utils import env_var_in_scope, hpu

<<<<<<< HEAD:tests/test_habanaframework_api.py
pytestmark = pytest.mark.skip(reason="Tests in this file are chaning env variables")
=======
>>>>>>> 1c8e3092c... [SW-140881] python tests for PT, part 6:tests/pytest_working/test_habanaframework_api.py

# Use torch_hpu APIs, equivalent to torch.cuda APIs
@pytest.mark.xfail(
    reason="libhlml.so: cannot open shared object file: No such file or directory"
)
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
