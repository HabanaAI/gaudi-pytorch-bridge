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

import torch


# Check API doesn't acquire any device on invoke
def test_hpu_deterministic_api():
    try:
        import habana_frameworks.torch.hpu as htcore

        device_count = htcore.device_count()
        old_deterministic = torch.are_deterministic_algorithms_enabled()
        torch.use_deterministic_algorithms(True)
        new_device_count = htcore.device_count()
        torch.use_deterministic_algorithms(old_deterministic)
        assert device_count == new_device_count
    except ImportError as e:
        print(f"failed importing habana_frameworks.torch.hpu with ImportError: {e=}")


def test_hpu_deterministic_api_init():
    try:
        import habana_frameworks.torch.hpu as htcore

        device_status = htcore.is_initialized()
        old_deterministic = torch.are_deterministic_algorithms_enabled()
        torch.use_deterministic_algorithms(True)
        new_device_status = htcore.is_initialized()
        torch.use_deterministic_algorithms(old_deterministic)
        assert device_status == new_device_status
    except ImportError as e:
        print(f"failed importing habana_frameworks.torch.hpu with ImportError: {e=}")
