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


if __name__ == "__main__":
    test_hpu_deterministic_api_init()
    test_hpu_deterministic_api()
