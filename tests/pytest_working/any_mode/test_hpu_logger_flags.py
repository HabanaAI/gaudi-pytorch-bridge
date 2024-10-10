###############################################################################
# Copyright (C) 2024 Habana Labs, Ltd. an Intel Company
# All Rights Reserved.
#
# Unauthorized copying of this file or any element(s) within it, via any medium
# is strictly prohibited.
# This file contains Habana Labs, Ltd. proprietary and confidential information
# and is subject to the confidentiality and license agreements under which it
# was provided.
#
###############################################################################

from habana_frameworks.torch.utils import _debug_C
from habana_frameworks.torch.utils.debug.logger import enable_logging, get_log_level


def test_hpu_logger_values():
    def check(log_type):
        enable_logging("LOG_LEVEL_ALL", log_type)
        assert _debug_C.is_log_python_enabled(get_log_level(log_type))

    assert not _debug_C.is_log_python_enabled(get_log_level("info"))
    check("info")
    check("trace")
    check("critical")
