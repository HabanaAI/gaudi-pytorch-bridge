###############################################################################
# Copyright (C) 2022 Habana Labs, Ltd. an Intel Company
# All Rights Reserved.
#
# Unauthorized copying of this file or any element(s) within it, via any medium
# is strictly prohibited.
# This file contains Habana Labs, Ltd. proprietary and confidential information
# and is subject to the confidentiality and license agreements under which it
# was provided.
#
###############################################################################
import build_profiles.profiles as profiles


def test_check_profile_integrity(capsys):
    profiles.check_profile_file_integrity()
    captured = capsys.readouterr()
    assert captured.out == "OK\n"
