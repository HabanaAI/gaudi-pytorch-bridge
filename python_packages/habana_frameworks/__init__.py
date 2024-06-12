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

is_loaded = False  # A member variable of habana_frameworks module to track if our module has been imported


def __autoload():
    # This is an entrypoint for pytorch autoload mechanism
    # If the following confition is true, that means our backend has already been loaded, either explicitly
    # or by the autoload mechanism and importing it again should be skipped to avoid circular imports
    global is_loaded
    if is_loaded:
        return
    import habana_frameworks.torch
