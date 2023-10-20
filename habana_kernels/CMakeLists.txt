
# ##############################################################################
# Copyright (C) 2021-2023 Habana Labs, Ltd. an Intel Company
# All Rights Reserved.
#
# Unauthorized copying of this file or any element(s) within it, via any medium
# is strictly prohibited. This file contains Habana Labs, Ltd. proprietary and
# confidential information and is subject to the confidentiality and license
# agreements under which it was provided.
#
# ##############################################################################

add_habana_library(habana_kernels OBJECT
  index_kernels.cpp
  )

set_target_properties(habana_kernels PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
)