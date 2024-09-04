# ##############################################################################
# Copyright (C) 2024 Habana Labs, Ltd. an Intel Company
# All Rights Reserved.
#
# Unauthorized copying of this file or any element(s) within it, via any medium
# is strictly prohibited. This file contains Habana Labs, Ltd. proprietary and
# confidential information and is subject to the confidentiality and license
# agreements under which it was provided.
#
# ##############################################################################

file(GLOB_RECURSE jit_sources ${CMAKE_CURRENT_SOURCE_DIR}/**.cpp )
set(JIT_FORK_SRCS ${jit_sources} PARENT_SCOPE)

