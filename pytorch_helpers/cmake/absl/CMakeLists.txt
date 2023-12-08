###############################################################################
# Copyright (C) 2021-2023 Habana Labs, Ltd. an Intel Company
# All Rights Reserved.
#
# Unauthorized copying of this file or any element(s) within it, via any medium
# is strictly prohibited.
# This file contains Habana Labs, Ltd. proprietary and confidential information
# and is subject to the confidentiality and license agreements under which it
# was provided.
#
###############################################################################

set(CMAKE_CXX_CLANG_TIDY "")
set(CMAKE_CXX_INCLUDE_WHAT_YOU_USE "")

set(CMAKE_CXX_FLAGS -w)
set(CMAKE_CXX_STANDARD 17)

set(BUILD_TESTING OFF)

set(ABSL_PROPAGATE_CXX_STD ON)
add_subdirectory($ENV{THIRD_PARTIES_ROOT}/abseil-cpp absl SYSTEM)

include(ExportAbseil)
