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

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
project(metrics_tests LANGUAGES CXX)
add_habana_library(metrics_tests SHARED metrics_test_op.cpp metrics_test_kernel.cpp)
target_include_directories(metrics_tests PRIVATE $ENV{PYTORCH_MODULES_ROOT_PATH} $ENV{SWTOOLS_SDK_ROOT}/hl_logger/include/ $ENV{THIRD_PARTIES_ROOT}/magic_enum-0.8.1/include $ENV{THIRD_PARTIES_ROOT} $ENV{PYTORCH_MODULES_ROOT_PATH}/pytorch_helpers $ENV{PYTORCH_MODULES_ROOT_PATH}/backend/helpers $ENV{THIRD_PARTIES_ROOT}/abseil-cpp $ENV{THIRD_PARTIES_ROOT}/fmt-9.1.0/include $ENV{SYNAPSE_ROOT}/include)
target_link_libraries(metrics_tests "${TORCH_LIBRARIES}")

