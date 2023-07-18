# ##############################################################################
# Copyright (C) 2023 Habana Labs, Ltd. an Intel Company
# All Rights Reserved.
#
# Unauthorized copying of this file or any element(s) within it, via any medium
# is strictly prohibited. This file contains Habana Labs, Ltd. proprietary and
# confidential information and is subject to the confidentiality and license
# agreements under which it was provided.
#
# ##############################################################################
cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
set(TARGET_NAME "_media_pyt_bridge_C")
add_library(${TARGET_NAME} SHARED
    media_pyt_bridge.cpp
    pyt_media_proxy.cpp
    pyt_media_shim.cpp)

target_compile_definitions(${TARGET_NAME} PRIVATE TORCH_EXTENSION_NAME=${TARGET_NAME})

set_target_properties(${TARGET_NAME} PROPERTIES PREFIX "")
target_link_libraries(${TARGET_NAME} PUBLIC bindings torch ${TORCH_PYTHON_LIBRARY} habana_pytorch_plugin Media)

install(TARGETS ${TARGET_NAME} LIBRARY DESTINATION habana_torch_plugin)
