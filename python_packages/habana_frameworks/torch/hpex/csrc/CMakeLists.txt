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
set(TARGET_NAME "_hpex_C")
add_library(${TARGET_NAME} SHARED
    bindings.cpp
    embedding_bag_preproc.cpp
    embedding_bag_preproc_main.cpp)
target_compile_definitions(${TARGET_NAME} PRIVATE TORCH_EXTENSION_NAME=${TARGET_NAME})

set_target_properties(${TARGET_NAME} PROPERTIES PREFIX "")
target_link_libraries(${TARGET_NAME} PUBLIC bindings torch ${TORCH_PYTHON_LIBRARY})
install(TARGETS ${TARGET_NAME} LIBRARY DESTINATION habana_torch_plugin)
