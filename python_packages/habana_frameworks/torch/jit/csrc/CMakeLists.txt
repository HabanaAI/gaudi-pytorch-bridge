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


set(TARGET_NAME "_torch_jit_C")
set(TORCH_JIT_ROOT "${CMAKE_CURRENT_SOURCE_DIR}")
set(TORCH_JIT_FORK_ROOT "${TORCH_JIT_ROOT}/jit_fork")

set(JIT_FORK_SRCS)
add_subdirectory(jit_fork)

add_habana_library(${TARGET_NAME} SHARED
    init_python_jit_bindings.cpp
    _C.cpp
    ${JIT_FORK_SRCS}
    )

target_compile_definitions(${TARGET_NAME} PRIVATE TORCH_EXTENSION_NAME=${TARGET_NAME})

set_target_properties(${TARGET_NAME} PROPERTIES PREFIX "")
target_link_libraries(${TARGET_NAME} PUBLIC bindings torch ${TORCH_PYTHON_LIBRARY})
target_include_directories(${TARGET_NAME} PRIVATE "${TORCH_JIT_ROOT}")


install(TARGETS ${TARGET_NAME} LIBRARY DESTINATION habana_torch_plugin)
