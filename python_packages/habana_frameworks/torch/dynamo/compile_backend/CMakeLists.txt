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
set(TARGET_NAME "_recipe_compiler_C")
add_library(${TARGET_NAME} SHARED
    recipe_compiler.cpp)
target_compile_definitions(${TARGET_NAME} PRIVATE TORCH_EXTENSION_NAME=${TARGET_NAME})
set_target_properties(${TARGET_NAME} PROPERTIES PREFIX "")
target_link_libraries(${TARGET_NAME} PUBLIC bindings torch absl::hash ${TORCH_PYTHON_LIBRARY})
install(TARGETS ${TARGET_NAME} LIBRARY DESTINATION habana_torch_plugin/dynamo/compile_backend)

set(TARGET_NAME "_shared_layer_C")
add_custom_command(OUTPUT
${CMAKE_BINARY_DIR}/generated/check_kernel_support/op_def.cpp
COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_SOURCE_DIR}/scripts/gen_op.py
                     --output_dir=${CMAKE_BINARY_DIR}/generated
                     --yaml=${CMAKE_SOURCE_DIR}/scripts/hpu_op.yaml
                     ${CMAKE_SOURCE_DIR}/habana_kernels/wrap_kernels_declarations.h
                     ${PT_VER_DIR}/habana_kernels_ver/wrap_kernels_declarations.h
                     ${TORCH_INSTALL_PREFIX}/include/ATen/RegistrationDeclarations.h
                     ${TORCH_INSTALL_PREFIX}/include/ATen/RedispatchFunctions.h
                     ${TORCH_INSTALL_PREFIX}/../torchgen/packaged/ATen/native/native_functions.yaml
                     --check_kernel_support=True
                   MAIN_DEPENDENCY ${CMAKE_SOURCE_DIR}/scripts/gen_op.py
                   DEPENDS
                     ${CMAKE_SOURCE_DIR}/scripts/hpu_op.yaml
                     ${CMAKE_SOURCE_DIR}/habana_kernels/wrap_kernels_declarations.h
                     ${PT_VER_DIR}/habana_kernels_ver/wrap_kernels_declarations.h
                     ${TORCH_INSTALL_PREFIX}/include/ATen/RegistrationDeclarations.h
                     ${TORCH_INSTALL_PREFIX}/include/ATen/RedispatchFunctions.h
)
add_library(${TARGET_NAME} SHARED
    shared_layer_bindings.cpp
    ${CMAKE_BINARY_DIR}/generated/check_kernel_support/op_def.cpp)
target_include_directories(_shared_layer_C PRIVATE ${CMAKE_BINARY_DIR}/generated/check_kernel_support ${CMAKE_BINARY_DIR}
    ${CMAKE_SOURCE_DIR}/python_packages/habana_frameworks/torch/dynamo/compile_backend)
target_compile_definitions(${TARGET_NAME} PRIVATE TORCH_EXTENSION_NAME=${TARGET_NAME})
set_target_properties(${TARGET_NAME} PROPERTIES PREFIX "")
target_link_libraries(${TARGET_NAME} PUBLIC bindings torch absl::hash ${TORCH_PYTHON_LIBRARY})
install(TARGETS ${TARGET_NAME} LIBRARY DESTINATION habana_torch_plugin/dynamo/compile_backend)
