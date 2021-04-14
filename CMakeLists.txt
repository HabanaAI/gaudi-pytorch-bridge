###############################################################################
# Copyright (C) 2020 HabanaLabs, Ltd.
# All Rights Reserved.
#
# Unauthorized copying of this file, via any medium is strictly prohibited.
# Proprietary and confidential.
#
################################################################################
cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
if(POLICY CMP0076)
  cmake_policy(SET CMP0076 OLD)
endif()
project(habana_pytorch_plugin VERSION 0.8.0 LANGUAGES CXX)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
option(SANITIZER "build sanitizer configuration" OFF)
if(SANITIZER)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -fsanitize=undefined -fsanitize-address-use-after-scope -O0")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fsanitize=address -fsanitize=leak")
endif(SANITIZER)
find_program(CCACHE ccache)
if (CCACHE)
  set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE}")
endif()

set(DISABLE_CLANG_TIDY "OFF")
if (NOT DEFINED CLANG_TIDY OR CLANG_TIDY)
  find_program(CLANG_TIDY clang-tidy-10)
else (${CLANG_TIDY} STREQUAL "")
  set(DISABLE_CLANG_TIDY "ON")
endif ()
if (CLANG_TIDY)
  set(CMAKE_CXX_CLANG_TIDY "${CLANG_TIDY}")
elseif (${DISABLE_CLANG_TIDY} STREQUAL "ON")
  message(AUTHOR_WARNING "clang-tidy-10 was disabled")
else ()
  message(FATAL_ERROR "clang-tidy-10 was not detected. Please install it and rerun CMake or set CLANG_TIDY to a valid path")
endif ()

find_package(Torch REQUIRED)
add_subdirectory(cmake/json)
add_subdirectory(third_party/pybind11)
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
include(AddNPUStackDependencies)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
add_subdirectory(pytorch_helpers)

# workaround for cmake3.5. Starting from 3.13 sources are optional and we can add them in add_subdirectories
add_library(habana_pytorch_plugin SHARED
  habana_bridge/kernel/hpu_habana_launch_op_pt.cpp
  habana_bridge/kernel/hpu_habna_launch_op_opt_pass.cpp
  habana_bridge/kernel/hpu_habana_cache.cpp
  habana_bridge/kernel/hpu_process_control_edges.cpp
  habana_bridge/passes/habana_fuser.cpp
  habana_bridge/passes/remove_inplace_ops.cpp
  pytorch_helpers/habana_device/HPUAllocator.cpp
  pytorch_helpers/habana_device/PinnedMemoryAllocator.cpp
  pytorch_helpers/habana_helpers/tensor_info.cpp
  pytorch_helpers/habana_helpers/tensor_utils.cpp
  habana_hooks/HabanaHooks.cpp
  habana_kernels/aten_hpu_type_default.cpp
  habana_kernels/basic_kernels.cpp
  habana_kernels/binary_kernels.cpp
  habana_kernels/binary_out_kernels.cpp
  habana_kernels/binary_composite_kernels.cpp
  habana_kernels/binary_inplace_kernels.cpp
  habana_kernels/bitwise_kernels.cpp
  habana_kernels/compare_kernels.cpp
  habana_kernels/conv_kernels.cpp
  habana_kernels/conv_bwd_kernels.cpp
  habana_kernels/conv_pool_utils.cpp
  habana_kernels/embedding_kernels.cpp
  habana_kernels/fill_kernels.cpp
  habana_kernels/habana_operator.cpp
  habana_kernels/index_kernels.cpp
  habana_kernels/nonzero_kernel.cpp
  habana_kernels/kernels_register.cpp
  habana_kernels/kernel_utils.cpp
  habana_kernels/lazy_kernels.cpp
  habana_kernels/linear_kernels.cpp
  habana_kernels/loss_kernels.cpp
  habana_kernels/nms_kernels.cpp
  habana_kernels/norm_kernels.cpp
  habana_kernels/optimizer_kernels.cpp
  habana_kernels/optimizer_lamb.cpp
  habana_kernels/pool_kernels.cpp
  habana_kernels/random_gen_kernels.cpp
  habana_kernels/reduction_kernels.cpp
  habana_kernels/reduction2_kernels.cpp
  habana_kernels/repeat.cpp
  habana_kernels/scalar.cpp
  habana_kernels/simple_generic_kernel.cpp
  habana_kernels/softmax_kernels.cpp
  habana_kernels/special_function_kernels.cpp
  habana_kernels/tensor_factories.cpp
  habana_kernels/tensor_shape_kernels.cpp
  habana_kernels/threshold_kernels.cpp
  habana_kernels/topk_kernels.cpp
  habana_kernels/unary_kernels.cpp
  habana_kernels/upsample_kernels.cpp
  habana_kernels/where_kernels.cpp
  habana_lazy/aten_lazy_bridge.cpp
  habana_lazy/debug_utils.cpp
  habana_lazy/hpu_lazy_tensors.cpp
  habana_lazy/ir.cpp
  habana_lazy/ir_utils.cpp
  habana_lazy/tensor_impl.cpp
  habana_lazy/hlexec.cpp

  habana_lazy/passes/fuse_mm_transpose.cpp
  habana_lazy/passes/fuse_bn_relu_residual_add.cpp
  habana_lazy/passes/permute_graph.cpp
  habana_lazy/passes/transform_graph.cpp
  habana_lazy/passes/replace_inplace_ops.cpp
  habana_lazy/hpu_lazy_cache.cpp
  habana_lazy/lazy_executor.cpp
)
target_compile_features(habana_pytorch_plugin PRIVATE cxx_std_14)
target_compile_definitions(habana_pytorch_plugin PRIVATE "GENERIC_HELPERS" MAX_DEVICES_PER_BOX=8)
#TODO: remove -Wno-error=deprecated-declarations after removing functions marked [[deprecated]]
target_compile_options(habana_pytorch_plugin PRIVATE -Wall -Wextra -Werror -Wno-error=deprecated-declarations)
# Link against LibTorch and synapse
target_link_libraries(habana_pytorch_plugin PRIVATE "${TORCH_LIBRARIES}" pytorch_synapse_helpers_abi11 json)
target_include_directories(habana_pytorch_plugin PRIVATE ${CMAKE_SOURCE_DIR})

pybind11_add_module(hb_torch SHARED habana_bridge/register/register.cpp)
target_compile_definitions(hb_torch PRIVATE "GENERIC_HELPERS" MAX_DEVICES_PER_BOX=8)
target_compile_options(hb_torch PRIVATE -Wall -Wextra -Wno-error=deprecated-declarations)
target_link_libraries(hb_torch PRIVATE "${TORCH_LIBRARIES}" habana_pytorch_plugin pytorch_synapse_helpers_abi11)
target_include_directories(hb_torch PRIVATE ${CMAKE_SOURCE_DIR})
target_compile_features(hb_torch PRIVATE cxx_std_14)

add_custom_command(OUTPUT ${CMAKE_SOURCE_DIR}/habana_kernels/aten_hpu_type_default.cpp
                   COMMAND python ${CMAKE_SOURCE_DIR}/scripts/gen.py --gen_class_mode
                     --output_folder=${CMAKE_SOURCE_DIR}/habana_kernels
                     ${CMAKE_SOURCE_DIR}/habana_kernels/wrap_kernels_declarations.h
                     ${TORCH_INSTALL_PREFIX}/include/torch/csrc/autograd/generated/RegistrationDeclarations.h
                     ${TORCH_INSTALL_PREFIX}/include/ATen/Functions.h
                   MAIN_DEPENDENCY scripts/gen.py
                   DEPENDS
                     ${CMAKE_SOURCE_DIR}/habana_kernels/wrap_kernels_declarations.h
                     ${TORCH_INSTALL_PREFIX}/include/torch/csrc/autograd/generated/RegistrationDeclarations.h
                     ${TORCH_INSTALL_PREFIX}/include/ATen/Functions.h
)

# Create symlink to latest build from tests dir
add_custom_command(TARGET habana_pytorch_plugin POST_BUILD COMMAND
  ${CMAKE_COMMAND} -E create_symlink $<TARGET_FILE:habana_pytorch_plugin>
  ${CMAKE_SOURCE_DIR}/tests/$<TARGET_FILE_NAME:habana_pytorch_plugin>
  COMMAND ${CMAKE_COMMAND} -E create_symlink $<TARGET_FILE:habana_pytorch_plugin>
  $ENV{BUILD_ROOT_LATEST}/$<TARGET_FILE_NAME:habana_pytorch_plugin>
)
add_custom_command(TARGET hb_torch POST_BUILD COMMAND
  ${CMAKE_COMMAND} -E create_symlink $<TARGET_FILE:hb_torch>
  ${CMAKE_SOURCE_DIR}/tests/$<TARGET_FILE_NAME:hb_torch>
)

option(BUILD_TESTS "Build gtest binaries" ON)
if (BUILD_TESTS)
  add_subdirectory("$ENV{HABANA_SOFTWARE_STACK}/3rd-parties/googletest_1_10/"
                   ${CMAKE_BINARY_DIR}/ThirdParty/googletest/
                   EXCLUDE_FROM_ALL)
  add_subdirectory(tests/cpp)
endif()


option(BUILD_PKGS "Build python packages" ON)
option(INSTALL_PKGS "Install python packages" OFF)
if (BUILD_PKGS)
  find_program(PIP "pip" REQUIRED)
  set(PKGS
    ${CMAKE_SOURCE_DIR}/habana_kernels/pytorch-extensions/embedding_bag_extension
    ${CMAKE_SOURCE_DIR}/habana_kernels/pytorch-extensions/habana_custom-extension
    ${CMAKE_SOURCE_DIR}/habana_kernels/pytorch-extensions/gather2d-extension
    ${CMAKE_SOURCE_DIR}/habana_kernels/pytorch-extensions/optimizer_adagrad-extension
    ${CMAKE_SOURCE_DIR}/habana_kernels/pytorch-extensions/optimizer_sgd-extension
    ${CMAKE_SOURCE_DIR}/habana_kernels/pytorch-extensions/preproc-extension
    ${CMAKE_SOURCE_DIR}/pytorch_helpers/hmp
    $ENV{MODEL_GARDEN_PYTORCH_PATH}/nlp/bert/finetuning
  )

  foreach(PKG ${PKGS})
    file(GLOB_RECURSE DEPS ${PKG}/*)
    get_filename_component(name ${PKG} NAME)
    set(OUTPUT "${CMAKE_BINARY_DIR}/pkgs/${name}")
    if (INSTALL_PKGS)
      set(PIPARGS install -U ${PKG}/)
    else()
      set(PIPARGS wheel --no-deps --wheel-dir=${CMAKE_BINARY_DIR}/pkgs ${PKG})
    endif()
    add_custom_command(OUTPUT ${OUTPUT}
                       COMMAND ${PIP} ${PIPARGS}
                       COMMAND ${CMAKE_COMMAND} -E touch ${OUTPUT}
                       DEPENDS habana_pytorch_plugin
                       DEPENDS ${DEPS})
    add_custom_target(${name} ALL DEPENDS ${OUTPUT})
  endforeach()
endif()
