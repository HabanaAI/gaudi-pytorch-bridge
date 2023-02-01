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

cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
if(POLICY CMP0076)
  cmake_policy(SET CMP0076 OLD)
endif()
project(habana_pytorch_plugin VERSION 0.8.0 LANGUAGES CXX)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)

option(SANITIZER "build sanitizer configuration" OFF)
option(MANYLINUX "build for manylinux configuration, synapse shim will be used" ON)
option(UPSTREAM_COMPILE "build for pytorch-fork upstream branch" OFF)

if(MANYLINUX)
  message("Building with shim layer")
  set (PYTORCH_SYNAPSE_UTILS_LIB pytorch_synapse_utils_shim)
else()
  message("Building without shim layer")
  set (PYTORCH_SYNAPSE_UTILS_LIB SynapseUtils)
endif()

if(UPSTREAM_COMPILE)
  ADD_DEFINITIONS(-DUPSTREAM_COMPILE)
endif(UPSTREAM_COMPILE)

if(SANITIZER)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -fsanitize=undefined -fno-sanitize=vptr -fsanitize-address-use-after-scope -O0")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fsanitize=address -fsanitize=leak")
endif(SANITIZER)
find_program(CCACHE ccache)
if (CCACHE)
  set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE}")
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CLANG_TIDY "")
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

get_property(torch_flags TARGET torch PROPERTY INTERFACE_COMPILE_OPTIONS)
if(NOT "${torch_flags}" STREQUAL "")
  add_compile_options("${torch_flags}")
endif()

add_subdirectory(cmake/json)
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

include(pt_ver)
set_up_pt_ver_mechanism()

include(AddNPUStackDependencies)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)

set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
set(CMAKE_INSTALL_RPATH "$ORIGIN")

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
add_subdirectory(pytorch_helpers)

# workaround for cmake3.5. Starting from 3.13 sources are optional and we can add them in add_subdirectories
add_library(habana_pytorch_plugin SHARED
  habana_bridge/kernel/ds_graph_recompile.cpp
  habana_bridge/kernel/hpu_habana_launch_op_pt.cpp
  habana_bridge/kernel/hpu_habana_launch_op_pt_lowering.cpp
  habana_bridge/kernel/hpu_habana_launch_op_pt_sif.cpp
  habana_bridge/kernel/hpu_habana_cache.cpp
  habana_bridge/kernel/hpu_process_control_edges.cpp
  habana_bridge/kernel/hpu_shape_inference.cpp
  habana_bridge/kernel/refinement_engine.cpp
  habana_bridge/passes/habana_fuser.cpp
  habana_bridge/passes/remove_inplace_ops.cpp
  habana_bridge/passes/hpu_habana_persistence_marker_pass.cpp
  habana_bridge/program/cache.cpp
  habana_bridge/program/splitter.cpp
  habana_bridge/program/program.cpp
  habana_bridge/program/strategy.cpp
  habana_bridge/program/executor.cpp
  habana_bridge/program/eager_splitting_strategy.cpp
  habana_bridge/program/naive_rpo_scheduling_strategy.cpp
  pytorch_helpers/habana_device/HPUAllocator.cpp
  pytorch_helpers/habana_device/HPUGraph.cpp
  pytorch_helpers/habana_device/HPUStream.cpp
  pytorch_helpers/habana_device/PinnedMemoryAllocator.cpp
  pytorch_helpers/habana_helpers/logging.cpp
  pytorch_helpers/habana_helpers/signal_handler.cpp
  pytorch_helpers/habana_helpers/stats.cpp
  pytorch_helpers/habana_helpers/tensor_info.cpp
  pytorch_helpers/habana_helpers/tensor_utils.cpp
  pytorch_helpers/habana_helpers/tensor_shape.cpp
  pytorch_helpers/habana_helpers/dynamic_bucket_info.cpp
  pytorch_helpers/habana_helpers/dynamic_bucket_info_utils.cpp
  pytorch_helpers/habana_helpers/dynamic_shape_info.cpp
  pytorch_helpers/habana_helpers/compilation_statistics.cpp
  pytorch_helpers/habana_helpers/collective_kernel_info.cpp
  pytorch_helpers/habana_helpers/kernels_accumulation.cpp
  pytorch_helpers/habana_helpers/misc_utils.cpp
  pytorch_helpers/habana_helpers/thread_pool/acc_thread_pool.cpp
  pytorch_helpers/habana_helpers/thread_pool/thread_pool.cpp
  pytorch_helpers/habana_helpers/dtype_helpers.cpp
  pytorch_helpers/habana_helpers/cast_sequence.cpp
  pytorch_helpers/habana_helpers/profiling/json_activity_profiler.cpp
  pytorch_helpers/habana_helpers/profiling/kineto_activity_profiler.cpp
  pytorch_helpers/habana_helpers/event_dispatcher.cpp
  pytorch_helpers/habana_helpers/profiling/profiling.cpp
  pytorch_helpers/habana_helpers/profiling/trace_sources/synapse_logger_source.cpp
  pytorch_helpers/habana_helpers/profiling/trace_sources/synapse_profiler_source.cpp
  pytorch_helpers/habana_helpers/profiling/trace_sources/trace_parser.cpp
  pytorch_helpers/serialize/export.cpp
  pytorch_helpers/util/jitgraph_utils.cpp
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
  habana_kernels/fallback_helper.cpp
  habana_kernels/fill_kernels.cpp
  habana_kernels/eager_op.cpp
  habana_kernels/habana_operator.cpp
  habana_kernels/hccl_kernels.cpp
  habana_kernels/index_kernels.cpp
  habana_kernels/nonzero_kernel.cpp
  habana_kernels/kernel_input_checks.cpp
  habana_kernels/kernels_register.cpp
  habana_kernels/kernel_utils.cpp
  habana_kernels/lazy_kernels.cpp
  habana_kernels/lazy_optimizer_kernels.cpp
  habana_kernels/lazy_op.cpp
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
  habana_kernels/roi_align_kernels.cpp
  habana_kernels/scalar.cpp
  habana_kernels/simple_generic_kernel.cpp
  habana_kernels/softmax_kernels.cpp
  habana_kernels/tensor_shape_kernels.cpp
  habana_kernels/threshold_kernels.cpp
  habana_kernels/triangular_kernels.cpp
  habana_kernels/topk_kernels.cpp
  habana_kernels/unary_kernels.cpp
  habana_kernels/upsample_kernels.cpp
  habana_kernels/custom_op_kernel.cpp
  habana_kernels/lowering_util.cpp
  habana_lazy/aten_lazy_bridge.cpp
  habana_lazy/lazy_graph_hash_disabler.cpp
  habana_lazy/debug_utils.cpp
  habana_lazy/memlog.cpp
  habana_lazy/hpu_lazy_tensors.cpp
  habana_lazy/ir.cpp
  habana_lazy/ir_utils.cpp
  habana_lazy/tensor_impl.cpp
  habana_lazy/hlexec.cpp
  habana_lazy/habana_custom_op.cpp
  habana_lazy/passes/pass_utils.cpp
  habana_lazy/passes/recalculate_batchnorm_params.cpp
  habana_lazy/passes/fuse_mm_transpose.cpp
  habana_lazy/passes/fuse_bn_relu_residual_add.cpp
  habana_lazy/passes/permute_graph.cpp
  habana_lazy/passes/remove_redundant_memcpy.cpp
  habana_lazy/passes/transform_graph.cpp
  habana_lazy/passes/replace_inplace_ops.cpp
  habana_lazy/passes/replace_views_with_reshapes.cpp
  habana_lazy/passes/weight_permute_graph.cpp
  habana_lazy/hpu_lazy_cache.cpp
  habana_lazy/lazy_executor.cpp
  habana_lazy/permute_tensors.cpp
  habana_lazy/sbs_debug.cpp
  habana_lazy/sbs_runner.cpp
  habana_lazy/visualize.cpp
  habana_lazy/lazy_storage.cpp
  habana_lazy/lazy_graph_hash_builder.cpp
  habana_lazy/view_utils.cpp
  habana_lazy/hpu_stage_submission.cpp
  ${PT_VER_DIR}/habana_kernels_ver/lazy_kernels.cpp
)
add_subdirectory(hpu_ops)
target_compile_features(habana_pytorch_plugin PRIVATE cxx_std_14)
target_compile_definitions(habana_pytorch_plugin PRIVATE "GENERIC_HELPERS" MAX_DEVICES_PER_BOX=8)
#TODO: remove -Wno-error=deprecated-declarations after removing functions marked [[deprecated]]
target_compile_options(habana_pytorch_plugin PRIVATE -Wall -Wextra -Werror -Wno-error=deprecated-declarations)
# Link against LibTorch and synapse
target_link_libraries(habana_pytorch_plugin
  PRIVATE
    habana_serialization
    torch
    pytorch_synapse_helpers_abi11
    ${PYTORCH_SYNAPSE_UTILS_LIB}
    json
    tensor_comparator_core
    -lstdc++fs
)

target_include_directories(habana_pytorch_plugin PRIVATE ${CMAKE_SOURCE_DIR})
target_include_directories(habana_pytorch_plugin PRIVATE ${PYTHON_INCLUDE_DIR} ${CMAKE_BINARY_DIR})
if (DEFINED ENV{KINETO_ROOT})
  target_include_directories(habana_pytorch_plugin PRIVATE $ENV{KINETO_ROOT}/libkineto/include $ENV{KINETO_ROOT}/libkineto/third_party/fmt/include)
else()
  message(FATAL_ERROR "KINETO_ROOT is not defined, please update automation repository")
endif()

add_custom_command(OUTPUT
                     ${CMAKE_BINARY_DIR}/generated/lazy/hpu_op0.cpp
                     ${CMAKE_BINARY_DIR}/generated/lazy/hpu_op1.cpp
                     ${CMAKE_BINARY_DIR}/generated/lazy/hpu_op2.cpp
                     ${CMAKE_BINARY_DIR}/generated/lazy/hpu_op3.cpp
                     ${CMAKE_BINARY_DIR}/generated/lazy/hpu_op4.cpp
                     ${CMAKE_BINARY_DIR}/generated/hpu_autocast_ops.cpp
                     ${CMAKE_BINARY_DIR}/generated/wrap_kernels_registrations.cpp
                     ${CMAKE_BINARY_DIR}/generated/backend/hpu_op0.cpp
                     ${CMAKE_BINARY_DIR}/generated/backend/hpu_op1.cpp
                     ${CMAKE_BINARY_DIR}/generated/backend/hpu_op2.cpp
                     ${CMAKE_BINARY_DIR}/generated/backend/hpu_op3.cpp
                     ${CMAKE_BINARY_DIR}/generated/backend/hpu_op4.cpp
                   COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_SOURCE_DIR}/scripts/gen_op.py
                     --output_dir=${CMAKE_BINARY_DIR}/generated
                     --yaml=${CMAKE_SOURCE_DIR}/scripts/hpu_op.yaml
                     ${CMAKE_SOURCE_DIR}/habana_kernels/wrap_kernels_declarations.h
                     ${PT_VER_DIR}/habana_kernels_ver/wrap_kernels_declarations.h
                     ${TORCH_INSTALL_PREFIX}/include/ATen/RegistrationDeclarations.h
                     ${TORCH_INSTALL_PREFIX}/include/ATen/RedispatchFunctions.h
                     ${TORCH_INSTALL_PREFIX}/../torchgen/packaged/ATen/native/native_functions.yaml
                   MAIN_DEPENDENCY ${CMAKE_SOURCE_DIR}/scripts/gen_op.py
                   DEPENDS
                     ${CMAKE_SOURCE_DIR}/scripts/hpu_op.yaml
                     ${CMAKE_SOURCE_DIR}/habana_kernels/wrap_kernels_declarations.h
                     ${PT_VER_DIR}/habana_kernels_ver/wrap_kernels_declarations.h
                     ${TORCH_INSTALL_PREFIX}/include/ATen/RegistrationDeclarations.h
                     ${TORCH_INSTALL_PREFIX}/include/ATen/RedispatchFunctions.h
)

# Create symlink to latest build from tests dir
add_custom_command(TARGET habana_pytorch_plugin POST_BUILD COMMAND
  ${CMAKE_COMMAND} -E create_symlink $<TARGET_FILE:habana_pytorch_plugin>
  ${CMAKE_SOURCE_DIR}/tests/$<TARGET_FILE_NAME:habana_pytorch_plugin>
  COMMAND ${CMAKE_COMMAND} -E create_symlink $<TARGET_FILE:habana_pytorch_plugin>
  $ENV{BUILD_ROOT_LATEST}/$<TARGET_FILE_NAME:habana_pytorch_plugin>
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

# additional args will be treated as dependencies - relative to the PACKAGE location
function(build_or_install_wheel PACKAGE)
  get_filename_component(name ${PACKAGE} NAME)
  set(OUTPUT "${CMAKE_BINARY_DIR}/pkgs/${name}")

  set(PIP ${PYTHON_EXECUTABLE} -m pip)
  if (INSTALL_PKGS)
    set(PIPARGS install -U ${PACKAGE}/)
  else()
    set(PIPARGS wheel --no-deps --wheel-dir=${CMAKE_BINARY_DIR}/pkgs ${PACKAGE})
  endif()

  list(TRANSFORM ARGN PREPEND "${PACKAGE}/" OUTPUT_VARIABLE DEPS)
  add_custom_command(OUTPUT ${OUTPUT}
                    COMMAND ${PIP} ${PIPARGS}
                    COMMAND ${CMAKE_COMMAND} -E touch ${OUTPUT}
                    DEPENDS habana_pytorch_plugin
                    DEPENDS ${DEPS})
  add_custom_target(${name} ALL DEPENDS ${OUTPUT})
endfunction()

if (BUILD_PKGS)
  build_or_install_wheel(${PROJECT_SOURCE_DIR}/pytorch_helpers/dataloader/habana_dataloader
    __init__.py
    habana_dataloader/__init__.py
    habana_dataloader/aeon_config.py
    habana_dataloader/aeon_manifest.py
    habana_dataloader/aeon_ssd_configurator.py
    habana_dataloader/aeon_transformers.py
    habana_dataloader/habana_dataset.py
    include/blocking_queue.h
    include/data_loader.h
    include/pybind11_json.hpp
    LICENSE.txt
    main.cpp
    setup.py
  )
  build_or_install_wheel(${PROJECT_SOURCE_DIR}/python_packages
    habana_frameworks/torch/__init__.py
    habana_frameworks/torch/activity_profiler/__init__.py
    habana_frameworks/torch/activity_profiler/activity_profiler.py
    habana_frameworks/torch/activity_profiler/csrc/bindings.cpp
    habana_frameworks/torch/core/__init__.py
    habana_frameworks/torch/core/bindings.cpp
    habana_frameworks/torch/core/hccl/__init__.py
    habana_frameworks/torch/core/step_closure.py
    habana_frameworks/torch/core/weight_sharing.py
    habana_frameworks/torch/distributed/__init__.py
    habana_frameworks/torch/distributed/hccl/__init__.py
    habana_frameworks/torch/distributed/hccl/process_group_lazy_hccl.cpp
    habana_frameworks/torch/distributed/hccl/process_group_lazy_hccl.hpp
    habana_frameworks/torch/distributed/hccl/ProcessGroupHCCL.cpp
    habana_frameworks/torch/distributed/hccl/ProcessGroupHCCL.hpp
    habana_frameworks/torch/distributed/hccl/reducer_timer.cpp
    habana_frameworks/torch/hpex/__init__.py
    habana_frameworks/torch/hpex/csrc/bindings.cpp
    habana_frameworks/torch/hpex/csrc/bindings.h
    habana_frameworks/torch/hpex/csrc/embedding_bag_preproc_main.cpp
    habana_frameworks/torch/hpex/csrc/embedding_bag_preproc.cpp
    habana_frameworks/torch/hpex/csrc/radix_sort.h
    habana_frameworks/torch/hpex/experimental/fp8_autocast/__init__.py
    habana_frameworks/torch/hpex/experimental/fp8_autocast/Fp8Autocast.py
    habana_frameworks/torch/hpex/experimental/fp8_autocast/LinearFp8.py
    habana_frameworks/torch/hpex/experimental/fp8_autocast/MatmulFp8.py
    habana_frameworks/torch/hpex/hmp/__init__.py
    habana_frameworks/torch/hpex/hmp/config.py
    habana_frameworks/torch/hpex/hmp/hmp.py
    habana_frameworks/torch/hpex/hmp/ops_bf16.txt
    habana_frameworks/torch/hpex/hmp/ops_dict.py
    habana_frameworks/torch/hpex/hmp/ops_fp16.txt
    habana_frameworks/torch/hpex/hmp/ops_fp32_for_bf16.txt
    habana_frameworks/torch/hpex/hmp/ops_fp32_for_fp16.txt
    habana_frameworks/torch/hpex/hmp/ops_multi_inputs.txt
    habana_frameworks/torch/hpex/hmp/utils.py
    habana_frameworks/torch/hpex/kernels/__init__.py
    habana_frameworks/torch/hpex/kernels/CastToFp8.py
    habana_frameworks/torch/hpex/kernels/CustomNms.py
    habana_frameworks/torch/hpex/kernels/CustomRoiAlign.py
    habana_frameworks/torch/hpex/kernels/EmbeddingBag.py
    habana_frameworks/torch/hpex/kernels/EmbeddingBagPreproc.py
    habana_frameworks/torch/hpex/movingavrg/__init__.py
    habana_frameworks/torch/hpex/movingavrg/FusedEMA.py
    habana_frameworks/torch/hpex/normalization/__init__.py
    habana_frameworks/torch/hpex/normalization/FusedClipNorm.py
    habana_frameworks/torch/hpex/optimizers/__init__.py
    habana_frameworks/torch/hpex/optimizers/distributed/__init__.py
    habana_frameworks/torch/hpex/optimizers/distributed/FusedAdamW.py
    habana_frameworks/torch/hpex/optimizers/FusedAdagrad.py
    habana_frameworks/torch/hpex/optimizers/FusedAdamW.py
    habana_frameworks/torch/hpex/optimizers/FusedLamb.py
    habana_frameworks/torch/hpex/optimizers/FusedLars.py
    habana_frameworks/torch/hpex/optimizers/FusedResourceApplyMomentum.py
    habana_frameworks/torch/hpex/optimizers/FusedSGD.py
    habana_frameworks/torch/hpex/optimizers/SparseAdagrad.py
    habana_frameworks/torch/hpex/optimizers/SparseSGD.py
    habana_frameworks/torch/hpu/__init__.py
    habana_frameworks/torch/hpu/_utils.py
    habana_frameworks/torch/hpu/csrc/bindings.cpp
    habana_frameworks/torch/hpu/events.py
    habana_frameworks/torch/hpu/graphs.py
    habana_frameworks/torch/hpu/memory.py
    habana_frameworks/torch/hpu/random.py
    habana_frameworks/torch/hpu/streams.py
    habana_frameworks/torch/profiler/__init__.py
    habana_frameworks/torch/profiler/ProfilerApi.py
    habana_frameworks/torch/utils/__init__.py
    habana_frameworks/torch/utils/debug/__init__.py
    habana_frameworks/torch/utils/debug/csrc/bindings.cpp
    habana_frameworks/torch/utils/debug/README.txt
    habana_frameworks/torch/utils/distributed_utils.py
    habana_frameworks/torch/utils/event_dispatcher/__init__.py
    habana_frameworks/torch/utils/event_dispatcher/csrc/bindings.cpp
    habana_frameworks/torch/utils/experimental/__init__.py
    habana_frameworks/torch/utils/experimental/csrc/bindings.cpp
    habana_frameworks/torch/utils/experimental/distributed_emulation.py
    habana_frameworks/torch/utils/lib_utils.py
    habana_frameworks/torch/utils/library_loader.py
    habana_frameworks/torch/utils/profiler/__init__.py
    habana_frameworks/torch/utils/profiler/csrc/bindings.cpp
    LICENSE.txt
    setup.py
    torch_hpu/__init__.py
  )
endif()
