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

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)

add_custom_command(OUTPUT
                     ${CMAKE_BINARY_DIR}/generated/lazy/hpu_op0.cpp
                     ${CMAKE_BINARY_DIR}/generated/lazy/hpu_op1.cpp
                     ${CMAKE_BINARY_DIR}/generated/lazy/hpu_op2.cpp
                     ${CMAKE_BINARY_DIR}/generated/lazy/hpu_op3.cpp
                     ${CMAKE_BINARY_DIR}/generated/lazy/hpu_op4.cpp
                     ${CMAKE_BINARY_DIR}/generated/lazy/hpu_op5.cpp
                     ${CMAKE_BINARY_DIR}/generated/lazy/hpu_op6.cpp
                     ${CMAKE_BINARY_DIR}/generated/lazy/hpu_op7.cpp
                     ${CMAKE_BINARY_DIR}/generated/lazy/hpu_op8.cpp
                     ${CMAKE_BINARY_DIR}/generated/lazy/hpu_op9.cpp
                     ${CMAKE_BINARY_DIR}/generated/lazy/hpu_autocast_ops.cpp
                     ${CMAKE_BINARY_DIR}/generated/lazy/wrap_kernels_registrations.cpp
                     ${CMAKE_BINARY_DIR}/generated/eager/hpu_op0.cpp
                     ${CMAKE_BINARY_DIR}/generated/eager/hpu_op1.cpp
                     ${CMAKE_BINARY_DIR}/generated/eager/hpu_op2.cpp
                     ${CMAKE_BINARY_DIR}/generated/eager/hpu_op3.cpp
                     ${CMAKE_BINARY_DIR}/generated/eager/hpu_op4.cpp
                     ${CMAKE_BINARY_DIR}/generated/eager/hpu_op5.cpp
                     ${CMAKE_BINARY_DIR}/generated/eager/hpu_op6.cpp
                     ${CMAKE_BINARY_DIR}/generated/eager/hpu_op7.cpp
                     ${CMAKE_BINARY_DIR}/generated/eager/hpu_op8.cpp
                     ${CMAKE_BINARY_DIR}/generated/eager/hpu_op9.cpp
                     ${CMAKE_BINARY_DIR}/generated/eager/hpu_autocast_ops.cpp
                     ${CMAKE_BINARY_DIR}/generated/eager/wrap_kernels_registrations.cpp
                     ${CMAKE_BINARY_DIR}/generated/backend/hpu_op0.cpp
                     ${CMAKE_BINARY_DIR}/generated/backend/hpu_op1.cpp
                     ${CMAKE_BINARY_DIR}/generated/backend/hpu_op2.cpp
                     ${CMAKE_BINARY_DIR}/generated/backend/hpu_op3.cpp
                     ${CMAKE_BINARY_DIR}/generated/backend/hpu_op4.cpp
                     ${CMAKE_BINARY_DIR}/generated/backend/hpu_op5.cpp
                     ${CMAKE_BINARY_DIR}/generated/backend/hpu_op6.cpp
                     ${CMAKE_BINARY_DIR}/generated/backend/hpu_op7.cpp
                     ${CMAKE_BINARY_DIR}/generated/backend/hpu_op8.cpp
                     ${CMAKE_BINARY_DIR}/generated/backend/hpu_op9.cpp
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


add_library(habana_pytorch_backend_ops OBJECT
          backend/_assert_async_gen.cpp
          backend/_masked_scale_gen.cpp
          backend/adaptive_avg_pool2d_gen.cpp
          backend/add_composite_gen.cpp
          backend/addmm_gen.cpp
          backend/addmv_gen.cpp
          backend/addr_gen.cpp
          backend/all_gen.cpp
          backend/amax_amin_gen.cpp
          backend/any_gen.cpp
          backend/arange_gen.cpp
          backend/argmin_max_gen.cpp
          backend/avg_pool2d_gen.cpp
          backend/baddbmm_gen.cpp
          backend/bernoulli_gen.cpp
          backend/binary_gen.cpp
          backend/binarycrossentropy_gen.cpp
          backend/bitwise_gen.cpp
          backend/bitwise_shift_gen.cpp
          backend/bmm_gen.cpp
          backend/cat_gen.cpp
          backend/channelshuffle_gen.cpp
          backend/clamp_gen.cpp
          backend/compare_gen.cpp
          backend/cumsum_gen.cpp
          backend/diag_gen.cpp
          backend/div_mod_util.cpp
          backend/div_round_gen.cpp
          backend/dot_gen.cpp
          backend/elu_backward_gen.cpp
          backend/equal_gen.cpp
          backend/erfc_gen.cpp
          backend/exponential_gen.cpp
          backend/eye_gen.cpp
          backend/fbgemm.cpp
          backend/fill_gen.cpp
          backend/flip_gen.cpp
          backend/floor_divide_gen.cpp
          backend/foreach_gen.cpp
          backend/frac_gen.cpp
          backend/frexp_gen.cpp
          backend/gather_gen.cpp
          backend/gelu_gen.cpp
          backend/geometric_gen.cpp
          backend/glu_gen.cpp
          backend/grid_sampler_gen.cpp
          backend/hardsigmoid_gen.cpp
          backend/hardtanh_gen.cpp
          backend/hpu_op_helper.cpp
          backend/huber_loss_gen.cpp
          backend/index_gen.cpp
          backend/index_select_gen.cpp
          backend/isfinite_isinf_isnan_gen.cpp
          backend/isposneginf_gen.cpp
          backend/kthvalue_gen.cpp
          backend/lazy_cast.cpp
          backend/fp8_ops.cpp
          backend/leakyrelu_gen.cpp
          backend/lerp_gen.cpp
          backend/linalg_cross_gen.cpp
          backend/linspace_gen.cpp
          backend/log10_gen.cpp
          backend/log_normal_gen.cpp
          backend/logical_gen.cpp
          backend/logit_gen.cpp
          backend/logsigmoid_gen.cpp
          backend/logsoftmax_gen.cpp
          backend/logspace_gen.cpp
          backend/max_min_gen.cpp
          backend/maxpool_gen.cpp
          backend/median_gen.cpp
          backend/median_slice_util.cpp
          backend/mish_gen.cpp
          backend/mm_gen.cpp
          backend/mseloss_gen.cpp
          backend/multinomial_gen.cpp
          backend/mv_gen.cpp
          backend/nansum_gen.cpp
          backend/nantonum_gen.cpp
          backend/ne_gen.cpp
          backend/nll_loss_gen.cpp
          backend/norm_gen.cpp
          backend/normal_gen.cpp
          backend/op_backend.cpp
          backend/poission_gen.cpp
          backend/pow_gen.cpp
          backend/prelu_gen.cpp
          backend/random_gen.cpp
          backend/reduce_std_var_gen.cpp
          backend/reduction_template.cpp
          common/reduction_template.cpp
          backend/reflection_pad_gen.cpp
          backend/replication_pad_bwd_gen.cpp
          backend/replication_pad_gen.cpp
          backend/resize_gen.cpp
          backend/roll_gen.cpp
          backend/round_gen.cpp
          backend/rrelu_with_noise_gen.cpp
          backend/scatter_gen.cpp
          backend/shrink_gen.cpp
          backend/smooth_l1_loss_gen.cpp
          backend/softmax_gen.cpp
          backend/softplus_gen.cpp
          backend/take_gen.cpp
          backend/threshold_gen.cpp
          backend/topk_gen.cpp
          backend/topk_util.cpp
          backend/trace_gen.cpp
          backend/triangular_gen.cpp
          backend/upsample_gen.cpp
          backend/vdot_gen.cpp
          backend/where_gen.cpp
          backend/xlogy_gen.cpp
          ${CMAKE_BINARY_DIR}/generated/backend/hpu_op0.cpp
          ${CMAKE_BINARY_DIR}/generated/backend/hpu_op1.cpp
          ${CMAKE_BINARY_DIR}/generated/backend/hpu_op2.cpp
          ${CMAKE_BINARY_DIR}/generated/backend/hpu_op3.cpp
          ${CMAKE_BINARY_DIR}/generated/backend/hpu_op4.cpp
          ${CMAKE_BINARY_DIR}/generated/backend/hpu_op5.cpp
          ${CMAKE_BINARY_DIR}/generated/backend/hpu_op6.cpp
          ${CMAKE_BINARY_DIR}/generated/backend/hpu_op7.cpp
          ${CMAKE_BINARY_DIR}/generated/backend/hpu_op8.cpp
          ${CMAKE_BINARY_DIR}/generated/backend/hpu_op9.cpp
          )
target_include_directories(habana_pytorch_backend_ops
  PUBLIC
    ${PROJECT_SOURCE_DIR})
target_include_directories(habana_pytorch_backend_ops SYSTEM
  PUBLIC
    $<TARGET_PROPERTY:Synapse,INTERFACE_INCLUDE_DIRECTORIES>
    $<TARGET_PROPERTY:tpc_kernels,INTERFACE_INCLUDE_DIRECTORIES>
    $<TARGET_PROPERTY:hcl,INTERFACE_INCLUDE_DIRECTORIES>)
target_compile_definitions(habana_pytorch_backend_ops PRIVATE MAX_DEVICES_PER_BOX=8)
target_compile_definitions(habana_pytorch_backend_ops PUBLIC GENERIC_HELPERS)
set_target_properties(habana_pytorch_backend_ops PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF)
target_include_directories(habana_pytorch_backend_ops PRIVATE ${PROJECT_SOURCE_DIR}/pytorch_helpers)
target_include_directories(habana_pytorch_backend_ops PRIVATE ${CMAKE_BINARY_DIR})
target_link_libraries(habana_pytorch_backend_ops
  PRIVATE
    habana_serialization
    torch
    nlohmann_json::nlohmann_json
    absl::strings
    absl::str_format
    npu::spdlog
)

add_library(habana_pytorch_frontend_ops OBJECT
          cpu_fallback.cpp
          supported_dtypes.cpp
          lazy/add_composite_gen.cpp
          lazy/arange_gen.cpp
          lazy/bernoulli_gen.cpp
          lazy/binarycrossentropy_gen.cpp
          lazy/bitwise_gen.cpp
          lazy/bitwise_shift_gen.cpp
          lazy/clamp_gen.cpp
          lazy/compare_gen.cpp
          lazy/cumsum_gen.cpp
          lazy/div_round_gen.cpp
          lazy/embedding_gen.cpp
          lazy/exponential_gen.cpp
          lazy/fill_gen.cpp
          lazy/foreach_gen.cpp
          lazy/frexp_gen.cpp
          lazy/geometric_gen.cpp
          lazy/index_gen.cpp
          lazy/index_select_gen.cpp
          lazy/kthvalue_gen.cpp
          lazy/logical_gen.cpp
          lazy/max_min_gen.cpp
          lazy/maxpool_gen.cpp
          lazy/multinomial_gen.cpp
          lazy/nansum_gen.cpp
          lazy/norm_gen.cpp
          lazy/pow_gen.cpp
          lazy/random_gen.cpp
          lazy/reduction_template.cpp
          lazy/remainder_gen.cpp
          lazy/rrelu_with_noise_gen.cpp
          lazy/rsub_gen.cpp
          lazy/threshold_gen.cpp
          lazy/topk_gen.cpp
          lazy/topk_util.cpp
          lazy/trace_gen.cpp
          lazy/unique2_gen.cpp
          lazy/unique_gen.cpp
          lazy/upsample_gen.cpp
          lazy/where_gen.cpp
          lazy/xlogy_gen.cpp
          op_logger.cpp
          op_validator.cpp
          # autogenerated files:
          ${CMAKE_BINARY_DIR}/generated/lazy/hpu_op0.cpp
          ${CMAKE_BINARY_DIR}/generated/lazy/hpu_op1.cpp
          ${CMAKE_BINARY_DIR}/generated/lazy/hpu_op2.cpp
          ${CMAKE_BINARY_DIR}/generated/lazy/hpu_op3.cpp
          ${CMAKE_BINARY_DIR}/generated/lazy/hpu_op4.cpp
          ${CMAKE_BINARY_DIR}/generated/lazy/hpu_op5.cpp
          ${CMAKE_BINARY_DIR}/generated/lazy/hpu_op6.cpp
          ${CMAKE_BINARY_DIR}/generated/lazy/hpu_op7.cpp
          ${CMAKE_BINARY_DIR}/generated/lazy/hpu_op8.cpp
          ${CMAKE_BINARY_DIR}/generated/lazy/hpu_op9.cpp
          ${CMAKE_BINARY_DIR}/generated/lazy/hpu_autocast_ops.cpp
          ${CMAKE_BINARY_DIR}/generated/lazy/wrap_kernels_registrations.cpp)

target_include_directories(habana_pytorch_frontend_ops
  PUBLIC
    ${PROJECT_SOURCE_DIR})
target_include_directories(habana_pytorch_frontend_ops SYSTEM
  PUBLIC
    $<TARGET_PROPERTY:Synapse,INTERFACE_INCLUDE_DIRECTORIES>
    $<TARGET_PROPERTY:SynapseUtils,INTERFACE_INCLUDE_DIRECTORIES>
    $<TARGET_PROPERTY:tpc_kernels,INTERFACE_INCLUDE_DIRECTORIES>
    $<TARGET_PROPERTY:hcl,INTERFACE_INCLUDE_DIRECTORIES>)

target_compile_definitions(habana_pytorch_frontend_ops PRIVATE MAX_DEVICES_PER_BOX=8)
target_compile_definitions(habana_pytorch_frontend_ops PUBLIC GENERIC_HELPERS)
set_target_properties(habana_pytorch_frontend_ops PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF)
target_include_directories(habana_pytorch_frontend_ops PRIVATE ${PROJECT_SOURCE_DIR}/pytorch_helpers)
target_include_directories(habana_pytorch_frontend_ops PRIVATE ${CMAKE_BINARY_DIR})
target_link_libraries(habana_pytorch_frontend_ops
  PRIVATE
  habana_serialization
  pytorch_synapse_utils_shim
    torch
    nlohmann_json::nlohmann_json
    absl::strings
    absl::str_format
    npu::spdlog
)
