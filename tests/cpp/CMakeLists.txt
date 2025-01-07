###############################################################################
#
#  Copyright (c) 2021-2025 Intel Corporation
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
###############################################################################

set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

set(lazy_tests_src
  lazy_graph_passes/test_replace_inplace_ops.cpp
  lazy_add_composite_tests.cpp
  lazy_basic_kernel_tests.cpp
  lazy_binary_composite_kernel_tests.cpp
  lazy_binary_inplace_kernel_tests.cpp
  lazy_binary_kernel_tests.cpp
  lazy_bitwise_kernel_tests.cpp
  lazy_cache_tests.cpp
  lazy_compare_kernel_tests.cpp
  lazy_control_edge_tests.cpp
  lazy_conv_kernel_tests.cpp
  lazy_custom_kernel_tests.cpp
  lazy_custom_op_kernel_test.cpp
  lazy_dynamic_compute_output_shapes_tests.cpp
  lazy_dynamic_fallback_tests.cpp
  lazy_dynamic_launch_fallback_test.cpp
  lazy_dynamic_shapes_bucket_refine.cpp
  lazy_dynamic_shapes_serialize_tests.cpp
  lazy_dynamic_shapes_tests.cpp
  lazy_dynamic_shapes_tests2.cpp
  lazy_eager_tests.cpp
  lazy_embedding_kernel_tests.cpp
  lazy_execution_test.cpp
  lazy_fill_kernel_tests.cpp
  lazy_forward_running_hash_tests.cpp
  lazy_index_kernel_tests.cpp
  lazy_jit_graph_tests.cpp
  lazy_linear_kernel_tests.cpp
  lazy_loss_kernel_tests.cpp
  lazy_mask_scalar_kernel_test.cpp
  lazy_misc_tests.cpp
  lazy_norm_kernel_tests.cpp
  lazy_pool_kernel_tests.cpp
  lazy_random_gen_kernel_tests.cpp
  lazy_reduction_kernel_tests.cpp
  lazy_softmax_kernel_tests.cpp
  lazy_special_kernel_tests.cpp
  lazy_stride_tests.cpp
  lazy_tensor_api_tests.cpp
  lazy_tensor_shape_kernel_tests.cpp
  lazy_unary_kernel_tests.cpp
  lazy_upsample_kernel_tests.cpp
  lazy_where_kernel_tests.cpp
  post_order_tests.cpp
  sbs_tests.cpp
  dry_run_test.cpp
)

set(hpu_ops_tests_src
  hpu_ops/_assert_async.cpp
  hpu_ops/_masked_scale.cpp
  hpu_ops/adaptive_avg_pool2d.cpp
  hpu_ops/addbmm.cpp
  hpu_ops/addmm.cpp
  hpu_ops/addmv.cpp
  hpu_ops/addr.cpp
  hpu_ops/all.cpp
  hpu_ops/amax_amin.cpp
  hpu_ops/any.cpp
  hpu_ops/arange.cpp
  hpu_ops/argmin_max.cpp
  hpu_ops/avgpool_2d.cpp
  hpu_ops/batch_norm_tests.cpp
  hpu_ops/binary_op.cpp
  hpu_ops/binarycrossentropy_with_logits.cpp
  hpu_ops/binarycrossentropy.cpp
  hpu_ops/bitwise_op.cpp
  hpu_ops/bitwise_shift.cpp
  hpu_ops/bmm.cpp
  hpu_ops/cat.cpp
  hpu_ops/channelshuffle.cpp
  hpu_ops/compare.cpp
  hpu_ops/compute_shapes_hpu_ops.cpp
  hpu_ops/copysign.cpp
  hpu_ops/cosine.cpp
  hpu_ops/count_nonzero.cpp
  hpu_ops/cross.cpp
  hpu_ops/cumprod.cpp
  hpu_ops/cumsum.cpp
  hpu_ops/div_out.cpp
  hpu_ops/div_roundoption.cpp
  hpu_ops/div.cpp
  hpu_ops/dot.cpp
  hpu_ops/_efficientzerotensor.cpp
  hpu_ops/elu.cpp
  hpu_ops/embedding.cpp
  hpu_ops/entr.cpp
  hpu_ops/eq.cpp
  hpu_ops/erfc.cpp
  hpu_ops/special_erfcx.cpp
  hpu_ops/erfinv.cpp
  hpu_ops/exponential.cpp
  hpu_ops/eye.cpp
  hpu_ops/fill.cpp
  hpu_ops/flatten_dense_tensors.cpp
  hpu_ops/flip.cpp
  hpu_ops/floor_divide.cpp
  hpu_ops/fmod.cpp
  hpu_ops/foreach_binary.cpp
  hpu_ops/foreach.cpp
  hpu_ops/frac.cpp
  hpu_ops/frexp.cpp
  hpu_ops/full.cpp
  hpu_ops/fused_dropout.cpp
  hpu_ops/gather.cpp
  hpu_ops/ge.cpp
  hpu_ops/gelu.cpp
  hpu_ops/geometric.cpp
  hpu_ops/glu.cpp
  hpu_ops/gt.cpp
  hpu_ops/hardswish.cpp
  hpu_ops/hardtanh.cpp
  hpu_ops/heaviside.cpp
  hpu_ops/huber_loss.cpp
  hpu_ops/hypot.cpp
  hpu_ops/index.cpp
  hpu_ops/index_select.cpp
  hpu_ops/kthvalue.cpp
  hpu_ops/l1_loss.cpp
  hpu_ops/le.cpp
  hpu_ops/leakyrelu.cpp
  hpu_ops/lerp.cpp
  hpu_ops/linear.cpp
  hpu_ops/linspace.cpp
  hpu_ops/log_softmax.cpp
  hpu_ops/log10.cpp
  hpu_ops/log1p.cpp
  hpu_ops/logaddexp.cpp
  hpu_ops/logaddexp2.cpp
  hpu_ops/logcumsumexp.cpp
  hpu_ops/logical_op.cpp
  hpu_ops/logit.cpp
  hpu_ops/logsigmoid.cpp
  hpu_ops/logsoftmax.cpp
  hpu_ops/logspace.cpp
  hpu_ops/logsumexp.cpp
  hpu_ops/lt.cpp
  hpu_ops/masked_scatter.cpp
  hpu_ops/masked_select.cpp
  hpu_ops/matmul.cpp
  hpu_ops/max_dim.cpp
  hpu_ops/max.cpp
  hpu_ops/maxpool2d.cpp
  hpu_ops/maxpool3d.cpp
  hpu_ops/mean.cpp
  hpu_ops/median.cpp
  hpu_ops/min.cpp
  hpu_ops/mish.cpp
  hpu_ops/mm.cpp
  hpu_ops/mse_loss.cpp
  hpu_ops/mul.cpp
  hpu_ops/nan_to_num.cpp
  hpu_ops/nansum.cpp
  hpu_ops/native_group_norm_tests.cpp
  hpu_ops/ne.cpp
  hpu_ops/new_zeros.cpp
  hpu_ops/nextafter.cpp
  hpu_ops/nll_loss.cpp
  hpu_ops/norm.cpp
  hpu_ops/one_hot.cpp
  hpu_ops/op_logger.cpp
  hpu_ops/poisson.cpp
  hpu_ops/pow.cpp
  hpu_ops/prod.cpp
  hpu_ops/random.cpp
  hpu_ops/reflection_pad.cpp
  hpu_ops/relu.cpp
  hpu_ops/remainder.cpp
  hpu_ops/replication_pad.cpp
  hpu_ops/roll.cpp
  hpu_ops/round.cpp
  hpu_ops/rrelu_with_noise.cpp
  hpu_ops/scalar_tensor.cpp
  hpu_ops/scatter.cpp
  hpu_ops/scatter_add.cpp
  hpu_ops/scatter_reduce.cpp
  hpu_ops/searchsorted.cpp
  hpu_ops/select.cpp
  hpu_ops/sgn_sign.cpp
  hpu_ops/shrink.cpp
  hpu_ops/sigmoid.cpp
  hpu_ops/signbit.cpp
  hpu_ops/silu.cpp
  hpu_ops/sinc.cpp
  hpu_ops/smooth_l1_loss.cpp
  hpu_ops/softmax.cpp
  hpu_ops/softplus.cpp
  hpu_ops/sort.cpp
  hpu_ops/square.cpp
  hpu_ops/squeeze.cpp
  hpu_ops/sum.cpp
  hpu_ops/take.cpp
  hpu_ops/tanh.cpp
  hpu_ops/tensor_shape.cpp
  hpu_ops/threshold.cpp
  hpu_ops/topk.cpp
  hpu_ops/trace.cpp
  hpu_ops/type_promotion.cpp
  hpu_ops/unary_op.cpp
  hpu_ops/upsample_bicubic2d.cpp
  hpu_ops/upsample_bilinear2d.cpp
  hpu_ops/upsample_linear1d.cpp
  hpu_ops/upsample_nearest1d.cpp
  hpu_ops/upsample_nearest_exact1d.cpp
  hpu_ops/upsample_nearest2d.cpp
  hpu_ops/upsample_nearest3d.cpp
  hpu_ops/util.cpp
  hpu_ops/var_std.cpp
  hpu_ops/vdot.cpp
  hpu_ops/vec_norm.cpp
  hpu_ops/view.cpp
  hpu_ops/weight_norm.cpp
  hpu_ops/where.cpp
  hpu_ops/xlogy.cpp
  hpu_ops/zero.cpp
)

list(APPEND misc_tests_src
  acc_thread_pool_tests.cpp
  common_functions_custom_kernel_tests.cpp
  common_functions_helpers.cpp
  common_functions_norm_kernel_tests.cpp
  compute_shapes.cpp
  debug_utils_tests.cpp
  ds_bucket_tests.cpp
  ds_cache_tests.cpp
  eager_custom_kernel_tests.cpp
  env_flags_test.cpp
  fallback_tests.cpp
  gen_ops.cpp
  graph_optimize_passes.cpp
  habana_lazy_test_infra.cpp
  inference_pass_tests.cpp
  linear_test.cpp
  $<$<TARGET_EXISTS:npu::specs_embedded>:mem_hlml_tests.cpp>
  main.cpp
  matmul_bwd.cpp
  misc_utils_tests.cpp
  nms_tests.cpp
  serialization/habana_serialization_tests.cpp
  sif_tests.cpp
  synapse_helpers_memory_test.cpp
  synapse_helpers_tensor_test.cpp
  synapse_helpers_tensor_info_test.cpp
  synapse_helpers_graph_test.cpp
  tensor_usage.cpp
  type_promotion_tests.cpp
  user_stream_tests.cpp
  utils/check_device_type.cpp
  utils/device_type_util.cpp
  utils/dtype_supported_on_device.cpp
  utils/hint_tolerance_values.cpp
  view_utils_tests.cpp
  common_tests.cpp
)

list(APPEND eager_tests_src
  eager_kernel_tests.cpp
  eager_pipeline_tests.cpp
  eager_view_tests.cpp
  scalar_dtype_range_tests.cpp
  shape_agnostic_tests.cpp
)

add_habana_executable(
  test_pt_integration
  ${lazy_tests_src}
  ${hpu_ops_tests_src}
  ${misc_tests_src}
)

target_compile_definitions(test_pt_integration PRIVATE LAZY_TESTS)
set_target_properties(test_pt_integration PROPERTIES COMPILE_OPTIONS "-Wno-deprecated-declarations")

# Remove tests unrelated or not working with PT2
list(REMOVE_ITEM misc_tests_src
  graph_optimize_passes.cpp

  # Doesn't compile
  linear_2d_bwd.cpp
  matmul_bwd.cpp
  linear_test.cpp
  nms_tests.cpp
  sif_tests.cpp
  common_functions_custom_kernel_tests.cpp
  eager_custom_kernel_tests.cpp
)

add_habana_executable(
  test_pt2_integration
  ${hpu_ops_tests_src}
  ${misc_tests_src}
  ${eager_tests_src})
target_compile_definitions(test_pt2_integration PRIVATE EAGER_TESTS)
set_target_properties(test_pt2_integration PROPERTIES COMPILE_OPTIONS "-Wno-deprecated-declarations")

configure_file(pattern.json ${CMAKE_BINARY_DIR}/pattern.json COPYONLY)
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--no-as-needed")
target_link_libraries(
  test_pt_integration
  PRIVATE
  Python::Python
  torch
  habana_pytorch_backend
  habana_pytorch_plugin
  gtest
  gtest_main
  util
  pthread
  stdc++
  stdc++fs
  nlohmann_json::nlohmann_json
  absl::strings
  dl
  npu::hllogger)

add_custom_command(
  TARGET test_pt_integration
  POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E create_symlink $<TARGET_FILE:test_pt_integration>
  ${CMAKE_SOURCE_DIR}/tests/$<TARGET_FILE_NAME:test_pt_integration>)

target_link_libraries(
  test_pt2_integration
  PRIVATE
  Python::Python
  torch
  habana_pytorch_backend
  habana_pytorch2_plugin
  gtest
  gtest_main
  util
  pthread
  stdc++
  stdc++fs
  nlohmann_json::nlohmann_json
  absl::strings
  dl
  npu::hllogger)
add_custom_command(
  TARGET test_pt2_integration
  POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E create_symlink $<TARGET_FILE:test_pt2_integration>
  ${CMAKE_SOURCE_DIR}/tests/$<TARGET_FILE_NAME:test_pt2_integration>)
