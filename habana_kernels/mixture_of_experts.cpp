/**
 * Copyright (c) 2021-2025 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "hpu_ops/mixture_of_experts.h"
#include <habana_kernels/mixture_of_experts.h>
#include "common/dump_args.h"
#include "generated/backend/mixture_of_experts_bwd.h"
#include "generated/backend/mixture_of_experts_fwd.h"
#include "generated/backend/mixture_of_experts_recomp_bwd.h"
#include "generated/backend/mixture_of_experts_recomp_fwd.h"
#include "habana_helpers/logging.h"
#include "habana_kernels/lazy_kernels.h"
#include "habana_lazy/hlexec.h"
#include "habana_lazy/hpu_stage_submission.h"
#include "habana_lazy/lazy_executor.h"
#include "hpu_ops/common/mixture_of_experts.h"
#include "hpu_ops/op_logger.h"

namespace habana {
using namespace habana_lazy;

static std::vector<std::vector<int64_t>> get_bwd_output_shapes(
    const at::Tensor& grad_tokens_in,
    const std::vector<int64_t> router_weights_size,
    std::vector<at::TensorList> weights_lists,
    const size_t amax_outputs) {
  const int64_t num_weights = weights_lists[0].size();

  std::vector<std::vector<int64_t>> output_shapes;
  output_shapes.reserve(2 + weights_lists.size() * num_weights + amax_outputs);
  output_shapes.push_back(grad_tokens_in.sizes().vec());
  output_shapes.push_back(router_weights_size);

  for (size_t i = 0; i < amax_outputs; ++i) {
    output_shapes.push_back({num_weights});
  }

  for (auto weights_list : weights_lists) {
    for (const at::Tensor& weights : weights_list) {
      output_shapes.push_back(weights.sizes().vec());
    }
  }

  return output_shapes;
}

std::vector<at::Tensor> mixture_of_experts_fwd(
    const at::Tensor& hidden_states,
    const at::Tensor& expert_routing_table,
    const at::Tensor& router_weights,
    const at::TensorList w1,
    const at::TensorList w2,
    const at::TensorList w3,
    const bool permuted_weights,
    const std::string_view activation,
    const int64_t experts_min,
    const int64_t experts_max,
    const int64_t chunk_size,
    const int64_t total_experts) {
  PT_LAZY_OP_TRACE;
  PT_OP_INFO(
      "mixture_of_experts_fwd :",
      DUMP_12ARGS(
          hidden_states,
          expert_routing_table,
          router_weights,
          w1,
          w2,
          w3,
          permuted_weights,
          activation,
          experts_min,
          experts_max,
          chunk_size,
          total_experts));

  exec::OptPassCfg::GetInstance()->BkupAndDisableAndAllOptPass();

  LazyOp<std::vector<at::Tensor>> op{
      "hpu::mixture_of_experts_fwd",
      {hidden_states,
       expert_routing_table,
       router_weights,
       w1,
       w2,
       w3,
       permuted_weights,
       activation,
       experts_min,
       experts_max,
       chunk_size,
       total_experts},
      MixtureOfExpertsFwdShapes,
      0};

  op.viewUpdateInputs();
  op.SetOutputMetaFn(MixtureOfExpertsFwdMeta);

  RUN_MAYBE_WITH_ACC_THREAD(mixture_of_experts, op)
}
at::Tensor mixture_of_experts_recomp_fwd(
    const at::Tensor& hidden_states,
    const at::Tensor& expert_routing_table,
    const at::Tensor& router_weights,
    const at::TensorList w1,
    const at::TensorList w2,
    const at::TensorList w3,
    const bool permuted_weights,
    const std::string_view activation,
    const int64_t experts_min,
    const int64_t experts_max,
    const int64_t chunk_size,
    const int64_t total_experts) {
  PT_LAZY_OP_TRACE;
  PT_OP_INFO(
      "mixture_of_experts_recomp_fwd :",
      DUMP_12ARGS(
          hidden_states,
          expert_routing_table,
          router_weights,
          w1,
          w2,
          w3,
          permuted_weights,
          activation,
          experts_min,
          experts_max,
          chunk_size,
          total_experts));

  exec::OptPassCfg::GetInstance()->BkupAndDisableAndAllOptPass();

  LazyOp<at::Tensor> op{
      "hpu::mixture_of_experts_recomp_fwd",
      {
          hidden_states,
          expert_routing_table,
          router_weights,
          w1,
          w2,
          w3,
          permuted_weights,
          activation,
          experts_min,
          experts_max,
          chunk_size,
          total_experts,
      },
      {hidden_states.sizes().vec()},
      0};

  op.viewUpdateInputs();
  op.SetOutputMetaFn(MixtureOfExpertsFwdRecompMeta);

  RUN_MAYBE_WITH_ACC_THREAD(mixture_of_experts, op)
}

std::vector<at::Tensor> mixture_of_experts_fwd_fused_weights(
    const at::Tensor& hidden_states,
    const at::Tensor& expert_routing_table,
    const at::Tensor& router_weights,
    const at::TensorList w12,
    const at::TensorList w3,
    const bool permuted_weights,
    const std::string_view activation,
    const int64_t experts_min,
    const int64_t experts_max,
    const int64_t chunk_size,
    const int64_t total_experts) {
  PT_LAZY_TRACE;
  PT_OP_INFO(
      "mixture_of_experts_fwd.fused_weights :",
      DUMP_11ARGS(
          hidden_states,
          expert_routing_table,
          router_weights,
          w12,
          w3,
          permuted_weights,
          activation,
          experts_min,
          experts_max,
          chunk_size,
          total_experts));

  exec::OptPassCfg::GetInstance()->BkupAndDisableAndAllOptPass();

  LazyOp<std::vector<at::Tensor>> op{
      "hpu::mixture_of_experts_fwd",
      {hidden_states,
       expert_routing_table,
       router_weights,
       w12,
       w3,
       permuted_weights,
       activation,
       experts_min,
       experts_max,
       chunk_size,
       total_experts},
      MixtureOfExpertsFwdShapes,
      0};

  op.viewUpdateInputs();
  op.SetOutputMetaFn(MixtureOfExpertsFwdMeta);

  RUN_MAYBE_WITH_ACC_THREAD(mixture_of_experts, op)
}

at::Tensor mixture_of_experts_recomp_fwd_fused_weights(
    const at::Tensor& hidden_states,
    const at::Tensor& expert_routing_table,
    const at::Tensor& router_weights,
    const at::TensorList w12,
    const at::TensorList w3,
    const bool permuted_weights,
    const std::string_view activation,
    const int64_t experts_min,
    const int64_t experts_max,
    const int64_t chunk_size,
    const int64_t total_experts) {
  PT_LAZY_TRACE;
  PT_OP_INFO(
      "mixture_of_experts_recomp_fwd.fused_weights :",
      DUMP_11ARGS(
          hidden_states,
          expert_routing_table,
          router_weights,
          w12,
          w3,
          permuted_weights,
          activation,
          experts_min,
          experts_max,
          chunk_size,
          total_experts));

  exec::OptPassCfg::GetInstance()->BkupAndDisableAndAllOptPass();

  LazyOp<at::Tensor> op{
      "hpu::mixture_of_experts_recomp_fwd",
      {hidden_states,
       expert_routing_table,
       router_weights,
       w12,
       w3,
       permuted_weights,
       activation,
       experts_min,
       experts_max,
       chunk_size,
       total_experts},
      {hidden_states.sizes().vec()},
      0};

  op.viewUpdateInputs();

  RUN_MAYBE_WITH_ACC_THREAD(mixture_of_experts, op)
}

std::vector<at::Tensor> mixture_of_experts_bwd(
    const at::Tensor& grad_tokens_in,
    const at::Tensor& chunks_input,
    const at::Tensor& token_to_chunk,
    const at::Tensor& token_in_chunk,
    const at::Tensor& chunks_routing_table,
    const at::Tensor& chunks_routing_weights,
    const at::Tensor& gemm1_out,
    const at::Tensor& gemm2_out,
    const at::Tensor& activation_out,
    const at::Tensor& mult_out,
    const at::Tensor& mlp_out,
    const at::TensorList w1,
    const at::TensorList w2,
    const at::TensorList w3,
    const bool permuted_weights,
    const std::string_view activation,
    const int64_t experts_min,
    const int64_t experts_max,
    const std::vector<int64_t> router_weights_size) {
  PT_LAZY_OP_TRACE;
  PT_OP_INFO(
      "mixture_of_experts_bwd :",
      DUMP_19ARGS(
          grad_tokens_in,
          chunks_input,
          token_to_chunk,
          token_in_chunk,
          chunks_routing_table,
          chunks_routing_weights,
          gemm1_out,
          gemm2_out,
          activation_out,
          mult_out,
          mlp_out,
          w1,
          w2,
          w3,
          permuted_weights,
          activation,
          experts_min,
          experts_max,
          router_weights_size));

  std::vector<std::vector<int64_t>> out_shapes;
  out_shapes.reserve(2 + 3 * w1.size());
  out_shapes.push_back(grad_tokens_in.sizes().vec());
  out_shapes.push_back(router_weights_size);
  for (size_t i = 0; i < w1.size(); ++i) {
    out_shapes.push_back(w1[i].sizes().vec());
  }
  for (size_t i = 0; i < w2.size(); ++i) {
    out_shapes.push_back(w2[i].sizes().vec());
  }
  for (size_t i = 0; i < w3.size(); ++i) {
    out_shapes.push_back(w3[i].sizes().vec());
  }

  LazyOp<std::vector<at::Tensor>> op{
      "hpu::mixture_of_experts_bwd",
      {grad_tokens_in,
       chunks_input,
       token_to_chunk,
       token_in_chunk,
       chunks_routing_table,
       chunks_routing_weights,
       gemm1_out,
       gemm2_out,
       activation_out,
       mult_out,
       mlp_out,
       w1,
       w2,
       w3,
       permuted_weights,
       activation,
       experts_min,
       experts_max,
       router_weights_size},
      std::move(out_shapes),
      0};

  op.viewUpdateInputs();
  op.SetOutputMetaFn(MixtureOfExpertsBwdMeta);

  RUN_MAYBE_WITH_ACC_THREAD(mixture_of_experts, op)
}

std::vector<at::Tensor> mixture_of_experts_recomp_bwd(
    const at::Tensor& grad_tokens_in,
    const at::Tensor& hidden_states,
    const at::Tensor& expert_routing_table,
    const at::Tensor& router_weights,
    const at::TensorList w1,
    const at::TensorList w2,
    const at::TensorList w3,
    const bool permuted_weights,
    const std::string_view activation,
    const int64_t experts_min,
    const int64_t experts_max,
    const int64_t chunk_size,
    const int64_t total_experts) {
  PT_LAZY_OP_TRACE;
  PT_OP_INFO(
      "mixture_of_experts_recomp_bwd :",
      DUMP_13ARGS(
          grad_tokens_in,
          hidden_states,
          expert_routing_table,
          router_weights,
          w1,
          w2,
          w3,
          permuted_weights,
          activation,
          experts_min,
          experts_max,
          chunk_size,
          total_experts));

  std::vector<std::vector<int64_t>> out_shapes;
  out_shapes.reserve(2 + 3 * w1.size());
  out_shapes.push_back(hidden_states.sizes().vec());
  out_shapes.push_back(router_weights.sizes().vec());
  for (size_t i = 0; i < w1.size(); ++i) {
    out_shapes.push_back(w1[i].sizes().vec());
  }
  for (size_t i = 0; i < w2.size(); ++i) {
    out_shapes.push_back(w2[i].sizes().vec());
  }
  for (size_t i = 0; i < w3.size(); ++i) {
    out_shapes.push_back(w3[i].sizes().vec());
  }

  LazyOp<std::vector<at::Tensor>> op{
      "hpu::mixture_of_experts_recomp_bwd",
      {grad_tokens_in,
       hidden_states,
       expert_routing_table,
       router_weights,
       w1,
       w2,
       w3,
       permuted_weights,
       activation,
       experts_min,
       experts_max,
       chunk_size,
       total_experts},
      std::move(out_shapes),
      0};

  op.viewUpdateInputs();
  op.SetOutputMetaFn(MixtureOfExpertsBwdMeta);

  RUN_MAYBE_WITH_ACC_THREAD(mixture_of_experts, op)
}

std::vector<at::Tensor> mixture_of_experts_bwd_fused_weights(
    const at::Tensor& grad_tokens_in,
    const at::Tensor& chunks_input,
    const at::Tensor& token_to_chunk,
    const at::Tensor& token_in_chunk,
    const at::Tensor& chunks_routing_table,
    const at::Tensor& chunks_routing_weights,
    const at::Tensor& gemm12_out,
    const at::Tensor& activation_out,
    const at::Tensor& mult_out,
    const at::Tensor& mlp_out,
    const at::TensorList w12,
    const at::TensorList w3,
    const bool permuted_weights,
    const std::string_view activation,
    const int64_t experts_min,
    const int64_t experts_max,
    const std::vector<int64_t> router_weights_size) {
  PT_LAZY_OP_TRACE;
  PT_OP_INFO(
      "mixture_of_experts_bwd.fused_weights :",
      DUMP_17ARGS(
          grad_tokens_in,
          chunks_input,
          token_to_chunk,
          token_in_chunk,
          chunks_routing_table,
          chunks_routing_weights,
          gemm12_out,
          activation_out,
          mult_out,
          mlp_out,
          w12,
          w3,
          permuted_weights,
          activation,
          experts_min,
          experts_max,
          router_weights_size));

  std::vector<std::vector<int64_t>> out_shapes;
  out_shapes.reserve(2 + 2 * w12.size());
  out_shapes.push_back(grad_tokens_in.sizes().vec());
  out_shapes.push_back(router_weights_size);
  for (size_t i = 0; i < w12.size(); ++i) {
    out_shapes.push_back(w12[i].sizes().vec());
  }
  for (size_t i = 0; i < w3.size(); ++i) {
    out_shapes.push_back(w3[i].sizes().vec());
  }

  LazyOp<std::vector<at::Tensor>> op{
      "hpu::mixture_of_experts_bwd",
      {grad_tokens_in,
       chunks_input,
       token_to_chunk,
       token_in_chunk,
       chunks_routing_table,
       chunks_routing_weights,
       gemm12_out,
       activation_out,
       mult_out,
       mlp_out,
       w12,
       w3,
       permuted_weights,
       activation,
       experts_min,
       experts_max,
       router_weights_size},
      std::move(out_shapes),
      0};

  op.viewUpdateInputs();
  op.SetOutputMetaFn(MixtureOfExpertsBwdMeta);

  RUN_MAYBE_WITH_ACC_THREAD(mixture_of_experts, op)
}

std::vector<at::Tensor> mixture_of_experts_recomp_bwd_fused_weights(
    const at::Tensor& grad,
    const at::Tensor& hidden_states,
    const at::Tensor& expert_routing_table,
    const at::Tensor& router_weights,
    const at::TensorList w12,
    const at::TensorList w3,
    const bool permuted_weights,
    const std::string_view activation,
    const int64_t experts_min,
    const int64_t experts_max,
    const int64_t chunk_size,
    const int64_t total_experts) {
  PT_LAZY_TRACE;
  PT_OP_INFO(
      "mixture_of_experts_recomp_bwd.fused_weights :",
      DUMP_12ARGS(
          grad,
          hidden_states,
          expert_routing_table,
          router_weights,
          w12,
          w3,
          permuted_weights,
          activation,
          experts_min,
          experts_max,
          chunk_size,
          total_experts));

  std::vector<std::vector<int64_t>> out_shapes;
  out_shapes.reserve(2 + 2 * w12.size());
  out_shapes.push_back(hidden_states.sizes().vec());
  out_shapes.push_back(router_weights.sizes().vec());
  for (size_t i = 0; i < w12.size(); ++i) {
    out_shapes.push_back(w12[i].sizes().vec());
  }
  for (size_t i = 0; i < w3.size(); ++i) {
    out_shapes.push_back(w3[i].sizes().vec());
  }

  LazyOp<std::vector<at::Tensor>> op{
      "hpu::mixture_of_experts_recomp_bwd",
      {grad,
       hidden_states,
       expert_routing_table,
       router_weights,
       w12,
       w3,
       permuted_weights,
       activation,
       experts_min,
       experts_max,
       chunk_size,
       total_experts},
      {hidden_states.sizes().vec()},
      0};

  op.viewUpdateInputs();
  op.SetOutputMetaFn(MixtureOfExpertsBwdMeta);

  RUN_MAYBE_WITH_ACC_THREAD(mixture_of_experts, op)
}

} // namespace habana

namespace habana_lazy {
using namespace habana;

std::tuple<at::Tensor, at::Tensor> mixture_of_experts_fp8_measurement_lazy(
    const at::Tensor& hidden_states,
    const at::Tensor& expert_routing_table,
    const at::Tensor& router_weights,
    const at::TensorList w1,
    const at::TensorList w2,
    const at::TensorList w3,
    const bool permuted_weights,
    const std::string_view activation,
    const int64_t experts_min,
    const int64_t experts_max,
    const bool measurement_mode,
    const int64_t chunk_size,
    const int64_t total_experts) {
  PT_LAZY_TRACE;
  PT_OP_INFO(
      "mixture_of_experts.fp8_measurement :",
      DUMP_13ARGS(
          hidden_states,
          expert_routing_table,
          router_weights,
          w1,
          w2,
          w3,
          permuted_weights,
          activation,
          experts_min,
          experts_max,
          measurement_mode,
          chunk_size,
          total_experts));

  LazyOp<std::tuple<at::Tensor, at::Tensor>> op{
      "hpu::mixture_of_experts_fp8_measurement",
      {hidden_states,
       expert_routing_table,
       router_weights,
       w1,
       w2,
       w3,
       permuted_weights,
       activation,
       experts_min,
       experts_max,
       measurement_mode,
       chunk_size,
       total_experts},
      {hidden_states.sizes().vec(), {static_cast<int64_t>(w1.size())}},
      0};

  RUN_MAYBE_WITH_ACC_THREAD(mixture_of_experts, op)
}

std::tuple<at::Tensor, at::Tensor>
mixture_of_experts_fp8_measurement_fused_weights_lazy(
    const at::Tensor& hidden_states,
    const at::Tensor& expert_routing_table,
    const at::Tensor& router_weights,
    const at::TensorList w12,
    const at::TensorList w3,
    const bool permuted_weights,
    const std::string_view activation,
    const int64_t experts_min,
    const int64_t experts_max,
    const bool measurement_mode,
    const int64_t chunk_size,
    const int64_t total_experts) {
  PT_LAZY_TRACE;
  PT_OP_INFO(
      "mixture_of_experts.fp8_measurement_fused_weights :",
      DUMP_12ARGS(
          hidden_states,
          expert_routing_table,
          router_weights,
          w12,
          w3,
          permuted_weights,
          activation,
          experts_min,
          experts_max,
          measurement_mode,
          chunk_size,
          total_experts));

  LazyOp<std::tuple<at::Tensor, at::Tensor>> op{
      "hpu::mixture_of_experts_fp8_measurement",
      {hidden_states,
       expert_routing_table,
       router_weights,
       w12,
       w3,
       permuted_weights,
       activation,
       experts_min,
       experts_max,
       measurement_mode,
       chunk_size,
       total_experts},
      {hidden_states.sizes().vec(), {static_cast<int64_t>(w12.size())}},
      0};

  RUN_MAYBE_WITH_ACC_THREAD(mixture_of_experts, op)
}

std::vector<at::Tensor> mixture_of_experts_fwd_fp8_fused_weights_lazy(
    const at::Tensor& hidden_states,
    const at::Tensor& expert_routing_table,
    const at::Tensor& router_weights,
    const at::TensorList w12,
    const at::TensorList w3,
    const at::TensorList d_scale_hidden_states,
    const at::TensorList d_scale_intermediate_hidden_states,
    const at::TensorList d_scale_w12,
    const at::TensorList d_scale_w3,
    const bool permuted_weights,
    const std::string_view activation,
    const int64_t experts_min,
    const int64_t experts_max,
    const bool scaled_swiglu,
    const bool hybrid_mode,
    const bool is_first_amax,
    const bool is_second_amax) {
  PT_LAZY_TRACE;
  PT_OP_INFO(
      "mixture_of_experts_fwd.fp8_fused_weights :",
      DUMP_17ARGS(
          hidden_states,
          expert_routing_table,
          router_weights,
          w12,
          w3,
          d_scale_hidden_states,
          d_scale_intermediate_hidden_states,
          d_scale_w12,
          d_scale_w3,
          permuted_weights,
          activation,
          experts_min,
          experts_max,
          scaled_swiglu,
          hybrid_mode,
          is_first_amax,
          is_second_amax));

  exec::OptPassCfg::GetInstance()->BkupAndDisableAndAllOptPass();

  LazyOp<std::vector<at::Tensor>> op{
      "hpu::mixture_of_experts_fwd",
      {hidden_states,
       expert_routing_table,
       router_weights,
       w12,
       w3,
       d_scale_hidden_states,
       d_scale_intermediate_hidden_states,
       d_scale_w12,
       d_scale_w3,
       permuted_weights,
       activation,
       experts_min,
       experts_max,
       scaled_swiglu,
       hybrid_mode,
       is_first_amax,
       is_second_amax},
      MixtureOfExpertsFwdFp8Shapes,
      0};

  op.viewUpdateInputs();
  op.SetOutputMetaFn(MixtureOfExpertsFwdFp8Meta);

  RUN_MAYBE_WITH_ACC_THREAD(mixture_of_experts, op)
}

std::vector<at::Tensor> mixture_of_experts_recomp_fwd_fp8_fused_weights_lazy(
    const at::Tensor& hidden_states,
    const at::Tensor& expert_routing_table,
    const at::Tensor& router_weights,
    const at::TensorList w12,
    const at::TensorList w3,
    const at::TensorList d_scale_hidden_states,
    const at::TensorList d_scale_intermediate_hidden_states,
    const at::TensorList d_scale_w12,
    const at::TensorList d_scale_w3,
    const bool permuted_weights,
    const std::string_view activation,
    const int64_t experts_min,
    const int64_t experts_max,
    const bool scaled_swiglu,
    const bool hybrid_mode,
    const bool is_first_amax,
    const bool is_second_amax) {
  PT_LAZY_TRACE;
  PT_OP_INFO(
      "mixture_of_experts_recomp_fwd.fp8_fused_weights :",
      DUMP_17ARGS(
          hidden_states,
          expert_routing_table,
          router_weights,
          w12,
          w3,
          d_scale_hidden_states,
          d_scale_intermediate_hidden_states,
          d_scale_w12,
          d_scale_w3,
          permuted_weights,
          activation,
          experts_min,
          experts_max,
          scaled_swiglu,
          hybrid_mode,
          is_first_amax,
          is_second_amax));
  exec::OptPassCfg::GetInstance()->BkupAndDisableAndAllOptPass();

  LazyOp<std::vector<at::Tensor>> op{
      "hpu::mixture_of_experts_recomp_fwd",
      {hidden_states,
       expert_routing_table,
       router_weights,
       w12,
       w3,
       d_scale_hidden_states,
       d_scale_intermediate_hidden_states,
       d_scale_w12,
       d_scale_w3,
       permuted_weights,
       activation,
       experts_min,
       experts_max,
       scaled_swiglu,
       hybrid_mode,
       is_first_amax,
       is_second_amax},
      MixtureOfExpertsRecompFwdFp8Shapes,
      0};

  op.viewUpdateInputs();
  op.SetOutputMetaFn(MixtureOfExpertsRecompFwdFp8Meta);

  RUN_MAYBE_WITH_ACC_THREAD(mixture_of_experts_recomp_fwd, op)
}

std::vector<at::Tensor> mixture_of_experts_bwd_fp8_fused_weights_lazy(
    const at::Tensor& grad_tokens_in,
    const at::Tensor& chunks_input,
    const at::Tensor& token_to_chunk,
    const at::Tensor& token_in_chunk,
    const at::Tensor& chunks_routing_table,
    const at::Tensor& chunks_routing_weights,
    const at::Tensor& gemm12_out,
    const at::Tensor& activation_out,
    const at::Tensor& mult_out,
    const at::Tensor& mlp_out,
    const at::TensorList w12,
    const at::TensorList w3,
    const at::TensorList d_scale_hidden_states,
    const at::TensorList d_scale_intermediate_hidden_states,
    const at::TensorList d_scale_w12,
    const at::TensorList d_scale_w3,
    const at::TensorList d_scale_first_gemm_grad,
    const at::TensorList d_scale_second_gemm_grad,
    const bool permuted_weights,
    const std::string_view activation,
    const int64_t experts_min,
    const int64_t experts_max,
    const std::vector<int64_t> router_weights_size,
    const bool scaled_swiglu,
    const bool hybrid_mode,
    const bool is_first_amax,
    const bool is_second_amax) {
  PT_LAZY_TRACE;
  PT_OP_INFO(
      "mixture_of_experts_bwd.fp8_fused_weights :",
      DUMP_27ARGS(
          grad_tokens_in,
          chunks_input,
          token_to_chunk,
          token_in_chunk,
          chunks_routing_table,
          chunks_routing_weights,
          gemm12_out,
          activation_out,
          mult_out,
          mlp_out,
          w12,
          w3,
          d_scale_hidden_states,
          d_scale_intermediate_hidden_states,
          d_scale_w12,
          d_scale_w3,
          d_scale_first_gemm_grad,
          d_scale_second_gemm_grad,
          permuted_weights,
          activation,
          experts_min,
          experts_max,
          router_weights_size,
          scaled_swiglu,
          hybrid_mode,
          is_first_amax,
          is_second_amax));

  exec::OptPassCfg::GetInstance()->BkupAndDisableAndAllOptPass();

  std::vector<std::vector<int64_t>> out_shapes = get_bwd_output_shapes(
      grad_tokens_in,
      router_weights_size,
      {w12, w3},
      static_cast<size_t>(is_first_amax + is_second_amax));

  LazyOp<std::vector<at::Tensor>> op{
      "hpu::mixture_of_experts_bwd",
      {grad_tokens_in,
       chunks_input,
       token_to_chunk,
       token_in_chunk,
       chunks_routing_table,
       chunks_routing_weights,
       gemm12_out,
       activation_out,
       mult_out,
       mlp_out,
       w12,
       w3,
       d_scale_hidden_states,
       d_scale_intermediate_hidden_states,
       d_scale_w12,
       d_scale_w3,
       d_scale_first_gemm_grad,
       d_scale_second_gemm_grad,
       permuted_weights,
       activation,
       experts_min,
       experts_max,
       router_weights_size,
       scaled_swiglu,
       hybrid_mode,
       is_first_amax,
       is_second_amax},
      out_shapes,
      0};

  op.viewUpdateInputs();
  op.SetOutputMetaFn(MixtureOfExpertsBwdFp8Meta);

  RUN_MAYBE_WITH_ACC_THREAD(mixture_of_experts_recomp_bwd, op)
}

std::vector<at::Tensor> mixture_of_experts_recomp_bwd_fp8_fused_weights_lazy(
    const at::Tensor& grad_tokens_in,
    const at::Tensor& hidden_states,
    const at::Tensor& expert_routing_table,
    const at::Tensor& router_weights,
    const at::TensorList w12,
    const at::TensorList w3,
    const at::TensorList d_scale_hidden_states,
    const at::TensorList d_scale_intermediate_hidden_states,
    const at::TensorList d_scale_w12,
    const at::TensorList d_scale_w3,
    const at::TensorList d_scale_gemm1_grad,
    const at::TensorList d_scale_gemm2_grad,
    const bool permuted_weights,
    const std::string_view activation,
    const int64_t experts_min,
    const int64_t experts_max,
    const bool scaled_swiglu,
    const bool hybrid_mode,
    const bool is_first_amax,
    const bool is_second_amax) {
  PT_LAZY_TRACE;
  PT_OP_INFO(
      "mixture_of_experts_recomp_recomp_bwd.fp8_fused_weights :",
      DUMP_20ARGS(
          grad_tokens_in,
          hidden_states,
          expert_routing_table,
          router_weights,
          w12,
          w3,
          d_scale_hidden_states,
          d_scale_intermediate_hidden_states,
          d_scale_w12,
          d_scale_w3,
          d_scale_gemm1_grad,
          d_scale_gemm2_grad,
          permuted_weights,
          activation,
          experts_min,
          experts_max,
          scaled_swiglu,
          hybrid_mode,
          is_first_amax,
          is_second_amax));
  exec::OptPassCfg::GetInstance()->BkupAndDisableAndAllOptPass();

  std::vector<std::vector<int64_t>> out_shapes = get_bwd_output_shapes(
      grad_tokens_in,
      router_weights.sizes().vec(),
      {w12, w3},
      static_cast<size_t>(is_first_amax + is_second_amax));

  LazyOp<std::vector<at::Tensor>> op{
      "hpu::mixture_of_experts_recomp_bwd",
      {grad_tokens_in,
       hidden_states,
       expert_routing_table,
       router_weights,
       w12,
       w3,
       d_scale_hidden_states,
       d_scale_intermediate_hidden_states,
       d_scale_w12,
       d_scale_w3,
       d_scale_gemm1_grad,
       d_scale_gemm2_grad,
       permuted_weights,
       activation,
       experts_min,
       experts_max,
       scaled_swiglu,
       hybrid_mode,
       is_first_amax,
       is_second_amax},
      out_shapes,
      0};

  op.viewUpdateInputs();
  op.SetOutputMetaFn(MixtureOfExpertsBwdFp8Meta);

  RUN_MAYBE_WITH_ACC_THREAD(mixture_of_experts_recomp_bwd, op)
}

} // namespace habana_lazy
