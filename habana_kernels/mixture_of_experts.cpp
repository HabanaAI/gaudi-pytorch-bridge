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
#include "common/mixture_of_experts.hpp"
#include "habana_helpers/logging.h"
#include "habana_kernels/lazy_kernels.h"
#include "habana_lazy/hlexec.h"
#include "habana_lazy/hpu_stage_submission.h"
#include "habana_lazy/lazy_executor.h"
#include "hpu_ops/op_logger.h"

namespace habana {
using namespace habana_lazy;

std::vector<at::Tensor> mixture_of_experts_fwd(
    const at::Tensor& hidden_states,
    const at::Tensor& expert_routing_table,
    const at::Tensor& router_weights,
    const at::TensorList w1,
    const at::TensorList w2,
    const at::TensorList w3,
    const bool permuted_weights,
    const c10::string_view activation,
    const int64_t experts_min,
    const int64_t experts_max) {
  PT_LAZY_OP_TRACE;
  PT_OP_INFO(
      "mixture_of_experts_fwd :",
      DUMP_10ARGS(
          hidden_states,
          expert_routing_table,
          router_weights,
          w1,
          w2,
          w3,
          permuted_weights,
          activation,
          experts_min,
          experts_max));

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
       experts_max},
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
    const c10::string_view activation,
    const int64_t experts_min,
    const int64_t experts_max) {
  PT_LAZY_OP_TRACE;
  PT_OP_INFO(
      "mixture_of_experts_recomp_fwd :",
      DUMP_10ARGS(
          hidden_states,
          expert_routing_table,
          router_weights,
          w1,
          w2,
          w3,
          permuted_weights,
          activation,
          experts_min,
          experts_max));

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
    const c10::string_view activation,
    const int64_t experts_min,
    const int64_t experts_max) {
  PT_LAZY_TRACE;
  PT_OP_INFO(
      "mixture_of_experts_fwd.fused_weights :",
      DUMP_9ARGS(
          hidden_states,
          expert_routing_table,
          router_weights,
          w12,
          w3,
          permuted_weights,
          activation,
          experts_min,
          experts_max));

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
       experts_max},
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
    const c10::string_view activation,
    const int64_t experts_min,
    const int64_t experts_max) {
  PT_LAZY_TRACE;
  PT_OP_INFO(
      "mixture_of_experts_recomp_fwd.fused_weights :",
      DUMP_9ARGS(
          hidden_states,
          expert_routing_table,
          router_weights,
          w12,
          w3,
          permuted_weights,
          activation,
          experts_min,
          experts_max));

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
       experts_max},
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
    const c10::string_view activation,
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
    const c10::string_view activation,
    const int64_t experts_min,
    const int64_t experts_max) {
  PT_LAZY_OP_TRACE;
  PT_OP_INFO(
      "mixture_of_experts_recomp_bwd :",
      DUMP_11ARGS(
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
          experts_max));

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
       experts_max},
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
    const c10::string_view activation,
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
    const c10::string_view activation,
    const int64_t experts_min,
    const int64_t experts_max) {
  PT_LAZY_TRACE;
  PT_OP_INFO(
      "mixture_of_experts_recomp_bwd.fused_weights :",
      DUMP_10ARGS(
          grad,
          hidden_states,
          expert_routing_table,
          router_weights,
          w12,
          w3,
          permuted_weights,
          activation,
          experts_min,
          experts_max));

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
       experts_max},
      {hidden_states.sizes().vec()},
      0};

  op.viewUpdateInputs();
  op.SetOutputMetaFn(MixtureOfExpertsBwdMeta);

  RUN_MAYBE_WITH_ACC_THREAD(mixture_of_experts, op)
}

} // namespace habana

namespace habana_lazy {
using namespace habana;

at::Tensor mixture_of_experts_lazy(
    const at::Tensor& hidden_states,
    const at::Tensor& expert_routing_table,
    const at::Tensor& router_weights,
    const at::TensorList w1,
    const at::TensorList w2,
    const at::TensorList w3,
    const bool permuted_weights,
    const c10::string_view activation,
    const int64_t experts_min,
    const int64_t experts_max,
    const std::optional<bool> recomp) {
  PT_LAZY_OP_TRACE;
  PT_OP_INFO(
      "mixture_of_experts :",
      DUMP_11ARGS(
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
          recomp));

  LazyOp<at::Tensor> op{
      "hpu::mixture_of_experts",
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
       recomp},
      {hidden_states.sizes().vec()},
      0};

  RUN_MAYBE_WITH_ACC_THREAD(mixture_of_experts, op)
}

at::Tensor mixture_of_experts_fused_weights_lazy(
    const at::Tensor& hidden_states,
    const at::Tensor& expert_routing_table,
    const at::Tensor& router_weights,
    const at::TensorList w12,
    const at::TensorList w3,
    const bool permuted_weights,
    const c10::string_view activation,
    const int64_t experts_min,
    const int64_t experts_max,
    const std::optional<bool> recomp) {
  PT_LAZY_TRACE;
  PT_OP_INFO(
      "mixture_of_experts.fused_weights :",
      DUMP_10ARGS(
          hidden_states,
          expert_routing_table,
          router_weights,
          w12,
          w3,
          permuted_weights,
          activation,
          experts_min,
          experts_max,
          recomp));

  LazyOp<at::Tensor> op{
      "hpu::mixture_of_experts",
      {hidden_states,
       expert_routing_table,
       router_weights,
       w12,
       w3,
       permuted_weights,
       activation,
       experts_min,
       experts_max,
       recomp},
      {hidden_states.sizes().vec()},
      0};

  RUN_MAYBE_WITH_ACC_THREAD(mixture_of_experts, op)
}

std::tuple<at::Tensor, at::Tensor> mixture_of_experts_fp8_measurement_lazy(
    const at::Tensor& hidden_states,
    const at::Tensor& expert_routing_table,
    const at::Tensor& router_weights,
    const at::TensorList w1,
    const at::TensorList w2,
    const at::TensorList w3,
    const bool permuted_weights,
    const c10::string_view activation,
    const int64_t experts_min,
    const int64_t experts_max,
    const bool measurement_mode) {
  PT_LAZY_TRACE;
  PT_OP_INFO(
      "mixture_of_experts.fp8_measurement :",
      DUMP_11ARGS(
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
          measurement_mode));

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
       measurement_mode},
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
    const c10::string_view activation,
    const int64_t experts_min,
    const int64_t experts_max,
    const bool measurement_mode) {
  PT_LAZY_TRACE;
  PT_OP_INFO(
      "mixture_of_experts.fp8_measurement_fused_weights :",
      DUMP_10ARGS(
          hidden_states,
          expert_routing_table,
          router_weights,
          w12,
          w3,
          permuted_weights,
          activation,
          experts_min,
          experts_max,
          measurement_mode));

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
       measurement_mode},
      {hidden_states.sizes().vec(), {static_cast<int64_t>(w12.size())}},
      0};

  RUN_MAYBE_WITH_ACC_THREAD(mixture_of_experts, op)
}

at::Tensor mixture_of_experts_fwd_autograd_lazy(
    const at::Tensor& hidden_states,
    const at::Tensor& expert_routing_table,
    const at::Tensor& router_weights,
    const at::TensorList w1,
    const at::TensorList w2,
    const at::TensorList w3,
    const bool permuted_weights,
    const c10::string_view activation,
    const int64_t experts_min,
    const int64_t experts_max,
    const std::optional<bool> recomp) {
  PT_LAZY_TRACE;
  PT_OP_INFO(
      "mixture_of_experts.fwd :",
      DUMP_11ARGS(
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
          recomp));
  return recomp.value_or(true) ? MixtureOfExpertsRecompFunction::apply(
                                     hidden_states,
                                     expert_routing_table,
                                     router_weights,
                                     w1,
                                     w2,
                                     w3,
                                     permuted_weights,
                                     activation,
                                     experts_min,
                                     experts_max)[0]
                               : MixtureOfExpertsFunction::apply(
                                     hidden_states,
                                     expert_routing_table,
                                     router_weights,
                                     w1,
                                     w2,
                                     w3,
                                     permuted_weights,
                                     activation,
                                     experts_min,
                                     experts_max)[0];
}

at::Tensor mixture_of_experts_fwd_fused_weights_autograd_lazy(
    const at::Tensor& hidden_states,
    const at::Tensor& expert_routing_table,
    const at::Tensor& router_weights,
    const at::TensorList w12,
    const at::TensorList w3,
    const bool permuted_weights,
    const c10::string_view activation,
    const int64_t experts_min,
    const int64_t experts_max,
    const std::optional<bool> recomp) {
  PT_LAZY_TRACE;
  PT_OP_INFO(
      "mixture_of_experts_fwd_fused_weights :",
      DUMP_10ARGS(
          hidden_states,
          expert_routing_table,
          router_weights,
          w12,
          w3,
          permuted_weights,
          activation,
          experts_min,
          experts_max,
          recomp));

  return recomp.value_or(true)
      ? MixtureOfExpertsRecompFusedWeightsFunction::apply(
            hidden_states,
            expert_routing_table,
            router_weights,
            w12,
            w3,
            permuted_weights,
            activation,
            experts_min,
            experts_max)[0]
      : MixtureOfExpertsFusedWeightsFunction::apply(
            hidden_states,
            expert_routing_table,
            router_weights,
            w12,
            w3,
            permuted_weights,
            activation,
            experts_min,
            experts_max)[0];
}

} // namespace habana_lazy
