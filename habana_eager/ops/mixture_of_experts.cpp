/**
 * Copyright (c) 2025-2025 Intel Corporation
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

#include "habana_eager/ops/mixture_of_experts.h"
#include <torch/autograd.h>
#include "common/dump_args.h"
#include "common/mixture_of_experts.hpp"
#include "generated/backend/cast_from_fp8.h"
#include "generated/backend/cast_to_fp8_v2.h"
#include "generated/backend/fp8_gemm_v2.h"
#include "habana_eager/ops/eager_op.h"
#include "habana_helpers/logging.h"
#include "hpu_ops/op_logger.h"

namespace habana {
using namespace habana::eager;

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
  PT_EAGER_TRACE;
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

  static auto op = torch::Dispatcher::singleton()
                       .findSchemaOrThrow("hpu::mixture_of_experts_fwd", "")
                       .typed<decltype(mixture_of_experts_fwd)>();

  return op.call(
      hidden_states,
      expert_routing_table,
      router_weights,
      w1,
      w2,
      w3,
      permuted_weights,
      activation,
      experts_min,
      experts_max);
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
  PT_EAGER_TRACE;
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

  static auto op =
      torch::Dispatcher::singleton()
          .findSchemaOrThrow("hpu::mixture_of_experts_recomp_fwd", "")
          .typed<decltype(mixture_of_experts_recomp_fwd)>();

  return op.call(
      hidden_states,
      expert_routing_table,
      router_weights,
      w1,
      w2,
      w3,
      permuted_weights,
      activation,
      experts_min,
      experts_max);
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
  PT_EAGER_TRACE;
  PT_OP_INFO(
      "mixture_of_experts_fwd_fused_weights :",
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

  static auto op =
      torch::Dispatcher::singleton()
          .findSchemaOrThrow("hpu::mixture_of_experts_fwd", "fused_weights")
          .typed<decltype(mixture_of_experts_fwd_fused_weights)>();

  return op.call(
      hidden_states,
      expert_routing_table,
      router_weights,
      w12,
      w3,
      permuted_weights,
      activation,
      experts_min,
      experts_max);
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
  PT_EAGER_TRACE;
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

  static auto op =
      torch::Dispatcher::singleton()
          .findSchemaOrThrow(
              "hpu::mixture_of_experts_recomp_fwd", "fused_weights")
          .typed<decltype(mixture_of_experts_recomp_fwd_fused_weights)>();

  return op.call(
      hidden_states,
      expert_routing_table,
      router_weights,
      w12,
      w3,
      permuted_weights,
      activation,
      experts_min,
      experts_max);
}

std::vector<at::Tensor> mixture_of_experts_bwd(
    const at::Tensor& grad_tokens_in,
    const at::Tensor& router_weights,
    const at::Tensor& chunks_input,
    const at::Tensor& token_to_chunk,
    const at::Tensor& token_in_chunk,
    const at::Tensor& chunks_routing_table,
    const at::Tensor& gemm1_out,
    const at::Tensor& gemm2_out,
    const at::Tensor& activation_out,
    const at::Tensor& mult_out,
    const at::TensorList w1,
    const at::TensorList w2,
    const at::TensorList w3,
    const bool permuted_weights,
    const c10::string_view activation,
    const int64_t experts_min,
    const int64_t experts_max) {
  PT_EAGER_TRACE;
  PT_OP_INFO(
      "mixture_of_experts_bwd :",
      DUMP_17ARGS(
          grad_tokens_in,
          router_weights,
          chunks_input,
          token_to_chunk,
          token_in_chunk,
          chunks_routing_table,
          gemm1_out,
          gemm2_out,
          activation_out,
          mult_out,
          w1,
          w2,
          w3,
          permuted_weights,
          activation,
          experts_min,
          experts_max));

  static auto op = torch::Dispatcher::singleton()
                       .findSchemaOrThrow("hpu::mixture_of_experts_bwd", "")
                       .typed<decltype(mixture_of_experts_bwd)>();
  return op.call(
      grad_tokens_in,
      router_weights,
      chunks_input,
      token_to_chunk,
      token_in_chunk,
      chunks_routing_table,
      gemm1_out,
      gemm2_out,
      activation_out,
      mult_out,
      w1,
      w2,
      w3,
      permuted_weights,
      activation,
      experts_min,
      experts_max);
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
  PT_EAGER_TRACE;
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

  static auto op =
      torch::Dispatcher::singleton()
          .findSchemaOrThrow("hpu::mixture_of_experts_recomp_bwd", "")
          .typed<decltype(mixture_of_experts_recomp_bwd)>();

  return op.call(
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
      experts_max);
}

std::vector<at::Tensor> mixture_of_experts_bwd_fused_weights(
    const at::Tensor& grad_tokens_in,
    const at::Tensor& router_weights,
    const at::Tensor& chunks_input,
    const at::Tensor& token_to_chunk,
    const at::Tensor& token_in_chunk,
    const at::Tensor& chunks_routing_table,
    const at::Tensor& gemm12_out,
    const at::Tensor& activation_out,
    const at::Tensor& mult_out,
    const at::TensorList w12,
    const at::TensorList w3,
    const bool permuted_weights,
    const c10::string_view activation,
    const int64_t experts_min,
    const int64_t experts_max) {
  PT_EAGER_TRACE;
  PT_OP_INFO(
      "mixture_of_experts_bwd.fused_weights :",
      DUMP_15ARGS(
          grad_tokens_in,
          router_weights,
          chunks_input,
          token_to_chunk,
          token_in_chunk,
          chunks_routing_table,
          gemm12_out,
          activation_out,
          mult_out,
          w12,
          w3,
          permuted_weights,
          activation,
          experts_min,
          experts_max));

  static auto op =
      torch::Dispatcher::singleton()
          .findSchemaOrThrow("hpu::mixture_of_experts_bwd", "fused_weights")
          .typed<decltype(mixture_of_experts_bwd_fused_weights)>();

  return op.call(
      grad_tokens_in,
      router_weights,
      chunks_input,
      token_to_chunk,
      token_in_chunk,
      chunks_routing_table,
      gemm12_out,
      activation_out,
      mult_out,
      w12,
      w3,
      permuted_weights,
      activation,
      experts_min,
      experts_max);
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
  PT_EAGER_TRACE;
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

  static auto op =
      torch::Dispatcher::singleton()
          .findSchemaOrThrow(
              "hpu::mixture_of_experts_recomp_bwd", "fused_weights")
          .typed<decltype(mixture_of_experts_recomp_bwd_fused_weights)>();

  return op.call(
      grad,
      hidden_states,
      expert_routing_table,
      router_weights,
      w12,
      w3,
      permuted_weights,
      activation,
      experts_min,
      experts_max);
}
namespace eager {

static std::pair<std::vector<at::Tensor>, std::vector<at::Tensor>>
split_weights_tensor(const at::TensorList& w12, bool permuted_weights) {
  std::vector<at::Tensor> w1, w2;
  const auto split_dim = permuted_weights ? 0 : 1;
  const auto split_index = w12[0].size(split_dim) / 2;
  for (const auto& tensor : w12) {
    auto w12_split = tensor.split(split_index, split_dim);
    w1.push_back(w12_split[0]);
    w2.push_back(w12_split[1]);
  }
  return {w1, w2};
}

std::function<at::Tensor(const at::Tensor& x)> get_activation_fn(
    const c10::string_view& activation) {
  if (activation == "gelu") {
    return [](const at::Tensor& x) { return torch::nn::functional::gelu(x); };
  } else if (activation == "relu") {
    return [](const at::Tensor& x) { return torch::nn::functional::relu(x); };
  } else if (activation == "silu") {
    return [](const at::Tensor& x) { return torch::nn::functional::silu(x); };
  } else {
    throw std::invalid_argument("Unsupported activation");
  }
}

static std::tuple<at::Tensor, at::Tensor> mixture_of_experts_common(
    const at::Tensor& hidden_states,
    const at::Tensor& expert_routing_table,
    const at::Tensor& router_weights,
    const at::TensorList w1,
    const at::TensorList w2,
    const at::TensorList w3,
    const bool permuted_weights,
    const c10::string_view activation,
    const bool measurement_mode) {
  std::function<at::Tensor(const at::Tensor& x)> activation_fn =
      get_activation_fn(activation);
  const int num_experts = w1.size();
  const int num_tokens = hidden_states.size(0);
  const int hidden_dim = hidden_states.size(1);
  auto final_hidden_states =
      torch::zeros({1, num_tokens, hidden_dim}, hidden_states.options());
  auto padded_weights =
      torch::zeros({num_tokens, num_experts}, router_weights.options())
          .scatter_(-1, expert_routing_table, router_weights)
          .reshape({-1, num_tokens, num_experts})
          .permute({2, 0, 1})
          .unsqueeze(-1);

  auto amax_per_expert = torch::zeros(
      {num_experts},
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kHPU));
  for (int expert_idx = 0; expert_idx < num_experts; expert_idx++) {
    const at::Tensor current_expert_w1 =
        permuted_weights ? w1[expert_idx].transpose(0, 1) : w1[expert_idx];
    const at::Tensor current_expert_w2 =
        permuted_weights ? w2[expert_idx].transpose(0, 1) : w2[expert_idx];
    const at::Tensor current_expert_w3 =
        permuted_weights ? w3[expert_idx].transpose(0, 1) : w3[expert_idx];

    auto hidden_states_w1 =
        activation_fn(torch::matmul(hidden_states, current_expert_w1));
    auto hidden_states_w2 = torch::matmul(hidden_states, current_expert_w2);
    auto hidden_states_w12 = hidden_states_w1 * hidden_states_w2;
    auto expert_mask = (expert_routing_table == expert_idx);
    if (measurement_mode && expert_mask.sum().item<int>() > 0) {
      std::vector<int64_t> selected_token_indices;
      for (int64_t i = 0; i < expert_mask.size(0); i++) {
        if (expert_mask[i].sum().item<int>() > 0) {
          selected_token_indices.push_back(i);
        }
      }
      auto top_x = torch::tensor(
          selected_token_indices,
          torch::TensorOptions().dtype(torch::kInt64).device(torch::kHPU));

      auto current_state = hidden_states.index_select(0, top_x);
      auto hidden_states_w1_measure =
          activation_fn(torch::matmul(current_state, current_expert_w1));
      auto hidden_states_w2_measure =
          torch::matmul(current_state, current_expert_w2);
      amax_per_expert[expert_idx] =
          torch::amax(
              torch::abs(hidden_states_w1_measure * hidden_states_w2_measure))
              .to(torch::kFloat32);
    } else {
      amax_per_expert[expert_idx] = 0;
    }
    auto hidden_states_w3 = torch::matmul(hidden_states_w12, current_expert_w3);
    final_hidden_states += hidden_states_w3 * padded_weights[expert_idx];
  }
  auto result = final_hidden_states.reshape(hidden_states.sizes());
  return std::make_tuple(result, amax_per_expert);
}

at::Tensor mixture_of_experts(
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
    const c10::optional<bool> recomp) {
  PT_EAGER_TRACE;
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

  auto moe_common = mixture_of_experts_common(
      hidden_states,
      expert_routing_table,
      router_weights,
      w1,
      w2,
      w3,
      permuted_weights,
      activation,
      false);

  return std::get<0>(moe_common);
}

at::Tensor mixture_of_experts_fused_weights(
    const at::Tensor& hidden_states,
    const at::Tensor& expert_routing_table,
    const at::Tensor& router_weights,
    const at::TensorList w12,
    const at::TensorList w3,
    const bool permuted_weights,
    const c10::string_view activation,
    const int64_t experts_min,
    const int64_t experts_max,
    const c10::optional<bool> recomp) {
  PT_EAGER_TRACE;
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

  auto [w1, w2] = split_weights_tensor(w12, permuted_weights);

  auto moe_common = mixture_of_experts_common(
      hidden_states,
      expert_routing_table,
      router_weights,
      w1,
      w2,
      w3,
      permuted_weights,
      activation,
      false);
  return std::get<0>(moe_common);
}

std::tuple<at::Tensor, at::Tensor> mixture_of_experts_fp8_measurement(
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
  PT_EAGER_TRACE;
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
  return mixture_of_experts_common(
      hidden_states,
      expert_routing_table,
      router_weights,
      w1,
      w2,
      w3,
      permuted_weights,
      activation,
      measurement_mode);
}

std::tuple<at::Tensor, at::Tensor>
mixture_of_experts_fp8_measurement_fused_weights(
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
  PT_EAGER_TRACE;
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

  auto [w1, w2] = split_weights_tensor(w12, permuted_weights);

  return mixture_of_experts_common(
      hidden_states,
      expert_routing_table,
      router_weights,
      w1,
      w2,
      w3,
      permuted_weights,
      activation,
      measurement_mode);
}

template <typename Scale, typename Scales>
static at::Tensor mixture_of_experts_fp8_common(
    const at::Tensor& hidden_states,
    const at::Tensor& expert_routing_table,
    const at::Tensor& router_weights,
    const at::TensorList& w1,
    const at::TensorList& w2,
    const at::TensorList& w3,
    const Scale& d_scale_hidden_states,
    const Scales& d_scale_intermediate_hidden_states,
    const Scales& d_scale_w1,
    const Scales& d_scale_w2,
    const Scales& d_scale_w3,
    const bool permuted_weights,
    const c10::string_view activation) {
  std::function<at::Tensor(const at::Tensor& x)> activation_fn =
      get_activation_fn(activation);
  const at::ScalarType fp8_type = hidden_states.scalar_type();
  const int num_experts = w1.size();
  const int num_tokens = hidden_states.size(0);
  const int hidden_dim = hidden_states.size(1);
  auto final_hidden_states = torch::zeros(
      {1, num_tokens, hidden_dim},
      hidden_states.options().dtype(torch::kBFloat16));
  auto padded_weights =
      torch::zeros({num_tokens, num_experts}, router_weights.options())
          .scatter_(-1, expert_routing_table, router_weights)
          .reshape({-1, num_tokens, num_experts})
          .permute({2, 0, 1})
          .unsqueeze(-1);

  Scale default_scale;
  if constexpr (std::is_same<Scale, double>::value) {
    default_scale = 1.0;
  } else {
    default_scale = at::tensor(
        1.0, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kHPU));
  }

  for (int expert_idx = 0; expert_idx < num_experts; expert_idx++) {
    const at::Tensor current_expert_w1 =
        permuted_weights ? w1[expert_idx].transpose(0, 1) : w1[expert_idx];
    const at::Tensor current_expert_w2 =
        permuted_weights ? w2[expert_idx].transpose(0, 1) : w2[expert_idx];
    const at::Tensor current_expert_w3 =
        permuted_weights ? w3[expert_idx].transpose(0, 1) : w3[expert_idx];

    auto hidden_states_w1 = activation_fn(fp8_gemm_v2(
        hidden_states,
        false,
        current_expert_w1,
        false,
        std::nullopt,
        torch::kBFloat16,
        d_scale_hidden_states,
        d_scale_w1[expert_idx],
        std::nullopt,
        false,
        std::nullopt));

    auto hidden_states_w2 = fp8_gemm_v2(
        hidden_states,
        false,
        current_expert_w2,
        false,
        std::nullopt,
        torch::kBFloat16,
        d_scale_hidden_states,
        d_scale_w2[expert_idx],
        std::nullopt,
        false,
        std::nullopt);

    auto hidden_states_w12 = hidden_states_w1 * hidden_states_w2;

    hidden_states_w12 = std::get<0>(cast_to_fp8_v2(
        hidden_states_w12,
        d_scale_intermediate_hidden_states[expert_idx],
        false,
        false,
        fp8_type,
        std::nullopt));

    auto hidden_states_w3 = fp8_gemm_v2(
        hidden_states_w12,
        false,
        current_expert_w3,
        false,
        std::nullopt,
        torch::kBFloat16,
        default_scale,
        d_scale_w3[expert_idx],
        std::nullopt,
        false,
        std::nullopt);

    final_hidden_states += hidden_states_w3 * padded_weights[expert_idx];
  }

  auto result = final_hidden_states.reshape(hidden_states.sizes());
  return result;
}

at::Tensor mixture_of_experts_fp8(
    const at::Tensor& hidden_states,
    const at::Tensor& expert_routing_table,
    const at::Tensor& router_weights,
    const at::TensorList w1,
    const at::TensorList w2,
    const at::TensorList w3,
    const at::Tensor& d_scale_hidden_states,
    const at::TensorList d_scale_intermediate_hidden_states,
    const at::TensorList d_scale_w1,
    const at::TensorList d_scale_w2,
    const at::TensorList d_scale_w3,
    const bool permuted_weights,
    const c10::string_view activation,
    const int64_t experts_min,
    const int64_t experts_max) {
  PT_EAGER_TRACE;
  PT_OP_INFO(
      "mixture_of_experts.fp8 :",
      DUMP_15ARGS(
          hidden_states,
          expert_routing_table,
          router_weights,
          w1,
          w2,
          w3,
          d_scale_hidden_states,
          d_scale_intermediate_hidden_states,
          d_scale_w1,
          d_scale_w2,
          d_scale_w3,
          permuted_weights,
          activation,
          experts_min,
          experts_max));
  return mixture_of_experts_fp8_common(
      hidden_states,
      expert_routing_table,
      router_weights,
      w1,
      w2,
      w3,
      d_scale_hidden_states,
      d_scale_intermediate_hidden_states,
      d_scale_w1,
      d_scale_w2,
      d_scale_w3,
      permuted_weights,
      activation);
}

at::Tensor mixture_of_experts_fp8_fused_weights(
    const at::Tensor& hidden_states,
    const at::Tensor& expert_routing_table,
    const at::Tensor& router_weights,
    const at::TensorList w12,
    const at::TensorList w3,
    const at::Tensor& d_scale_hidden_states,
    const at::TensorList d_scale_intermediate_hidden_states,
    const at::TensorList d_scale_w12,
    const at::TensorList d_scale_w3,
    const bool permuted_weights,
    const c10::string_view activation,
    const int64_t experts_min,
    const int64_t experts_max) {
  PT_EAGER_TRACE;
  PT_OP_INFO(
      "mixture_of_experts.fp8_fused_weights :",
      DUMP_13ARGS(
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
          experts_max));
  auto [w1, w2] = split_weights_tensor(w12, permuted_weights);
  return mixture_of_experts_fp8_common(
      hidden_states,
      expert_routing_table,
      router_weights,
      w1,
      w2,
      w3,
      d_scale_hidden_states,
      d_scale_intermediate_hidden_states,
      d_scale_w12,
      d_scale_w12,
      d_scale_w3,
      permuted_weights,
      activation);
}

at::Tensor mixture_of_experts_fp8_scalars(
    const at::Tensor& hidden_states,
    const at::Tensor& expert_routing_table,
    const at::Tensor& router_weights,
    const at::TensorList w1,
    const at::TensorList w2,
    const at::TensorList w3,
    const double d_scale_hidden_states,
    const c10::ArrayRef<double>& d_scale_intermediate_hidden_states,
    const c10::ArrayRef<double>& d_scale_w1,
    const c10::ArrayRef<double>& d_scale_w2,
    const c10::ArrayRef<double>& d_scale_w3,
    const bool permuted_weights,
    const c10::string_view activation,
    const int64_t experts_min,
    const int64_t experts_max) {
  PT_EAGER_TRACE;
  PT_OP_INFO(
      "mixture_of_experts.fp8_scalars :",
      DUMP_15ARGS(
          hidden_states,
          expert_routing_table,
          router_weights,
          w1,
          w2,
          w3,
          d_scale_hidden_states,
          d_scale_intermediate_hidden_states,
          d_scale_w1,
          d_scale_w2,
          d_scale_w3,
          permuted_weights,
          activation,
          experts_min,
          experts_max));
  return mixture_of_experts_fp8_common(
      hidden_states,
      expert_routing_table,
      router_weights,
      w1,
      w2,
      w3,
      d_scale_hidden_states,
      d_scale_intermediate_hidden_states,
      d_scale_w1,
      d_scale_w2,
      d_scale_w3,
      permuted_weights,
      activation);
}

at::Tensor mixture_of_experts_fp8_fused_weights_scalars(
    const at::Tensor& hidden_states,
    const at::Tensor& expert_routing_table,
    const at::Tensor& router_weights,
    const at::TensorList w12,
    const at::TensorList w3,
    const double d_scale_hidden_states,
    const c10::ArrayRef<double>& d_scale_intermediate_hidden_states,
    const c10::ArrayRef<double>& d_scale_w12,
    const c10::ArrayRef<double>& d_scale_w3,
    const bool permuted_weights,
    const c10::string_view activation,
    const int64_t experts_min,
    const int64_t experts_max) {
  PT_EAGER_TRACE;
  PT_OP_INFO(
      "mixture_of_experts.fp8_fused_weights_scalars :",
      DUMP_13ARGS(
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
          experts_max));
  auto [w1, w2] = split_weights_tensor(w12, permuted_weights);
  return mixture_of_experts_fp8_common(
      hidden_states,
      expert_routing_table,
      router_weights,
      w1,
      w2,
      w3,
      d_scale_hidden_states,
      d_scale_intermediate_hidden_states,
      d_scale_w12,
      d_scale_w12,
      d_scale_w3,
      permuted_weights,
      activation);
}

at::Tensor mixture_of_experts_fwd_autograd(
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
  // In case of calling in compile without requiring gradients intermediate
  // tensors will be removed from the graph leading to error when trying to
  // store them. Therefore we call recomp-version that returns valid number of
  // tensors.
  return (recomp.value_or(false) || !hidden_states.requires_grad())
      ? MixtureOfExpertsRecompFunction::apply(
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

at::Tensor mixture_of_experts_fwd_fused_weights_autograd(
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
  return (recomp.value_or(false) || !hidden_states.requires_grad())
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

} // namespace eager
} // namespace habana
