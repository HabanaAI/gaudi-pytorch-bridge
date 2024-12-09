/**
 * Copyright (c) 2024 Intel Corporation
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
#include "common/dump_args.h"
#include "habana_eager/ops/eager_op.h"
#include "habana_helpers/logging.h"
#include "hpu_ops/op_logger.h"

namespace habana {
namespace eager {

std::tuple<at::Tensor, at::Tensor> mixture_of_experts_common(
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
      "mixture_of_experts_common :",
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
  // experts_min/max are used by CGuid path only,
  // so they don't affect eager execution

  std::function<at::Tensor(const at::Tensor& x)> activation_fn;
  if (activation == "gelu") {
    activation_fn = [](const at::Tensor& x) {
      return torch::nn::functional::gelu(x);
    };
  } else if (activation == "relu") {
    activation_fn = [](const at::Tensor& x) {
      return torch::nn::functional::relu(x);
    };
  } else if (activation == "silu") {
    activation_fn = [](const at::Tensor& x) {
      return torch::nn::functional::silu(x);
    };
  }
  const int num_experts = w1.size();
  const int num_tokens = hidden_states.size(0);
  const int hidden_dim = hidden_states.size(1);
  auto final_hidden_states =
      torch::zeros({1, num_tokens, hidden_dim}, hidden_states.options());
  auto padded_weights =
      torch::zeros({num_tokens, num_experts}, hidden_states.options())
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
    const int64_t experts_max) {
  PT_EAGER_TRACE;
  PT_OP_INFO(
      "mixture_of_experts :",
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

  auto moe_common = mixture_of_experts_common(
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
    const int64_t experts_max) {
  PT_EAGER_TRACE;
  PT_OP_INFO(
      "mixture_of_experts.fused_weights :",
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

  std::vector<at::Tensor> w1, w2;
  const auto splitDim = permuted_weights ? 0 : 1;
  const auto splitIndex = w12[0].size(splitDim) / 2;
  for (const auto& tensor : w12) {
    auto w12_split = tensor.split(splitIndex, splitDim);
    w1.push_back(w12_split[0]);
    w2.push_back(w12_split[1]);
  }

  auto moe_common = mixture_of_experts_common(
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
      experts_min,
      experts_max,
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

  std::vector<at::Tensor> w1, w2;
  const auto splitDim = permuted_weights ? 0 : 1;
  const auto splitIndex = w12[0].size(splitDim) / 2;
  for (const auto& tensor : w12) {
    auto w12_split = tensor.split(splitIndex, splitDim);
    w1.push_back(w12_split[0]);
    w2.push_back(w12_split[1]);
  }

  return mixture_of_experts_common(
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
      measurement_mode);
}

} // namespace eager
} // namespace habana
