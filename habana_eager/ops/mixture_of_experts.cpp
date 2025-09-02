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
#include "generated/autograd/autograd_ops.h"
#include "generated/backend/cast_from_fp8.h"
#include "generated/backend/cast_to_fp8_v2.h"
#include "generated/backend/fp8_gemm_v2.h"
#include "habana_eager/ops/eager_op.h"
#include "habana_helpers/logging.h"
#include "hpu_ops/common/mixture_of_experts.h"
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
    const std::string_view activation,
    const int64_t experts_min,
    const int64_t experts_max,
    const int64_t chunk_size,
    const int64_t total_experts) {
  return mixture_of_experts_fwd_dispatch(
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
      total_experts);
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
  return mixture_of_experts_recomp_fwd_dispatch(
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
      total_experts);
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
  return mixture_of_experts_fwd_fused_weights_dispatch(
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
      total_experts);
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
  return mixture_of_experts_recomp_fwd_fused_weights_dispatch(
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
      total_experts);
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
  PT_EAGER_TRACE;
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

  static auto op = torch::Dispatcher::singleton()
                       .findSchemaOrThrow("hpu::mixture_of_experts_bwd", "")
                       .typed<decltype(mixture_of_experts_bwd)>();
  return op.call(
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
      router_weights_size);
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
  PT_EAGER_TRACE;
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
      experts_max,
      chunk_size,
      total_experts);
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
  PT_EAGER_TRACE;
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

  static auto op =
      torch::Dispatcher::singleton()
          .findSchemaOrThrow("hpu::mixture_of_experts_bwd", "fused_weights")
          .typed<decltype(mixture_of_experts_bwd_fused_weights)>();

  return op.call(
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
      router_weights_size);
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
  PT_EAGER_TRACE;
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
      experts_max,
      chunk_size,
      total_experts);
}
namespace eager {

static std::pair<std::vector<at::Tensor>, std::vector<at::Tensor>>
split_weights_tensor(
    const at::TensorList& w12,
    bool permuted_weights,
    bool unsqueeze = false) {
  std::vector<at::Tensor> w1;
  std::vector<at::Tensor> w2;
  const auto split_dim = (unsqueeze or permuted_weights) ? 0 : 1;
  const auto split_index = w12[0].size(split_dim) / 2;
  for (const auto& tensor : w12) {
    auto w12_split = unsqueeze
        ? tensor.unsqueeze(0).split(w12[0].size(0) / 2, 1)
        : tensor.split(split_index, split_dim);
    w1.push_back(w12_split[0]);
    w2.push_back(w12_split[1]);
  }
  return {w1, w2};
}

std::function<at::Tensor(const at::Tensor& x)> get_activation_fn(
    const std::string_view& activation) {
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
    const std::string_view activation,
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

static std::pair<std::vector<at::Tensor>, std::vector<at::Tensor>>
prepare_expert_weights(
    const at::TensorList& w12,
    const at::TensorList& w3,
    const bool permuted_weights) {
  const size_t num_experts = w12.size();
  std::vector<at::Tensor> w12_transposed;
  std::vector<at::Tensor> w3_transposed;

  w12_transposed.reserve(num_experts);
  w3_transposed.reserve(num_experts);

  for (size_t i = 0; i < num_experts; ++i) {
    w12_transposed.emplace_back(
        permuted_weights ? w12[i].transpose(0, 1) : w12[i]);
    w3_transposed.emplace_back(
        permuted_weights ? w3[i].transpose(0, 1) : w3[i]);
  }
  return {std::move(w12_transposed), std::move(w3_transposed)};
}

static std::pair<at::Tensor, at::Tensor> split_gate_and_up(
    const at::Tensor& gate_up,
    const float limit) {
  at::Tensor gate =
      gate_up.index({"...", at::indexing::Slice(0, at::indexing::None, 2)});
  at::Tensor up =
      gate_up.index({"...", at::indexing::Slice(1, at::indexing::None, 2)});
  gate = torch::clamp(gate, std::nullopt, limit);
  up = torch::clamp(up, -limit, limit);

  return {gate, up};
}

static at::Tensor prepare_routing_weights(
    const at::Tensor& expert_routing_table,
    const at::Tensor& router_weights,
    const int64_t num_experts,
    const int64_t num_tokens) {
  auto routing_weights_scattered =
      torch::zeros({num_tokens, num_experts}, router_weights.options());
  routing_weights_scattered.scatter_(1, expert_routing_table, router_weights);

  return routing_weights_scattered.transpose(0, 1)
      .view({num_experts, -1})
      .unsqueeze(-1);
}

static at::Tensor calculate_amax(
    const at::Tensor& input,
    const at::Tensor& routing_weights_all,
    const std::optional<bool>& measure_per_token) {
  if (!measure_per_token.has_value()) {
    return {};
  }

  at::Tensor mask = routing_weights_all != 0;
  at::Tensor amax =
      std::get<0>(torch::max(torch::abs(input.to(torch::kFloat32)), -1));
  amax = amax * mask.squeeze(-1);

  if (!measure_per_token.value()) {
    amax = std::get<0>(torch::max(amax, -1));
  }

  return amax;
}

static std::tuple<at::Tensor, at::Tensor> mixture_of_experts_common_with_bias(
    const at::Tensor& hidden_states,
    const at::Tensor& expert_routing_table,
    const at::Tensor& router_weights,
    const at::TensorList w12,
    const at::TensorList w12_bias,
    const at::TensorList w3,
    const at::TensorList w3_bias,
    bool permuted_weights,
    std::optional<bool> measure_per_token,
    double alpha,
    double limit) {
  TORCH_CHECK(!w12.empty(), "Number of experts must be greater than zero");
  TORCH_CHECK(
      w12.size() == w12_bias.size(),
      "w12 and w12_bias must have the same number of experts");
  TORCH_CHECK(
      w3.size() == w3_bias.size(),
      "w3 and w3_bias must have the same number of experts");
  TORCH_CHECK(
      w12.size() == w3.size(),
      "w12 and w3 must have the same number of experts");

  auto [w12_maybe_transposed, w3_maybe_transposed] =
      prepare_expert_weights(w12, w3, permuted_weights);

  const int64_t hidden_size = w3_maybe_transposed[0].size(1);
  const int64_t num_experts = w12.size();

  at::Tensor w12_stacked = torch::stack(w12_maybe_transposed, 0);
  at::Tensor w12_bias_stacked = torch::stack(w12_bias, 0);
  at::Tensor w3_stacked = torch::stack(w3_maybe_transposed, 0);
  at::Tensor w3_bias_stacked = torch::stack(w3_bias, 0);

  at::Tensor hidden_states_repeated = hidden_states.repeat({num_experts, 1});
  hidden_states_repeated =
      hidden_states_repeated.view({num_experts, -1, hidden_size});

  at::Tensor gate_up = torch::bmm(hidden_states_repeated, w12_stacked) +
      w12_bias_stacked.unsqueeze(-2);

  auto [gate, up] = split_gate_and_up(gate_up, limit);

  at::Tensor glu = gate * torch::sigmoid(gate * alpha);

  at::Tensor hidden_states_w12 = (up + 1) * glu;
  at::Tensor next_states = torch::bmm(hidden_states_w12, w3_stacked);
  next_states = next_states + w3_bias_stacked.unsqueeze(-2);
  next_states = next_states.view({num_experts, -1, hidden_size});

  const at::Tensor routing_weights_scattered = prepare_routing_weights(
      expert_routing_table, router_weights, num_experts, hidden_states.size(0));

  next_states = next_states * routing_weights_scattered;

  return {
      next_states.sum(0),
      calculate_amax(
          hidden_states_w12, routing_weights_scattered, measure_per_token)};
}

at::Tensor mixture_of_experts(
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
    const std::optional<bool> recomp,
    const int64_t chunk_size,
    const int64_t total_experts) {
  PT_EAGER_TRACE;
  PT_OP_INFO(
      "mixture_of_experts :",
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
          recomp,
          chunk_size,
          total_experts));

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
    const std::string_view activation,
    const int64_t experts_min,
    const int64_t experts_max,
    const std::optional<bool> recomp,
    const int64_t chunk_size,
    const int64_t total_experts) {
  PT_EAGER_TRACE;
  PT_OP_INFO(
      "mixture_of_experts.fused_weights :",
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
          recomp,
          chunk_size,
          total_experts));

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

at::Tensor mixture_of_experts_bias_fused_weights(
    const at::Tensor& hidden_states,
    const at::Tensor& expert_routing_table,
    const at::Tensor& router_weights,
    const at::TensorList w12,
    const at::TensorList w3,
    const at::TensorList w12_bias,
    const at::TensorList w3_bias,
    const bool permuted_weights,
    const int64_t experts_min,
    const int64_t experts_max,
    const int64_t chunk_size,
    const int64_t total_experts,
    const double alpha,
    const double limit) {
  PT_EAGER_TRACE;
  PT_OP_INFO(
      "mixture_of_experts.bias_fused_weights :",
      DUMP_14ARGS(
          hidden_states,
          expert_routing_table,
          router_weights,
          w12,
          w3,
          w12_bias,
          w3_bias,
          permuted_weights,
          experts_min,
          experts_max,
          chunk_size,
          total_experts,
          alpha,
          limit));

  return std::get<0>(mixture_of_experts_common_with_bias(
      hidden_states,
      expert_routing_table,
      router_weights,
      w12,
      w12_bias,
      w3,
      w3_bias,
      permuted_weights,
      std::nullopt,
      alpha,
      limit));
}

std::tuple<at::Tensor, at::Tensor> mixture_of_experts_fp8_measurement(
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
  PT_EAGER_TRACE;
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
    const std::string_view activation,
    const int64_t experts_min,
    const int64_t experts_max,
    const bool measurement_mode,
    const int64_t chunk_size,
    const int64_t total_experts) {
  PT_EAGER_TRACE;
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

std::tuple<at::Tensor, at::Tensor>
mixture_of_experts_measurement_bias_fused_weights(
    const at::Tensor& hidden_states,
    const at::Tensor& expert_routing_table,
    const at::Tensor& router_weights,
    const at::TensorList w12,
    const at::TensorList w3,
    const at::TensorList w12_bias,
    const at::TensorList w3_bias,
    const bool permuted_weights,
    const int64_t experts_min,
    const int64_t experts_max,
    const bool measure_per_token,
    const int64_t chunk_size,
    const int64_t total_experts,
    const double alpha,
    const double limit) {
  PT_EAGER_TRACE;
  PT_OP_INFO(
      "mixture_of_experts.measurement_bias_fused_weights :",
      DUMP_15ARGS(
          hidden_states,
          expert_routing_table,
          router_weights,
          w12,
          w3,
          w12_bias,
          w3_bias,
          permuted_weights,
          experts_min,
          experts_max,
          measure_per_token,
          chunk_size,
          total_experts,
          alpha,
          limit));

  return mixture_of_experts_common_with_bias(
      hidden_states,
      expert_routing_table,
      router_weights,
      w12,
      w12_bias,
      w3,
      w3_bias,
      permuted_weights,
      measure_per_token,
      alpha,
      limit);
}

template <typename Scale>
static at::Tensor moe_cast_to_fp8_v2(
    const at::Tensor& input,
    const Scale& scale,
    at::ScalarType dtype) {
  if constexpr (std::is_same_v<Scale, double>) {
    return std::get<0>(
        cast_to_fp8_v2_scalar(input, scale, false, false, dtype, std::nullopt));
  } else {
    return std::get<0>(
        cast_to_fp8_v2(input, scale, false, false, dtype, std::nullopt));
  }
}

template <typename Scale>
static at::Tensor moe_fp8_gemm_v2(
    const at::Tensor& A,
    const at::Tensor& B,
    at::ScalarType out_dtype,
    const Scale& A_scale_inv,
    const Scale& B_scale_inv) {
  if constexpr (std::is_same_v<Scale, double>) {
    return fp8_gemm_v2_scalar(
        A,
        false,
        B,
        false,
        std::nullopt,
        out_dtype,
        A_scale_inv,
        B_scale_inv,
        std::nullopt,
        false,
        std::nullopt);
  } else {
    return fp8_gemm_v2(
        A,
        false,
        B,
        false,
        std::nullopt,
        out_dtype,
        A_scale_inv,
        B_scale_inv,
        std::nullopt,
        false,
        std::nullopt);
  }
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
    const std::string_view activation) {
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
  if constexpr (std::is_same_v<Scale, double>) {
    default_scale = 1.0;
  } else {
    default_scale = at::tensor(
        1.0, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kHPU));
    if (d_scale_hidden_states.dim() == 1) {
      const_cast<at::Tensor&>(d_scale_hidden_states) =
          d_scale_hidden_states.unsqueeze(1);
    }
  }

  for (int expert_idx = 0; expert_idx < num_experts; expert_idx++) {
    const at::Tensor current_expert_w1 =
        permuted_weights ? w1[expert_idx].transpose(0, 1) : w1[expert_idx];
    const at::Tensor current_expert_w2 =
        permuted_weights ? w2[expert_idx].transpose(0, 1) : w2[expert_idx];
    const at::Tensor current_expert_w3 =
        permuted_weights ? w3[expert_idx].transpose(0, 1) : w3[expert_idx];

    auto hidden_states_w1 = activation_fn(moe_fp8_gemm_v2(
        hidden_states,
        current_expert_w1,
        torch::kBFloat16,
        d_scale_hidden_states,
        d_scale_w1[expert_idx]));

    auto hidden_states_w2 = moe_fp8_gemm_v2(
        hidden_states,
        current_expert_w2,
        torch::kBFloat16,
        d_scale_hidden_states,
        d_scale_w2[expert_idx]);

    auto hidden_states_w12 = hidden_states_w1 * hidden_states_w2;

    hidden_states_w12 = moe_cast_to_fp8_v2(
        hidden_states_w12,
        d_scale_intermediate_hidden_states[expert_idx],
        fp8_type);

    auto hidden_states_w3 = moe_fp8_gemm_v2(
        hidden_states_w12,
        current_expert_w3,
        torch::kBFloat16,
        default_scale,
        d_scale_w3[expert_idx]);

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
    const std::string_view activation,
    const int64_t experts_min,
    const int64_t experts_max,
    const int64_t chunk_size,
    const int64_t total_experts) {
  PT_EAGER_TRACE;
  PT_OP_INFO(
      "mixture_of_experts.fp8 :",
      DUMP_17ARGS(
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
          experts_max,
          chunk_size,
          total_experts));
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
    const std::string_view activation,
    const int64_t experts_min,
    const int64_t experts_max,
    const int64_t chunk_size,
    const int64_t total_experts) {
  PT_EAGER_TRACE;
  PT_OP_INFO(
      "mixture_of_experts.fp8_fused_weights :",
      DUMP_15ARGS(
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
          chunk_size,
          total_experts));
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
    const std::string_view activation,
    const int64_t experts_min,
    const int64_t experts_max,
    const int64_t chunk_size,
    const int64_t total_experts) {
  PT_EAGER_TRACE;
  PT_OP_INFO(
      "mixture_of_experts.fp8_scalars :",
      DUMP_17ARGS(
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
          experts_max,
          chunk_size,
          total_experts));
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
    const std::string_view activation,
    const int64_t experts_min,
    const int64_t experts_max,
    const int64_t chunk_size,
    const int64_t total_experts) {
  PT_EAGER_TRACE;
  PT_OP_INFO(
      "mixture_of_experts.fp8_fused_weights_scalars :",
      DUMP_15ARGS(
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
          chunk_size,
          total_experts));
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

template <typename Scale, typename Scales>
static at::Tensor mixture_of_experts_fp8_common_dynamic(
    const at::Tensor& hidden_states,
    const at::Tensor& expert_routing_table,
    const at::Tensor& router_weights,
    const at::TensorList& w1,
    const at::TensorList& w2,
    const at::TensorList& w3,
    const Scale& d_scale_hidden_states,
    const Scales& d_scale_w1,
    const Scales& d_scale_w2,
    const Scales& d_scale_w3,
    const bool permuted_weights,
    const std::string_view activation) {
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
  if constexpr (std::is_same_v<Scale, double>) {
    default_scale = 1.0;
  } else {
    default_scale = at::tensor(
        1.0, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kHPU));
    if (d_scale_hidden_states.dim() == 1) {
      const_cast<at::Tensor&>(d_scale_hidden_states) =
          d_scale_hidden_states.unsqueeze(1);
    }
  }

  for (int expert_idx = 0; expert_idx < num_experts; expert_idx++) {
    const at::Tensor current_expert_w1 =
        permuted_weights ? w1[expert_idx].transpose(0, 1) : w1[expert_idx];
    const at::Tensor current_expert_w2 =
        permuted_weights ? w2[expert_idx].transpose(0, 1) : w2[expert_idx];
    const at::Tensor current_expert_w3 =
        permuted_weights ? w3[expert_idx].transpose(0, 1) : w3[expert_idx];

    auto hidden_states_w1 = activation_fn(moe_fp8_gemm_v2(
        hidden_states,
        current_expert_w1,
        torch::kBFloat16,
        d_scale_hidden_states,
        d_scale_w1[expert_idx]));

    auto hidden_states_w2 = moe_fp8_gemm_v2(
        hidden_states,
        current_expert_w2,
        torch::kBFloat16,
        d_scale_hidden_states,
        d_scale_w2[expert_idx]);

    auto hidden_states_w12 = hidden_states_w1 * hidden_states_w2;
    at::Tensor hidden_states_w3;

    const auto is_gaudi2 =
        HPUDeviceContext::get_device().type() == synDeviceGaudi2;
    const auto scaling_factor = is_gaudi2 ? 240 : 448;
    const auto max_values = std::get<0>(torch::abs(hidden_states_w12).max(1));

    auto calculated_dynamic_scale =
        ((max_values + 1e-8) / scaling_factor).unsqueeze(-1);

    hidden_states_w12 = std::get<0>(cast_to_fp8_v2(
        hidden_states_w12,
        calculated_dynamic_scale,
        false,
        false,
        fp8_type,
        std::nullopt));

    at::Tensor current_d_scale_w3;
    if constexpr (std::is_same_v<Scale, double>) {
      current_d_scale_w3 = at::tensor(
          d_scale_w3[expert_idx],
          torch::TensorOptions().dtype(torch::kFloat32).device(torch::kHPU));
    } else {
      current_d_scale_w3 = d_scale_w3[expert_idx];
    }
    hidden_states_w3 = fp8_gemm_v2(
        hidden_states_w12,
        false,
        current_expert_w3,
        false,
        std::nullopt,
        torch::kBFloat16,
        std::nullopt,
        current_d_scale_w3,
        std::nullopt,
        false,
        std::nullopt);

    final_hidden_states += hidden_states_w3 * padded_weights[expert_idx];
  }

  auto result = final_hidden_states.reshape(hidden_states.sizes());
  return result;
}

at::Tensor mixture_of_experts_fp8_dynamic(
    const at::Tensor& hidden_states,
    const at::Tensor& expert_routing_table,
    const at::Tensor& router_weights,
    const at::TensorList w1,
    const at::TensorList w2,
    const at::TensorList w3,
    const at::Tensor& d_scale_hidden_states,
    const at::TensorList d_scale_w1,
    const at::TensorList d_scale_w2,
    const at::TensorList d_scale_w3,
    const bool permuted_weights,
    const std::string_view activation,
    const int64_t experts_min,
    const int64_t experts_max,
    const int64_t chunk_size,
    const int64_t total_experts) {
  PT_EAGER_TRACE;
  PT_OP_INFO(
      "mixture_of_experts.fp8 :",
      DUMP_16ARGS(
          hidden_states,
          expert_routing_table,
          router_weights,
          w1,
          w2,
          w3,
          d_scale_hidden_states,
          d_scale_w1,
          d_scale_w2,
          d_scale_w3,
          permuted_weights,
          activation,
          experts_min,
          experts_max,
          chunk_size,
          total_experts));
  return mixture_of_experts_fp8_common_dynamic(
      hidden_states,
      expert_routing_table,
      router_weights,
      w1,
      w2,
      w3,
      d_scale_hidden_states,
      d_scale_w1,
      d_scale_w2,
      d_scale_w3,
      permuted_weights,
      activation);
}

at::Tensor mixture_of_experts_fp8_fused_weights_dynamic(
    const at::Tensor& hidden_states,
    const at::Tensor& expert_routing_table,
    const at::Tensor& router_weights,
    const at::TensorList w12,
    const at::TensorList w3,
    const at::Tensor& d_scale_hidden_states,
    const at::TensorList d_scale_w12,
    const at::TensorList d_scale_w3,
    const bool permuted_weights,
    const std::string_view activation,
    const int64_t experts_min,
    const int64_t experts_max,
    const int64_t chunk_size,
    const int64_t total_experts) {
  PT_EAGER_TRACE;
  PT_OP_INFO(
      "mixture_of_experts.fp8_fused_weights :",
      DUMP_14ARGS(
          hidden_states,
          expert_routing_table,
          router_weights,
          w12,
          w3,
          d_scale_hidden_states,
          d_scale_w12,
          d_scale_w3,
          permuted_weights,
          activation,
          experts_min,
          experts_max,
          chunk_size,
          total_experts));
  auto [w1, w2] = split_weights_tensor(w12, permuted_weights);

  at::TensorList d_scale_w1 = d_scale_w12;
  at::TensorList d_scale_w2 = d_scale_w12;
  std::vector<at::Tensor> d_scale_w1_vec;
  std::vector<at::Tensor> d_scale_w2_vec;

  if (d_scale_w12[0].dim() != 0) {
    const auto unsqueeze = d_scale_w12[0].dim() == 1;
    std::tie(d_scale_w1_vec, d_scale_w2_vec) =
        split_weights_tensor(d_scale_w12, false, unsqueeze);
    d_scale_w1 = at::TensorList(d_scale_w1_vec);
    d_scale_w2 = at::TensorList(d_scale_w2_vec);
  }

  return mixture_of_experts_fp8_common_dynamic(
      hidden_states,
      expert_routing_table,
      router_weights,
      w1,
      w2,
      w3,
      d_scale_hidden_states,
      d_scale_w1,
      d_scale_w2,
      d_scale_w3,
      permuted_weights,
      activation);
}

at::Tensor mixture_of_experts_fp8_scalars_dynamic(
    const at::Tensor& hidden_states,
    const at::Tensor& expert_routing_table,
    const at::Tensor& router_weights,
    const at::TensorList w1,
    const at::TensorList w2,
    const at::TensorList w3,
    const double d_scale_hidden_states,
    const c10::ArrayRef<double>& d_scale_w1,
    const c10::ArrayRef<double>& d_scale_w2,
    const c10::ArrayRef<double>& d_scale_w3,
    const bool permuted_weights,
    const std::string_view activation,
    const int64_t experts_min,
    const int64_t experts_max,
    const int64_t chunk_size,
    const int64_t total_experts) {
  PT_EAGER_TRACE;
  PT_OP_INFO(
      "mixture_of_experts.fp8_scalars :",
      DUMP_16ARGS(
          hidden_states,
          expert_routing_table,
          router_weights,
          w1,
          w2,
          w3,
          d_scale_hidden_states,
          d_scale_w1,
          d_scale_w2,
          d_scale_w3,
          permuted_weights,
          activation,
          experts_min,
          experts_max,
          chunk_size,
          total_experts));
  return mixture_of_experts_fp8_common_dynamic(
      hidden_states,
      expert_routing_table,
      router_weights,
      w1,
      w2,
      w3,
      d_scale_hidden_states,
      d_scale_w1,
      d_scale_w2,
      d_scale_w3,
      permuted_weights,
      activation);
}

at::Tensor mixture_of_experts_fp8_fused_weights_scalars_dynamic(
    const at::Tensor& hidden_states,
    const at::Tensor& expert_routing_table,
    const at::Tensor& router_weights,
    const at::TensorList w12,
    const at::TensorList w3,
    const double d_scale_hidden_states,
    const c10::ArrayRef<double>& d_scale_w12,
    const c10::ArrayRef<double>& d_scale_w3,
    const bool permuted_weights,
    const std::string_view activation,
    const int64_t experts_min,
    const int64_t experts_max,
    const int64_t chunk_size,
    const int64_t total_experts) {
  PT_EAGER_TRACE;
  PT_OP_INFO(
      "mixture_of_experts.fp8_fused_weights_scalars :",
      DUMP_14ARGS(
          hidden_states,
          expert_routing_table,
          router_weights,
          w12,
          w3,
          d_scale_hidden_states,
          d_scale_w12,
          d_scale_w3,
          permuted_weights,
          activation,
          experts_min,
          experts_max,
          chunk_size,
          total_experts));
  auto [w1, w2] = split_weights_tensor(w12, permuted_weights);
  return mixture_of_experts_fp8_common_dynamic(
      hidden_states,
      expert_routing_table,
      router_weights,
      w1,
      w2,
      w3,
      d_scale_hidden_states,
      d_scale_w12,
      d_scale_w12,
      d_scale_w3,
      permuted_weights,
      activation);
}

// Broadcast scale with given block_size and crop it for uneven dims (padding)
at::Tensor broadcast_scales(
    const at::Tensor& scales,
    const int64_t block_size,
    const at::IntArrayRef& sizes) {
  return scales.repeat_interleave(block_size, 0)
      .repeat_interleave(block_size, 1)
      .slice(0, 0, sizes[0])
      .slice(1, 0, sizes[1]);
}

at::Tensor mixture_of_experts_fp8_blockwise(
    const at::Tensor& hidden_states,
    const at::Tensor& expert_routing_table,
    const at::Tensor& router_weights,
    const at::TensorList w1,
    const at::TensorList w2,
    const at::TensorList w3,
    const at::TensorList d_scale_w1,
    const at::TensorList d_scale_w2,
    const at::TensorList d_scale_w3,
    const int64_t block_size,
    const bool permuted_weights,
    const std::string_view activation,
    const int64_t experts_min,
    const int64_t experts_max,
    const int64_t chunk_size,
    const int64_t total_experts) {
  PT_EAGER_TRACE;
  PT_OP_INFO(
      "mixture_of_experts.fp8_blockwise :",
      DUMP_16ARGS(
          hidden_states,
          expert_routing_table,
          router_weights,
          w1,
          w2,
          w3,
          d_scale_w1,
          d_scale_w2,
          d_scale_w3,
          block_size,
          permuted_weights,
          activation,
          experts_min,
          experts_max,
          chunk_size,
          total_experts));

  c10::ScalarType scales_dtype = d_scale_w1[0].scalar_type();
  std::vector<at::Tensor> dequant_w1_vec;
  std::vector<at::Tensor> dequant_w2_vec;
  std::vector<at::Tensor> dequant_w3_vec;
  for (size_t i = 0; i < w1.size(); i++) {
    dequant_w1_vec.push_back(
        cast_from_fp8_scalar(w1[i], 1.0, scales_dtype, std::nullopt) *
        broadcast_scales(d_scale_w1[i], block_size, w1[0].sizes()));
    dequant_w2_vec.push_back(
        cast_from_fp8_scalar(w2[i], 1.0, scales_dtype, std::nullopt) *
        broadcast_scales(d_scale_w2[i], block_size, w2[0].sizes()));
    dequant_w3_vec.push_back(
        cast_from_fp8_scalar(w3[i], 1.0, scales_dtype, std::nullopt) *
        broadcast_scales(d_scale_w3[i], block_size, w3[0].sizes()));
  }

  const at::TensorList dequant_w1 = dequant_w1_vec;
  const at::TensorList dequant_w2 = dequant_w2_vec;
  const at::TensorList dequant_w3 = dequant_w3_vec;

  auto moe_common = mixture_of_experts_common(
      hidden_states,
      expert_routing_table,
      router_weights,
      dequant_w1,
      dequant_w2,
      dequant_w3,
      permuted_weights,
      activation,
      false);

  return std::get<0>(moe_common);
}

at::Tensor mixture_of_experts_fp8_fused_weights_blockwise(
    const at::Tensor& hidden_states,
    const at::Tensor& expert_routing_table,
    const at::Tensor& router_weights,
    const at::TensorList w12,
    const at::TensorList w3,
    const at::TensorList d_scale_w12,
    const at::TensorList d_scale_w3,
    const int64_t block_size,
    const bool permuted_weights,
    const std::string_view activation,
    const int64_t experts_min,
    const int64_t experts_max,
    const int64_t chunk_size,
    const int64_t total_experts) {
  PT_EAGER_TRACE;
  PT_OP_INFO(
      "mixture_of_experts.fp8_fused_weights_blockwise :",
      DUMP_14ARGS(
          hidden_states,
          expert_routing_table,
          router_weights,
          w12,
          w3,
          d_scale_w12,
          d_scale_w3,
          block_size,
          permuted_weights,
          activation,
          experts_min,
          experts_max,
          chunk_size,
          total_experts));
  // Fused weights flavor needs individual frontend, as in many cases padding
  // for w1 and w2 might be different than padding for w12
  c10::ScalarType scales_dtype = d_scale_w12[0].scalar_type();
  std::vector<at::Tensor> dequant_w12_vec;
  std::vector<at::Tensor> dequant_w3_vec;
  for (size_t i = 0; i < w12.size(); i++) {
    dequant_w12_vec.push_back(
        cast_from_fp8_scalar(w12[i], 1.0, scales_dtype, std::nullopt) *
        broadcast_scales(d_scale_w12[i], block_size, w12[0].sizes()));
    dequant_w3_vec.push_back(
        cast_from_fp8_scalar(w3[i], 1.0, scales_dtype, std::nullopt) *
        broadcast_scales(d_scale_w3[i], block_size, w3[0].sizes()));
  }

  const at::TensorList dequant_w12 = dequant_w12_vec;
  const at::TensorList dequant_w3 = dequant_w3_vec;

  auto [dequant_w1, dequant_w2] =
      split_weights_tensor(dequant_w12, permuted_weights);

  auto moe_common = mixture_of_experts_common(
      hidden_states,
      expert_routing_table,
      router_weights,
      dequant_w1,
      dequant_w2,
      dequant_w3,
      permuted_weights,
      activation,
      false);
  return std::get<0>(moe_common);
}

} // namespace eager
} // namespace habana
