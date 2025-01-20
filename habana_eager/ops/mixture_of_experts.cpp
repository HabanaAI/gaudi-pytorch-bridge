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
#include "habana_eager/ops/eager_op.h"
#include "habana_helpers/logging.h"
#include "hpu_ops/op_logger.h"

namespace habana {
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

at::Tensor mixture_of_experts_fwd_dispatch(
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
      "mixture_of_experts_fwd_dispatch :",
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
                       .typed<decltype(mixture_of_experts_fwd_dispatch)>();
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

at::Tensor mixture_of_experts_fwd_fused_weights_dispatch(
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
      "mixture_of_experts_fwd_fused_weights_dispatch :",
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
          .typed<decltype(mixture_of_experts_fwd_fused_weights_dispatch)>();
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

at::Tensor mixture_of_experts_bwd_dispatch(
    const at::Tensor& grad,
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
      "mixture_of_experts_bwd_dispatch :",
      DUMP_11ARGS(
          grad,
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
                       .findSchemaOrThrow("hpu::mixture_of_experts_bwd", "")
                       .typed<decltype(mixture_of_experts_bwd_dispatch)>();
  return op.call(
      grad,
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

at::Tensor mixture_of_experts_bwd_fused_weights_dispatch(
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
      "mixture_of_experts_bwd_fused_weights_dispatch :",
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
          .findSchemaOrThrow("hpu::mixture_of_experts_bwd", "fused_weights")
          .typed<decltype(mixture_of_experts_bwd_fused_weights_dispatch)>();
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

class MixtureOfExpertsFunction
    : public torch::autograd::Function<MixtureOfExpertsFunction> {
 public:
  static torch::autograd::variable_list forward(
      torch::autograd::AutogradContext* ctx,
      const torch::autograd::Variable& hidden_states,
      const torch::autograd::Variable& expert_routing_table,
      const torch::autograd::Variable& router_weights,
      const c10::ArrayRef<torch::autograd::Variable>& w1,
      const c10::ArrayRef<torch::autograd::Variable>& w2,
      const c10::ArrayRef<torch::autograd::Variable>& w3,
      const bool permuted_weights,
      const c10::string_view activation,
      const int64_t experts_min,
      const int64_t experts_max) {
    at::AutoDispatchBelowADInplaceOrView g;

    int64_t num_experts = w1.size();
    torch::autograd::variable_list to_save;
    to_save.reserve(non_list_tensors + weights_list_amount * num_experts);
    to_save.push_back(hidden_states);
    to_save.push_back(expert_routing_table);
    to_save.push_back(router_weights);
    to_save.insert(to_save.begin() + non_list_tensors, w1.begin(), w1.end());
    to_save.insert(
        to_save.begin() + non_list_tensors + num_experts, w2.begin(), w2.end());
    to_save.insert(
        to_save.begin() + non_list_tensors + 2 * num_experts,
        w3.begin(),
        w3.end());

    ctx->save_for_backward(to_save);
    ctx->saved_data["num_experts"] = num_experts;
    ctx->saved_data["permuted_weights"] = permuted_weights;
    ctx->saved_data["activation"] = activation;
    ctx->saved_data["experts_min"] = experts_min;
    ctx->saved_data["experts_max"] = experts_max;

    auto output = mixture_of_experts_fwd_dispatch(
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

    return {output};
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* ctx,
      const torch::autograd::variable_list& grads) {
    torch::autograd::variable_list saved_vars = ctx->get_saved_variables();
    auto num_experts = ctx->saved_data["num_experts"].toInt();
    auto w1 = torch::autograd::variable_list(
        saved_vars.begin() + non_list_tensors,
        saved_vars.begin() + non_list_tensors + num_experts);
    auto w2 = torch::autograd::variable_list(
        saved_vars.begin() + non_list_tensors + num_experts,
        saved_vars.begin() + non_list_tensors + 2 * num_experts);
    auto w3 = torch::autograd::variable_list(
        saved_vars.begin() + non_list_tensors + 2 * num_experts,
        saved_vars.begin() + non_list_tensors +
            weights_list_amount * num_experts);

    auto result = mixture_of_experts_bwd_dispatch(
        grads[0],
        saved_vars[0],
        saved_vars[1],
        saved_vars[2],
        w1,
        w2,
        w3,
        ctx->saved_data["permuted_weights"].toBool(),
        ctx->saved_data["activation"].toStringRef(),
        ctx->saved_data["experts_min"].toInt(),
        ctx->saved_data["experts_max"].toInt());

    torch::autograd::variable_list grad_input;
    grad_input.reserve(non_list_inputs + weights_list_amount * num_experts);
    for (int64_t i = 0; i < non_list_inputs + weights_list_amount * num_experts;
         i++) {
      grad_input.push_back(at::Tensor());
    }
    grad_input[0] = result;
    return grad_input;
  }

 private:
  static const int64_t weights_list_amount = 3;
  static const int64_t non_list_tensors = 3;
  static const int64_t non_list_inputs = 7;
};

class MixtureOfExpertsFusedWeightsFunction
    : public torch::autograd::Function<MixtureOfExpertsFusedWeightsFunction> {
 public:
  static torch::autograd::variable_list forward(
      torch::autograd::AutogradContext* ctx,
      const torch::autograd::Variable& hidden_states,
      const torch::autograd::Variable& expert_routing_table,
      const torch::autograd::Variable& router_weights,
      const c10::ArrayRef<torch::autograd::Variable>& w12,
      const c10::ArrayRef<torch::autograd::Variable>& w3,
      const bool permuted_weights,
      const c10::string_view activation,
      const int64_t experts_min,
      const int64_t experts_max) {
    at::AutoDispatchBelowADInplaceOrView g;

    int64_t num_experts = w12.size();
    torch::autograd::variable_list to_save;
    to_save.reserve(non_list_tensors + weights_list_amount * num_experts);
    to_save.push_back(hidden_states);
    to_save.push_back(expert_routing_table);
    to_save.push_back(router_weights);
    to_save.insert(to_save.begin() + non_list_tensors, w12.begin(), w12.end());
    to_save.insert(
        to_save.begin() + non_list_tensors + num_experts, w3.begin(), w3.end());
    ctx->save_for_backward(to_save);

    ctx->saved_data["num_experts"] = num_experts;
    ctx->saved_data["permuted_weights"] = permuted_weights;
    ctx->saved_data["activation"] = activation;
    ctx->saved_data["experts_min"] = experts_min;
    ctx->saved_data["experts_max"] = experts_max;

    auto output = mixture_of_experts_fwd_fused_weights_dispatch(
        hidden_states,
        expert_routing_table,
        router_weights,
        w12,
        w3,
        permuted_weights,
        activation,
        experts_min,
        experts_max);

    return {output};
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* ctx,
      const torch::autograd::variable_list& grads) {
    torch::autograd::variable_list saved_vars = ctx->get_saved_variables();

    int64_t num_experts = ctx->saved_data["num_experts"].toInt();
    auto w12 = torch::autograd::variable_list(
        saved_vars.begin() + non_list_tensors,
        saved_vars.begin() + non_list_tensors + num_experts);
    auto w3 = torch::autograd::variable_list(
        saved_vars.begin() + non_list_tensors + num_experts,
        saved_vars.begin() + non_list_tensors +
            weights_list_amount * num_experts);

    auto result = mixture_of_experts_bwd_fused_weights_dispatch(
        grads[0],
        saved_vars[0],
        saved_vars[1],
        saved_vars[2],
        w12,
        w3,
        ctx->saved_data["permuted_weights"].toBool(),
        ctx->saved_data["activation"].toStringRef(),
        ctx->saved_data["experts_min"].toInt(),
        ctx->saved_data["experts_max"].toInt());

    torch::autograd::variable_list grad_input;
    grad_input.reserve(non_list_inputs + weights_list_amount * num_experts);
    for (int64_t i = 0; i < non_list_inputs + weights_list_amount * num_experts;
         i++) {
      grad_input.push_back(at::Tensor());
    }
    grad_input[0] = result;
    return grad_input;
  }

 private:
  static const int64_t weights_list_amount = 2;
  static const int64_t non_list_tensors = 3;
  static const int64_t non_list_inputs = 7;
};

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
    const int64_t experts_max) {
  return MixtureOfExpertsFunction::apply(
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
    const int64_t experts_max) {
  return MixtureOfExpertsFusedWeightsFunction::apply(
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
