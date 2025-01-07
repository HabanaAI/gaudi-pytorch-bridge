/**
* Copyright (c) 2021-2024 Intel Corporation
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
#include "hpu_ops/stack_getter.h"
#include "perf_lib_layer_params.h"

namespace sh = synapse_helpers;

namespace habana {

class OptimizerFusedLarsOperator : public OpBackend {
 public:
  OptimizerFusedLarsOperator(int device_id, c10::ScalarType scalar_type)
      : OpBackend(
            device_id,
            NO_TPC + "optimizer_fused_LarsOperator_",
            scalar_type,
            {},
            {1}, // inplace ids
            {},
            false) {}

  void AddNode(sh::graph& graph, const at::Stack& stack) override;
};

static std::pair<synTensor, synTensor> NormalizeInput(
    OpBackend* op,
    sh::graph& graph,
    const TensorsPair& input,
    const synTensor zero_t,
    const std::vector<NodeAttr::NodeOutputAttr>& scalar_attr,
    const std::string& reduce_sum_sq_node,
    const std::string& sqrt_node,
    const std::string& greater_node,
    std::vector<sh::tensor>& storage) {
  synTensor input_t = input.syn_t;

  ns_Reduction::ParamsV2 reduce_params{};
  reduce_params.reductionDimensionMask = 0;
  reduce_params.keepDim = false;
  auto sum_sq = op->BuildNode(
      op,
      graph,
      {reduce_sum_sq_node,
       {input_t},
       scalar_attr,
       &reduce_params,
       sizeof(reduce_params)});

  auto sqrt_sum_sq =
      op->BuildNode(op, graph, {sqrt_node, {sum_sq[0].get()}, scalar_attr});

  auto greater = op->BuildNode(
      op, graph, {greater_node, {sqrt_sum_sq[0].get(), zero_t}, scalar_attr});

  storage.emplace_back(std::move(sqrt_sum_sq[0]));
  auto returned1 = storage.back().get();

  storage.emplace_back(std::move(greater[0]));
  auto returned2 = storage.back().get();

  return {returned1, returned2};
}

void OptimizerFusedLarsOperator::AddNode(
    sh::graph& graph,
    const at::Stack& stack) {
  StackGetter stackGetter(this, stack, "OptimizerFusedLarsOperator::AddNode");
  auto params = stackGetter.getNextInput<std::vector<TensorsPair>>();
  auto grads = stackGetter.getNextInput<std::vector<TensorsPair>>();
  auto skip_masks = stackGetter.getNextInput<std::vector<int64_t>>();
  auto eeta = stackGetter.getNextInput<double>();
  auto weight_decay = stackGetter.getNextInput<double>();
  auto eps = stackGetter.getNextInput<double>();
  auto lr = stackGetter.getNextInput<TensorsPair>();

  if ((params.size() != grads.size()) || (params.size() != skip_masks.size())) {
    std::stringstream ss;
    ss << "params, grads and skip_masks vector inputs must have the same number of elements but they respectively have: "
       << params.size() << ", " << grads.size() << ", " << skip_masks.size();
    AT_ERROR(ss.str());
  }

  std::string add_node = get_guid_with_precision("add_fwd", ScalarType());
  std::string mul_node = get_guid_with_precision("mult_fwd", ScalarType());
  std::string div_node = get_guid_with_precision("div_fwd", ScalarType());
  std::string reduce_sum_sq_node =
      get_guid_with_precision("reduce_sum_square_multi_dim_fwd", ScalarType());
  std::string sqrt_node = get_guid_with_precision("sqrt_fwd", ScalarType());
  std::string greater_node =
      get_guid_with_precision("greater_fwd", ScalarType());
  std::string where_node = get_guid_with_precision("where_fwd", ScalarType());

  double constant_values[] = {eeta, weight_decay, eps, 0.0, 1.0};
  std::array<synTensor, std::size(constant_values)> constant_ts{};
  const auto& eeta_t = constant_ts[0];
  const auto& weight_decay_t = constant_ts[1];
  const auto& eps_t = constant_ts[2];
  const auto& zero_t = constant_ts[3];
  const auto& one_t = constant_ts[4];

  int64_t skip_mask_ored = std::accumulate(
      skip_masks.begin(), skip_masks.end(), 0, std::bit_or<int64_t>());
  size_t num_constants = skip_mask_ored ? constant_ts.size() : 1;

  int64_t scalar_shape[] = {1};
  std::vector<sh::tensor> storage;
  storage.reserve(num_constants);
  for (size_t i = 0; i < num_constants; ++i) {
    storage.push_back(ConstantHelper(
        graph,
        static_cast<float>(constant_values[i]),
        ScalarType(),
        scalar_shape));
    constant_ts[i] = storage.back().get();
  }

  size_t vec_size = params.size();
  for (size_t i = 0; i < vec_size; ++i) {
    const auto& param = params[i];
    const auto& grad = grads[i];
    const auto& skip_mask = skip_masks[i];

    const auto& outshape = param.pt_t.sizes();
    const auto dtype = ScalarType();

    std::vector<NodeAttr::NodeOutputAttr> out_attr = {{outshape, dtype}};
    std::vector<NodeAttr::NodeOutputAttr> scalar_attr = {{scalar_shape, dtype}};

    if (!skip_mask) {
      auto mul = BuildOp(
          graph, mul_node, {grad.syn_t, lr.syn_t}, {{outshape, dtype, i}});
      syn_out(i) = std::move(mul[0]);
    } else {
      std::vector<sh::tensor> local_storage;
      local_storage.reserve(6);

#define NORMALIZE_INPUT_PARAMS(INPUT)                                     \
  this, graph, INPUT, zero_t, scalar_attr, reduce_sum_sq_node, sqrt_node, \
      greater_node, local_storage

      auto [param_norm_t, param_greater_t] =
          NormalizeInput(NORMALIZE_INPUT_PARAMS(param));

      auto [grad_norm_t, grad_greater_t] =
          NormalizeInput(NORMALIZE_INPUT_PARAMS(grad));

#undef NORMALIZE_INPUT_PARAMS

      auto pnorm_times_eeta =
          BuildOp(graph, mul_node, {param_norm_t, eeta_t}, scalar_attr);

      auto pnorm_times_wd =
          BuildOp(graph, mul_node, {param_norm_t, weight_decay_t}, scalar_attr);

      auto pnorm_times_wd_plus_eps = BuildOp(
          graph, add_node, {pnorm_times_wd[0].get(), eps_t}, scalar_attr);

      auto denom = BuildOp(
          graph,
          add_node,
          {pnorm_times_wd_plus_eps[0].get(), grad_norm_t},
          scalar_attr);

      auto div = BuildOp(
          graph,
          div_node,
          {pnorm_times_eeta[0].get(), denom[0].get()},
          scalar_attr);

      auto selected_div_part = BuildOp(
          graph,
          where_node,
          {grad_greater_t, div[0].get(), one_t},
          scalar_attr);

      auto selected_div = BuildOp(
          graph,
          where_node,
          {param_greater_t, selected_div_part[0].get(), one_t},
          scalar_attr);

      auto scaled_lr = BuildOp(
          graph, mul_node, {selected_div[0].get(), lr.syn_t}, scalar_attr);

      auto param_times_wd =
          BuildOp(graph, mul_node, {param.syn_t, weight_decay_t}, out_attr);

      auto param_times_wd_plus_grad = BuildOp(
          graph, add_node, {grad.syn_t, param_times_wd[0].get()}, out_attr);

      auto result = BuildOp(
          graph,
          mul_node,
          {param_times_wd_plus_grad[0].get(), scaled_lr[0].get()},
          {{outshape, dtype, i}});

      syn_out(i) = std::move(result[0]);
    }
  }
}

} // namespace habana

static auto& OptimizerKernelsKernelRegistry = habana::KernelRegistry().add(
    "hpu::optimizer_lars",
    KERNEL_FN(OptimizerFusedLarsOperator));
