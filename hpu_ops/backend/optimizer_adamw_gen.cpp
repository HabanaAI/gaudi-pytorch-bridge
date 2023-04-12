/*******************************************************************************
 * Copyright (C) 2023 Habana Labs, Ltd. an Intel Company
 * All Rights Reserved.
 *
 * Unauthorized copying of this file or any element(s) within it, via any medium
 * is strictly prohibited.
 * This file contains Habana Labs, Ltd. proprietary and confidential information
 * and is subject to the confidentiality and license agreements under which it
 * was provided.
 *
 *******************************************************************************
 */
#include "hpu_ops/op_backend.h"

namespace sh = synapse_helpers;

namespace habana {

class OptimizerFusedAdamWOperator : public OpBackend {
 public:
  OptimizerFusedAdamWOperator(int device_id, c10::ScalarType scalar_type)
      : OpBackend(
            device_id,
            NO_TPC + "optimizer_fused_AdamwOperator_",
            scalar_type,
            {},
            {1, 2, 3}, // inplace ids
            {},
            false) {}

  void AddNode(sh::graph& graph, const at::Stack& stack) override;
};

void OptimizerFusedAdamWOperator::AddNode(
    sh::graph& graph,
    const at::Stack& stack) {
  StackGetter stackGetter(stack, "OptimizerFusedAdamWOperator::AddNode");
  auto gradient_vec = getNextInput<std::vector<TensorsPair>>(stackGetter);
  auto weight_vec = getNextInput<std::vector<TensorsPair>>(stackGetter);
  auto exp_avg_vec = getNextInput<std::vector<TensorsPair>>(stackGetter);
  auto exp_avg_sq_vec = getNextInput<std::vector<TensorsPair>>(stackGetter);
  auto neg_step_t = getNextInput<TensorsPair>(stackGetter);
  auto beta1 = getNextInput<double>(stackGetter);
  auto beta2 = getNextInput<double>(stackGetter);
  auto epsilon = getNextInput<double>(stackGetter);
  auto weight_decay = getNextInput<TensorsPair>(stackGetter);
  auto has_weight_decay = getNextInput<bool>(stackGetter);

  if ((gradient_vec.size() != weight_vec.size()) ||
      (gradient_vec.size() != exp_avg_vec.size()) ||
      (gradient_vec.size() != exp_avg_sq_vec.size())) {
    std::stringstream ss;
    ss << "All 4 vector inputs must have the same number of elements but they respectively have: "
       << gradient_vec.size() << ", " << weight_vec.size() << ", "
       << exp_avg_vec.size() << ", " << exp_avg_sq_vec.size();
    AT_ERROR(ss.str());
  }

  std::string add_node = get_guid_with_precision("add_fwd", ScalarType());
  std::string mul_node = get_guid_with_precision("mult_fwd", ScalarType());
  std::string div_node = get_guid_with_precision("div_fwd", ScalarType());
  std::string sqrt_node = get_guid_with_precision("sqrt_fwd", ScalarType());

  int64_t scalar_shape[] = {1};

  double constant_values[] = {beta1, beta2, 1.0 - beta1, 1.0 - beta2, epsilon};
  std::array<synTensor, std::size(constant_values)> constant_ts{};
  const auto& beta1_t = constant_ts[0];
  const auto& beta2_t = constant_ts[1];
  const auto& one_minus_beta1_t = constant_ts[2];
  const auto& one_minus_beta2_t = constant_ts[3];
  const auto& epsilon_t = constant_ts[4];

  std::vector<sh::tensor> storage;
  storage.reserve(constant_ts.size() + 1);
  for (size_t i = 0; i < constant_ts.size(); ++i) {
    storage.push_back(ConstantHelper(
        graph,
        static_cast<float>(constant_values[i]),
        ScalarType(),
        scalar_shape));
    constant_ts[i] = storage.back().get();
  }

  size_t vec_size = gradient_vec.size();
  for (size_t i = 0; i < vec_size; ++i) {
    const auto& gradient = gradient_vec[i];
    const auto& weight = weight_vec[i];
    const auto& exp_avg = exp_avg_vec[i];
    const auto& exp_avg_sq = exp_avg_sq_vec[i];

    std::vector<NodeAttr::NodeOutputAttr> gradient_attr = {
        {gradient.pt_t.sizes(), ScalarType()}};
    std::vector<NodeAttr::NodeOutputAttr> weight_attr = {
        {weight.pt_t.sizes(), ScalarType()}};
    std::vector<NodeAttr::NodeOutputAttr> exp_avg_attr = {
        {exp_avg.pt_t.sizes(), ScalarType()}};
    std::vector<NodeAttr::NodeOutputAttr> exp_avg_sq_attr = {
        {exp_avg_sq.pt_t.sizes(), ScalarType()}};

    auto exp_avg_mul_beta1 =
        BuildOp(graph, mul_node, {exp_avg.syn_t, beta1_t}, exp_avg_attr);

    auto grad_scaled = BuildOp(
        graph, mul_node, {gradient.syn_t, one_minus_beta1_t}, gradient_attr);

    auto exp_avg_1 = BuildOp(
        graph,
        add_node,
        {exp_avg_mul_beta1[0].get(), grad_scaled[0].get()},
        {{gradient.pt_t.sizes(), ScalarType()}});

    auto exp_avg_1_out = IdentityHelper(
        graph,
        exp_avg_1[0].get(),
        gradient.pt_t.sizes(),
        ScalarType(),
        i + vec_size);

    auto grad_sq = BuildOp(
        graph, mul_node, {gradient.syn_t, gradient.syn_t}, gradient_attr);

    auto grad_sq_scaled = BuildOp(
        graph, mul_node, {grad_sq[0].get(), one_minus_beta2_t}, gradient_attr);

    auto exp_avg_sq_mul_beta2 =
        BuildOp(graph, mul_node, {exp_avg_sq.syn_t, beta2_t}, exp_avg_sq_attr);

    auto exp_avg_sq_1 = BuildOp(
        graph,
        add_node,
        {exp_avg_sq_mul_beta2[0].get(), grad_sq_scaled[0].get()},
        {NodeAttr::NodeOutputAttr{gradient.pt_t.sizes(), ScalarType()}});

    auto exp_avg_sq_1_out = IdentityHelper(
        graph,
        exp_avg_sq_1[0].get(),
        gradient.pt_t.sizes(),
        ScalarType(),
        i + 2 * vec_size);

    auto exp_avg_sq_sqrt =
        BuildOp(graph, sqrt_node, {exp_avg_sq_1[0].get()}, exp_avg_sq_attr);

    auto denom = BuildOp(
        graph,
        add_node,
        {exp_avg_sq_sqrt[0].get(), epsilon_t},
        exp_avg_sq_attr);

    auto ratio = BuildOp(
        graph, div_node, {exp_avg_1[0].get(), denom[0].get()}, gradient_attr);

    auto scaled_ratio = BuildOp(
        graph, mul_node, {ratio[0].get(), neg_step_t.syn_t}, gradient_attr);

    auto weight_modified = weight.syn_t;
    if (has_weight_decay) {
      storage.push_back(std::move(BuildOp(
          graph,
          mul_node,
          {weight_modified, weight_decay.syn_t},
          weight_attr)[0]));
      weight_modified = storage.back().get();
    }

    auto result = BuildOp(
        graph,
        add_node,
        {weight_modified, scaled_ratio[0].get()},
        {NodeAttr::NodeOutputAttr{weight.pt_t.sizes(), ScalarType(), i}});

    syn_out(i) = std::move(result[0]);
    syn_out(i + vec_size) = std::move(exp_avg_1_out);
    syn_out(i + 2 * vec_size) = std::move(exp_avg_sq_1_out);
  }
}

} // namespace habana

static auto& OptimizerKernelsKernelRegistry = habana::KernelRegistry().add(
    "hpu::optimizer_adamw",
    KERNEL_FN(OptimizerFusedAdamWOperator));