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
#include <perf_lib_layer_params.h>
#include "hpu_ops/op_backend.h"

namespace sh = synapse_helpers;

namespace habana {

class OptimizerFusedSGDOperator : public OpBackend {
 public:
  OptimizerFusedSGDOperator(int device_id, c10::ScalarType scalar_type)
      : OpBackend(
            device_id,
            "optimizer_sgd_",
            scalar_type,
            {},
            {1}, // inplace ids
            {},
            false) {}

  void AddNode(sh::graph& graph, const at::Stack& stack) override;
};

class OptimizerFusedSGDMomentumOperator : public OpBackend {
 public:
  OptimizerFusedSGDMomentumOperator(int device_id, c10::ScalarType scalar_type)
      : OpBackend(
            device_id,
            "optimizer_sgd_",
            scalar_type,
            {},
            {1, 2}, // inplace ids
            {},
            false) {}

  void AddNode(sh::graph& graph, const at::Stack& stack) override;
};

void OptimizerFusedSGDOperator::AddNode(
    sh::graph& graph,
    const at::Stack& stack) {
  StackGetter stackGetter(stack, "OptimizerFusedSGDOperator::AddNode");
  auto gradients = getNextInput<std::vector<TensorsPair>>(stackGetter);
  auto weights = getNextInput<std::vector<TensorsPair>>(stackGetter);
  auto lr = getNextInput<TensorsPair>(stackGetter);
  auto wd = getNextInput<double>(stackGetter);
  auto mom = getNextInput<double>(stackGetter);
  auto damp = getNextInput<double>(stackGetter);
  auto nesterov = getNextInput<bool>(stackGetter);

  if (gradients.size() != weights.size()) {
    std::stringstream ss;
    ss << "Both vector inputs must have the same number of elements but they respectively have: "
       << gradients.size() << ", " << weights.size();
    AT_ERROR(ss.str());
  }

  ns_OptimizerSGD::Params sgd_params;
  sgd_params.wd = wd;
  sgd_params.damp = damp;
  sgd_params.mom = mom;
  sgd_params.nesterov = nesterov;
  SetScalarType(gradients[0].pt_t.scalar_type());
  std::string sgd_guid =
      get_guid_with_precision("optimizer_sgd_bwd", ScalarType());
  size_t vec_size = gradients.size();
  for (size_t i = 0; i < vec_size; ++i) {
    const auto& gradient = gradients[i];
    const auto& weight = weights[i];
    std::vector<synTensor> input = {gradient.syn_t, weight.syn_t, lr.syn_t};

    auto sgd = OpBackend::BuildOp(
        graph,
        sgd_guid,
        std::move(input),
        {NodeAttr::NodeOutputAttr{weight.pt_t.sizes(), ScalarType(), i}},
        &sgd_params,
        sizeof(sgd_params));
    syn_out(i) = std::move(sgd.at(0));
  }
}

void OptimizerFusedSGDMomentumOperator::AddNode(
    sh::graph& graph,
    const at::Stack& stack) {
  StackGetter stackGetter(stack, "OptimizerFusedSGDOperator::AddNode");
  auto gradients = getNextInput<std::vector<TensorsPair>>(stackGetter);
  auto weights = getNextInput<std::vector<TensorsPair>>(stackGetter);
  auto momentums = getNextInput<std::vector<TensorsPair>>(stackGetter);
  auto epoch_num = getNextInput<TensorsPair>(stackGetter);
  auto lr = getNextInput<TensorsPair>(stackGetter);
  auto mom = getNextInput<TensorsPair>(stackGetter);
  auto wd = getNextInput<double>(stackGetter);
  auto damp = getNextInput<double>(stackGetter);
  auto nesterov = getNextInput<bool>(stackGetter);

  if ((gradients.size() != weights.size()) ||
      (weights.size() != momentums.size())) {
    std::stringstream ss;
    ss << "All vector inputs must have the same number of elements but they respectively have: "
       << gradients.size() << ", " << weights.size() << ", "
       << momentums.size();
    AT_ERROR(ss.str());
  }

  ns_OptimizerSGD::Params sgd_params;
  sgd_params.wd = wd;
  sgd_params.damp = damp;
  sgd_params.mom = (float)0.1;
  sgd_params.nesterov = nesterov;
  SetScalarType(gradients[0].pt_t.scalar_type());
  std::string sgd_guid =
      get_guid_with_precision("optimizer_sgd_bwd", ScalarType());

  size_t vec_size = gradients.size();
  for (size_t i = 0; i < vec_size; ++i) {
    const auto& gradient = gradients[i];
    const auto& weight = weights[i];
    const auto& momentum = momentums[i];
    std::vector<synTensor> input = {
        gradient.syn_t,
        weight.syn_t,
        momentum.syn_t,
        epoch_num.syn_t,
        lr.syn_t,
        mom.syn_t};

    auto sgd = OpBackend::BuildOp(
        graph,
        sgd_guid,
        std::move(input),
        {NodeAttr::NodeOutputAttr{weight.pt_t.sizes(), ScalarType(), i},
         NodeAttr::NodeOutputAttr{
             momentum.pt_t.sizes(), ScalarType(), vec_size + i}},
        &sgd_params,
        sizeof(sgd_params));
    syn_out(i) = std::move(sgd.at(0));
    syn_out(vec_size + i) = std::move(sgd.at(1));
  }
}

} // namespace habana

static auto& OptimizerKernelsKernelRegistry =
    habana::KernelRegistry()
        .add("hpu::optimizer_sgd", KERNEL_FN(OptimizerFusedSGDOperator))
        .add(
            "hpu::optimizer_sgd_momentum",
            KERNEL_FN(OptimizerFusedSGDMomentumOperator));
