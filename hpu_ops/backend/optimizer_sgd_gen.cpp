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
#include <perf_lib_layer_params.h>
#include "hpu_ops/hpu_op_helper.h"
#include "hpu_ops/stack_getter.h"

namespace sh = synapse_helpers;

namespace habana {

SharedMetaDataVector OptimizerSgdSharedMeta(
    const at::Stack& stack,
    habana_helpers::HabanaExecutionMode /*unused*/) {
  const auto& gradients = stack.at(0).toTensorVector();
  const auto& weights = stack.at(1).toTensorVector();
  const auto& lr = stack.at(2).toTensor();
  auto precision_type = gradients[0].scalar_type();
  if (at::isFloatingType(precision_type))
    precision_type = at::ScalarType::Float;

  SharedMetaDataVector shared_meta_vec;
  size_t weights_size = weights.size();
  for (size_t i = 0; i < weights_size; ++i) {
    const auto& gradient = gradients[i];
    const auto& weight = weights[i];
    SharedMetaData optimizer_shared_meta{"optimizer_sgd_bwd"};
    optimizer_shared_meta.inputs_data = {
        {gradient.dim(), precision_type},
        getSharedMetaFromTensor(weight),
        createOptionalNotPresentSharedMetaTensor(),
        createOptionalNotPresentSharedMetaTensor(),
        getSharedMetaFromTensor(lr)};
    optimizer_shared_meta.outputs_data.emplace_back(
        getSharedMetaFromTensor(weight));
    shared_meta_vec.push_back(optimizer_shared_meta);
  }

  return shared_meta_vec;
}

SharedMetaDataVector OptimizerSgdMomentumSharedMeta(
    const at::Stack& stack,
    habana_helpers::HabanaExecutionMode /*unused*/) {
  const auto& gradients = stack.at(0).toTensorVector();
  const auto& weights = stack.at(1).toTensorVector();
  const auto& momentums = stack.at(2).toTensorVector();
  const auto& epoch_num = stack.at(3).toTensor();
  const auto& lr = stack.at(4).toTensor();
  const auto& mom = stack.at(5).toTensor();
  auto precision_type = gradients[0].scalar_type();
  if (at::isFloatingType(precision_type))
    precision_type = at::ScalarType::Float;

  SharedMetaDataVector shared_meta_vec;
  size_t weights_size = weights.size();
  for (size_t i = 0; i < weights_size; ++i) {
    const auto& gradient = gradients[i];
    const auto& weight = weights[i];
    const auto& momentum = momentums[i];
    SharedMetaData optimizer_shared_meta{"optimizer_sgd_bwd"};
    optimizer_shared_meta.inputs_data = {
        {gradient.dim(), precision_type},
        getSharedMetaFromTensor(weight),
        getSharedMetaFromTensor(momentum),
        getSharedMetaFromTensor(epoch_num),
        getSharedMetaFromTensor(lr),
        getSharedMetaFromTensor(mom)};
    optimizer_shared_meta.outputs_data = {
        getSharedMetaFromTensor(weight),
        createOptionalNotPresentSharedMetaTensor(),
        getSharedMetaFromTensor(momentum)};
    shared_meta_vec.push_back(optimizer_shared_meta);
  }

  return shared_meta_vec;
}

class OptimizerFusedSGDOperator : public OpBackend {
 public:
  OptimizerFusedSGDOperator(int device_id, c10::ScalarType scalar_type)
      : OpBackend(
            device_id,
            NO_TPC + "optimizer_sgd_",
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
            NO_TPC + "optimizer_sgd_momentum",
            scalar_type,
            {},
            {1, 2}, // inplace ids
            {},
            false) {}

  void AddNode(sh::graph& graph, const at::Stack& stack) override;
};

using namespace std::literals;

void OptimizerFusedSGDOperator::AddNode(
    sh::graph& graph,
    const at::Stack& stack) {
  StackGetter stackGetter(this, stack, "OptimizerFusedSGDOperator::AddNode");
  auto gradients = stackGetter.getNextInput<std::vector<TensorsPair>>();
  auto weights = stackGetter.getNextInput<std::vector<TensorsPair>>();
  auto lr = stackGetter.getNextInput<TensorsPair>();
  auto wd = stackGetter.getNextInput<double>();
  auto mom = stackGetter.getNextInput<double>();
  auto damp = stackGetter.getNextInput<double>();
  auto nesterov = stackGetter.getNextInput<bool>();

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
      get_guid_with_precision("optimizer_sgd_bwd"sv, at::ScalarType::Float);
  size_t vec_size = weights.size();
  for (size_t i = 0; i < vec_size; ++i) {
    const auto& gradient = gradients[i];
    const auto& weight = weights[i];
    std::vector<synTensor> input = {gradient.syn_t, weight.syn_t, lr.syn_t};

    auto sgd = BuildOp(
        graph,
        sgd_guid,
        std::move(input),
        {NodeAttr::NodeOutputAttr{
            weight.pt_t.sizes(), weight.pt_t.scalar_type(), i}},
        &sgd_params,
        sizeof(sgd_params));
    syn_out(i) = std::move(sgd.at(0));
  }
}

void OptimizerFusedSGDMomentumOperator::AddNode(
    sh::graph& graph,
    const at::Stack& stack) {
  StackGetter stackGetter(this, stack, "OptimizerFusedSGDOperator::AddNode");
  auto gradients = stackGetter.getNextInput<std::vector<TensorsPair>>();
  auto weights = stackGetter.getNextInput<std::vector<TensorsPair>>();
  auto momentums = stackGetter.getNextInput<std::vector<TensorsPair>>();
  auto epoch_num = stackGetter.getNextInput<TensorsPair>();
  auto lr = stackGetter.getNextInput<TensorsPair>();
  auto mom = stackGetter.getNextInput<TensorsPair>();
  auto wd = stackGetter.getNextInput<double>();
  auto damp = stackGetter.getNextInput<double>();
  auto nesterov = stackGetter.getNextInput<bool>();

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
      get_guid_with_precision("optimizer_sgd_bwd"sv, at::ScalarType::Float);

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
        {NodeAttr::NodeOutputAttr{
             weight.pt_t.sizes(), weight.pt_t.scalar_type(), i},
         NodeAttr::NodeOutputAttr{
             momentum.pt_t.sizes(), momentum.pt_t.scalar_type(), vec_size + i}},
        &sgd_params,
        sizeof(sgd_params));
    syn_out(i) = std::move(sgd.at(0));
    syn_out(vec_size + i) = std::move(sgd.at(1));
  }
}

} // namespace habana

static auto& OptimizerKernelsKernelRegistry =
    habana::KernelRegistry()
        .REGISTER_HPU_BACKEND(
            "hpu::optimizer_sgd",
            habana::OptimizerFusedSGDOperator)
        .REGISTER_HPU_BACKEND(
            "hpu::optimizer_sgd_momentum",
            habana::OptimizerFusedSGDMomentumOperator);
