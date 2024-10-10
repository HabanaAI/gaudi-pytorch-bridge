/*******************************************************************************
 * Copyright (C) 2021-2024 Habana Labs, Ltd. an Intel Company
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
#include "habana_kernels/binary_kernels.h"
#include "habana_kernels/custom_op_kernel.h"
#include "habana_kernels/lazy_kernels.h"
#include "habana_lazy/ops/custom_op.h"
#include "include/habanalabs/hpu_custom_op.h"

namespace habana {
namespace custom_op {

std::vector<at::Tensor> HabanaCustomOpDescriptor::execute(
    const std::vector<c10::IValue>& inputs) {
  habana_lazy::ir::NodePtr node =
      std::make_shared<habana_lazy::ir::CustomOp>(getSchemaName(), inputs);

  verifyInputOutputIndexes();

  std::vector<at::Tensor> results;
  auto outputs_desc = getOutputs();
  for (unsigned out_idx = 0; out_idx < getOutputsSize(); ++out_idx) {
    auto in_tensor = inputs.at(0).toTensor();
    std::vector<int64_t> result_sizes = in_tensor.sizes().vec();
    if (hasOutputShapeFunc(out_idx)) {
      compute_output_shape_function output_shape_func =
          getOutputShapeFunc(out_idx);
      result_sizes = output_shape_func(inputs);
    }

    auto options = in_tensor.options().dtype(outputs_desc[out_idx].dtype);
    auto result = habana_lazy::empty_hpu_lazy(
        result_sizes, options, in_tensor.suggest_memory_format(), false);
    const auto hlresult = habana_lazy::GetHbLazyTensor(result);
    hlresult.IrSetNode(node, outputs_desc[out_idx].index);
    results.emplace_back(result);
  }
  return results;
}

const HabanaCustomOpDescriptor HabanaCustomOpDescriptor::getCustomOpDescriptor(
    std::string op) {
  return habana::KernelRegistry().get_legacy_user_custom_op_desc(op);
}

const std::string& HabanaCustomOpDescriptor::getSchemaName() const {
  return node_desc_.schema_name;
}

const std::string& HabanaCustomOpDescriptor::getGuid() const {
  return node_desc_.tpc_guid;
}

unsigned HabanaCustomOpDescriptor::getInputsSize() const {
  return inputs_.size();
}

unsigned HabanaCustomOpDescriptor::getOutputsSize() const {
  return outputs_.size();
}

const std::vector<InputDesc>& HabanaCustomOpDescriptor::getInputs() const {
  return inputs_;
}

const std::vector<OutputDesc>& HabanaCustomOpDescriptor::getOutputs() const {
  return outputs_;
}

bool HabanaCustomOpDescriptor::hasUserParamsFunc() const {
  return node_desc_.user_param_func != nullptr;
}
const allocate_user_params_func& HabanaCustomOpDescriptor::
    getUserParamsAllocFunc() const {
  return node_desc_.user_param_func;
}

bool HabanaCustomOpDescriptor::hasOutputShapeFunc(unsigned index) const {
  TORCH_CHECK(
      index < getOutputsSize(),
      getSchemaName(),
      " has ",
      getOutputsSize(),
      ", requested index: ",
      index);
  return getOutputs().at(index).compute_output_shape_func != nullptr;
}

const compute_output_shape_function& HabanaCustomOpDescriptor::
    getOutputShapeFunc(unsigned index) const {
  TORCH_CHECK(
      index < getOutputsSize(),
      getSchemaName(),
      " has ",
      getOutputsSize(),
      ", requested index: ",
      index);
  return getOutputs().at(index).compute_output_shape_func;
}

void registerKernel(HabanaCustomOpDescriptor& new_desc) {
  habana::KernelRegistry().add_legacy_user_custom_op(
      new_desc.getSchemaName(),
      [&](const int device_id, std::string schema_name) {
        auto& desc = habana::KernelRegistry().get_legacy_user_custom_op_desc(
            schema_name);
        return std::make_shared<habana::CustomOperator>(device_id, desc);
      },
      new_desc);
}

void HabanaCustomOpDescriptor::verifyInputOutputIndexes() {
  auto check_unique = [](auto&& descriptors) {
    std::unordered_set<unsigned> unique_indexes;
    for (auto&& descriptor : descriptors) {
      TORCH_CHECK(
          unique_indexes.insert(descriptor.index).second,
          "Indexes must be unique");
    }
  };
  check_unique(inputs_);
  check_unique(outputs_);
}
} // namespace custom_op
} // namespace habana
