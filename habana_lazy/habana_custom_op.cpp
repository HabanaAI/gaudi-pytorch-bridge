#include <iostream>
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
    habana_lazy::ir::Value& out = hlresult.CurrentIrValue();
    out.SetNode(
        node,
        hlresult.GetDevice(),
        hlresult.GetSizes(),
        hlresult.dtype_optional(),
        outputs_desc[out_idx].index);
    results.emplace_back(result);
  }
  return results;
}

const HabanaCustomOpDescriptor HabanaCustomOpDescriptor::getCustomOpDescriptor(
    std::string op) {
  return habana::KernelRegistry().get_custom_op_desc(op);
}

std::string HabanaCustomOpDescriptor::getSchemaName() const {
  return node_desc_.schema_name;
}

std::string HabanaCustomOpDescriptor::getGuid() const {
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
  habana::KernelRegistry().add_custom_op(
      new_desc.getSchemaName(),
      [&](const int device_id, std::string schema_name) {
        auto& desc = habana::KernelRegistry().get_custom_op_desc(schema_name);
        return std::make_shared<habana::CustomOperator>(device_id, desc);
      },
      new_desc);
}
} // namespace custom_op
} // namespace habana