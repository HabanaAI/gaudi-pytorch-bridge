#include "habana_lazy/habana_lazy_custom.h"
#include <iostream>
#include "habana_kernels/binary_kernels.h"
#include "habana_kernels/custom_op_kernel.h"
#include "habana_kernels/lazy_kernels.h"
#include "habana_lazy/ops/custom_op.h"

namespace habana {
namespace custom_op {

std::vector<at::Tensor> HabanaCustomOpDescriptor::execute(
    const std::vector<c10::IValue>& inputs) {
  habana_lazy::ir::NodePtr node =
      std::make_shared<habana_lazy::ir::CustomOp>(getSchemaName(), inputs);

  std::vector<at::Tensor> results;
  for (const auto& o : outputs_) {
    // TODO: handle output size
    auto t = inputs.at(0).toTensor();
    auto result = habana_lazy::empty_hpu_lazy(
        t.sizes(), t.options(), t.suggest_memory_format(), false);
    const auto hlresult = habana_lazy::GetHbLazyTensor(result);
    habana_lazy::ir::Value& out = hlresult.CurrentIrValue();
    out.SetNode(
        node,
        hlresult.GetDevice(),
        hlresult.GetSizes(),
        hlresult.dtype_optional(),
        o.index);
    results.emplace_back(result);
  }
  return results;
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

const std::vector<OutputDesc>& HabanaCustomOpDescriptor::getOutputs() const {
  return outputs_;
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