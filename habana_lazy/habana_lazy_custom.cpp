#include "habana_lazy/habana_lazy_custom.h"
#include <iostream>
#include "habana_kernels/binary_kernels.h"
#include "habana_kernels/custom_op_kernel.h"
#include "habana_kernels/lazy_kernels.h"

namespace habana {
namespace custom_op {

at::Tensor HabanaCustomOpDescriptor::execute(
    const std::vector<c10::IValue>& inputs) {
  // Currently supported for only single output [SW-60948]
  if (outputs_.size() == 1) {
    habana_lazy::LazyOp<at::Tensor> k{getSchemaName(), inputs};
    return k.call();
  } else {
    HABANA_ASSERT(0 && "Custom op with multiple outputs not implemented");
  }
  return at::empty({});
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