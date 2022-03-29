/******************************************************************************
 * Copyright (C) 2021 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */
#include <ATen/ExpandUtils.h>
#include <torch/script.h>
#include <memory>

#include "habana_device/HPUCheck.h"
#include "habana_device/hpu_cached_devices.h"
#include "habana_helpers/tensor_utils.h"
#include "habana_kernels/binary_kernels.h"
#include "habana_kernels/kernel_utils.h"
#include "habana_kernels/simple_generic_kernel.h"
#include "habana_kernels/tensor_shape_kernels.h"
#include "habana_kernels/where_kernels.h"

using namespace torch;
using namespace habana;

/************************************************************************
 * @brief This function implements synapse node addition for where function
 * with 3 input arguments (where all arguments are tensors)
 ************************************************************************/
void WhereOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    const OutputMetaDataVector& output_metadata) {
  TORCH_CHECK(
      inputs.size() == 3,
      "Incorrect size of input expected for where operator");
  TORCH_CHECK(
      inputs[0].isTensor(), "Input condition type expected to be a tensor");
  TORCH_CHECK(inputs[1].isTensor(), "Input arg1 type expected to be a tensor");
  TORCH_CHECK(inputs[2].isTensor(), "Input arg2 type expected to be a tensor");

  auto condition = inputs[0].toTensor();
  auto self = inputs[1].toTensor();
  auto other = inputs[2].toTensor();

  // Node type is decided by 2nd input's type
  guid_ =
      "where_fwd_" + habana_helpers::name_suffix_from_type(self.scalar_type());
  auto output_shape = compute_output_shape(condition, self, other);

  auto output =
      at::empty(output_shape, self.options(), self.suggest_memory_format());

  AllocateSynapseOutput(graph, output, output_metadata.at(0));
  AddNodeToSynapseGraph(graph, nullptr, 0);
}

void WhereOperator::SetPTOutputs(torch::jit::Stack& inputs) {
  Tensor output;

  auto self = inputs[1].toTensor();
  auto output_shape =
      compute_output_shape(inputs[0].toTensor(), self, inputs[2].toTensor());
  output =
      at::empty(output_shape, self.options(), self.suggest_memory_format());

  std::vector<at::Tensor> v{output};
  HabanaOperator::SetPTOutputs(v);
}

std::vector<int64_t> WhereOperator::compute_output_shape(
    const Tensor& condition,
    const Tensor& self,
    const Tensor& other) {
  auto out_dims = std::max(
      condition.ndimension(), std::max(self.ndimension(), other.ndimension()));

  std::vector<int64_t> result_tensor_sizes(out_dims, 1);

  auto cond_tensor_sizes =
      std::vector<int64_t>(out_dims - condition.ndimension(), 1);
  auto self_tensor_sizes =
      std::vector<int64_t>(out_dims - self.ndimension(), 1);
  auto other_tensor_sizes =
      std::vector<int64_t>(out_dims - other.ndimension(), 1);

  auto cond_tensor_sizes_actual = condition.sizes().vec();
  auto self_tensor_sizes_actual = self.sizes().vec();
  auto other_tensor_sizes_actual = other.sizes().vec();

  cond_tensor_sizes.insert(
      cond_tensor_sizes.end(),
      cond_tensor_sizes_actual.begin(),
      cond_tensor_sizes_actual.end());
  self_tensor_sizes.insert(
      self_tensor_sizes.end(),
      self_tensor_sizes_actual.begin(),
      self_tensor_sizes_actual.end());
  other_tensor_sizes.insert(
      other_tensor_sizes.end(),
      other_tensor_sizes_actual.begin(),
      other_tensor_sizes_actual.end());

  for (unsigned i = 0; i < result_tensor_sizes.size(); i++) {
    result_tensor_sizes[i] = std::max(
        cond_tensor_sizes[i],
        std::max(self_tensor_sizes[i], other_tensor_sizes[i]));
  }

  return result_tensor_sizes;
}

Tensor process_where_op(
    const std::vector<at::Tensor>& pt_inputs,
    torch::jit::Stack& stack,
    const std::string& node_guid) {
  size_t device_id = pt_inputs[0].device().index();
  at::ScalarType scalar_type = pt_inputs[1].scalar_type();
  std::string node_type =
      node_guid + "_fwd_" + habana_helpers::name_suffix_from_type(scalar_type);

  auto& device = synapse_helpers::HPURegistrar::get_device(device_id);
  WhereOperator Op(device_id, scalar_type);

  size_t key = Op.GetRecipeKey(node_type, stack);

  if (device.get_recipe_handle_cache().isCached(key)) {
    Op.Execute(key, pt_inputs, stack);
  } else {
    // both inputs are not required, just to match graph mode stack
    OutputMetaDataVector output_metadata(1);
    output_metadata.at(0).persistent = true;
    // compile and execute the graph
    Op.CreateGraphAndCompile(key, pt_inputs, stack, output_metadata, true);
  }
  std::vector<at::Tensor> out = Op.GetOutputs();
  TORCH_CHECK(out.size() == 1, "Incorrect size of outputs");
  return out[0];
}

Tensor where_tensor_hpu(
    const Tensor& condition,
    const Tensor& self,
    const Tensor& other) {
  PT_KERNEL_BEGIN;

  auto condition_hpu = get_hpu_tensor(condition);
  auto self_hpu = get_hpu_tensor(self);
  auto other_hpu = get_hpu_tensor(other);

  std::vector<at::Tensor> pt_inputs{condition_hpu, self_hpu, other_hpu};
  torch::jit::Stack stack{
      IValue(condition_hpu), IValue(self_hpu), IValue(other_hpu)};

  auto output = process_where_op(pt_inputs, stack, "where");

  PT_KERNEL_END;
  return output;
}

static auto& KernelRegistry =
    habana::KernelRegistry().add("aten::_s_where", KERNEL_FN(WhereOperator));
