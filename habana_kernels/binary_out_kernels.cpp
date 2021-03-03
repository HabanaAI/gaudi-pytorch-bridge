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
#include "habana_helpers/graph.h"
#include "habana_helpers/tensor_utils.h"
#include "habana_kernels/binary_kernels.h"
#include "habana_kernels/binary_out_kernels.h"
#include "habana_kernels/kernel_utils.h"
#include "habana_kernels/simple_generic_kernel.h"
#include "habana_kernels/tensor_shape_kernels.h"

using namespace torch;

void habana::BinaryOutOperator::insert_reshape_op(
    synapse_helpers::graph& graph,
    ReshapeOperator& reshapeOp,
    Tensor& arg,
    int32_t position,
    int64_t out_dims) {
  auto arg_sizes = arg.sizes().vec();
  // Create view_sizes initialized to part which has size=1 for upper dims
  auto view_sizes = std::vector<int64_t>(out_dims - arg.ndimension(), 1);
  // and append the smaller tensor dims
  view_sizes.insert(view_sizes.end(), arg_sizes.begin(), arg_sizes.end());

  auto& reshape_syn_input =
      reshapeOp.SetSynapseInput(std::move(p_context_->syn_inputs_[position]));

  torch::jit::Stack reshapeOp_stack = {IValue(arg), IValue(view_sizes)};
  reshapeOp.AllocateAndAddSynapseNode(graph, reshapeOp_stack, false);
  p_context_->syn_inputs_[position] = std::move(reshape_syn_input);
}

/************************************************************************
 * @brief This function implements synapse node addition for
 * binary operators where both inputs are tensors. Mismatch in input
 * tensor dims is also taken care of using reshape nodes.
 ************************************************************************/
void habana::BinaryOutOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    bool is_output_persistent) {
  // this check is for stack during graph execution
  TORCH_CHECK(
      inputs.size() == 3,
      "Incorrect size of input expected for Binary out operator");
  TORCH_CHECK(inputs[0].isTensor(), "Input type expected to be tensor");
  TORCH_CHECK(inputs[1].isTensor(), "Input type expected to be tensor");
  TORCH_CHECK(inputs[2].isTensor(), "Input type expected to be tensor");

  Tensor output = inputs[0].toTensor();
  Tensor arg1 = inputs[1].toTensor();
  Tensor arg2 = inputs[2].toTensor();

  bool isArg1modified = false, isArg2modified = false;
  std::vector<synapse_helpers::tensor_or_ref> reshape_syn_output;
  auto out_dims = arg1.ndimension() > arg2.ndimension() ? arg1.ndimension()
                                                        : arg2.ndimension();

  ReshapeOperator reshapeOp(this->p_context_->device_id_, this->scalarType_);
  // Make sure that we give tensors that match dims to Synapse
  if (arg1.ndimension() > arg2.ndimension()) {
    isArg2modified = true;
    insert_reshape_op(graph, reshapeOp, arg2, 2, out_dims);
    reshape_syn_output.push_back(std::move(reshapeOp.GetSynOutputs()[0]));
  } else if (arg1.ndimension() < arg2.ndimension()) {
    isArg1modified = true;
    insert_reshape_op(graph, reshapeOp, arg1, 1, out_dims);
    reshape_syn_output.push_back(std::move(reshapeOp.GetSynOutputs()[0]));
  }

  synapse_helpers::tensor& arg1_syn_tensor =
      isArg1modified ? reshape_syn_output[0] : p_context_->syn_inputs_[1];
  synapse_helpers::tensor& arg2_syn_tensor =
      isArg2modified ? reshape_syn_output[0] : p_context_->syn_inputs_[2];

  p_context_->syn_outputs_.emplace_back(std::move(p_context_->syn_inputs_[0]));

  std::vector<synTensor> syn_inputs;
  syn_inputs.push_back(arg1_syn_tensor.get());
  syn_inputs.push_back(arg2_syn_tensor.get());

  p_context_->pt_outputs_.emplace_back(output);

  synapse_helpers::tensor& output_syn_tensor = p_context_->syn_outputs_[0];
  std::vector<synTensor> syn_outputs{output_syn_tensor.get()};

  graph.add_node(
      std::move(syn_inputs),
      std::move(syn_outputs),
      nullptr,
      0,
      std::move(guid_));
}

/************************************************************************
 * @brief Generic wrapper for all Eager mode Binary OP invocations that
 * are  .out
 ************************************************************************/
template <class BinaryOp>
void process_generic_tensor_binary_out_op(
    const std::vector<at::Tensor>& pt_inputs,
    torch::jit::Stack& stack,
    const std::string& node_guid) {
  size_t device_id = pt_inputs[0].device().index();
  at::ScalarType scalar_type = pt_inputs[1].scalar_type();
  std::string node_type =
      node_guid + "_fwd_" + habana_helpers::name_suffix_from_type(scalar_type);

  auto& device = synapse_helpers::HPURegistrar::get_device(device_id);
  BinaryOp Op(device_id, scalar_type);

  size_t key =
      Op.GetRecipeKey(node_type, stack, /*inplace*/ false, /*out*/ true);

  if (device.get_recipe_handle_cache().isCached(key)) {
    PT_KERNEL_DEBUG("Cache hit key:", key);
    Op.SetPTInputs(pt_inputs);
    Op.SetPTOutput(pt_inputs[0]);
    Op.Execute(key);
  } else {
    PT_KERNEL_DEBUG("key:", key);
    // Create Graph
    auto graph = habana_helpers::create_graph(device_id, node_type);

    // Assign Inputs to the Operator
    Op.AllocateSynapseInputs(graph, pt_inputs, true);

    // both inputs are not required, just to match graph mode stack
    Op.AllocateAndAddSynapseNode(graph, stack, true);

    // compile and execute the graph
    Op.Compile(graph);
  }
}

/*************************************************************************
 * @brief Kernel implementation for out = torch.mul(out, self, other)
 * @param self - first input
 * @param other - second input
 * out = self * other
 ************************************************************************/

Tensor& mul_out_hpu(Tensor& out, const Tensor& self, const Tensor& other) {
  PT_KERNEL_BEGIN;

  if (self.dim() == 0) {
    self.unsafeGetTensorImpl()->set_sizes_and_strides({1}, {1});
  }
  if (other.dim() == 0) {
    other.unsafeGetTensorImpl()->set_sizes_and_strides({1}, {1});
  }

  auto self_hpu = get_hpu_tensor(self);
  auto other_hpu = get_hpu_tensor(other);

  std::vector<at::Tensor> pt_inputs{out, self_hpu, other_hpu};
  torch::jit::Stack stack{IValue(out), IValue(self_hpu), IValue(other_hpu)};
  process_generic_tensor_binary_out_op<habana::MulOutOperator>(
      pt_inputs, stack, "mult");

  PT_KERNEL_END;
  return out;
}

/****************************************************************************
 * @brief Kernel implementation for result = torch.div(input, denom, out=out)
 * @param result - output
 * @param self - first input
 * @param other - second input
 ***************************************************************************/
Tensor& div_tensor_hpu_out(
    Tensor& out,
    const Tensor& self,
    const Tensor& other) {
  PT_KERNEL_BEGIN;

  if (self.dim() == 0) {
    self.unsafeGetTensorImpl()->set_sizes_and_strides({1}, {1});
  }
  if (other.dim() == 0) {
    other.unsafeGetTensorImpl()->set_sizes_and_strides({1}, {1});
  }

  auto self_hpu = get_hpu_tensor(self);
  auto other_hpu = get_hpu_tensor(other);

  std::vector<at::Tensor> pt_inputs{out, self_hpu, other_hpu};
  torch::jit::Stack stack{IValue(out), IValue(self_hpu), IValue(other_hpu)};
  process_generic_tensor_binary_out_op<habana::DivOutOperator>(
      pt_inputs, stack, "div");

  PT_KERNEL_END;
  return out;
}

static auto& KernelRegistry = habana::KernelRegistry().add(
    "aten::mul.out",
    [](const int device_id, c10::ScalarType node_type) {
      return std::make_shared<habana::MulOutOperator>(device_id, node_type);
    });