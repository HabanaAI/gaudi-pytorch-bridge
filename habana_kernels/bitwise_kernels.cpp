/******************************************************************************
 * Copyright (C) 2020 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */
#include <ATen/ExpandUtils.h>
#include <TH/THTensor.hpp>
#include <torch/script.h>
#include <memory>

#include "habana_helpers/graph.h"
#include "habana_helpers/tensor_utils.h"
#include "habana_kernels/bitwise_kernels.h"
#include "habana_kernels/kernel_utils.h"
#include "habana_kernels/resize.h"

using namespace habana;
using namespace torch;

std::vector<int64_t> BitwiseOutOperator::compute_output_shape(
    const Tensor& arg1,
    const Tensor& arg2) {
  auto out_size = habana_helpers::compute_broadcast_shape(arg1, arg2);
  return out_size;
}

void BitwiseOutOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    bool is_output_persistent) {
  static_cast<void>(inputs);
  static_cast<void>(is_output_persistent);
  p_context_->syn_outputs_.emplace_back(std::move(p_context_->syn_inputs_[0]));
  p_context_->syn_inputs_.erase(p_context_->syn_inputs_.cbegin());
  AddNodeToSynapseGraph(graph, nullptr, 0);
}

void BitwiseOutWrapOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    bool is_output_persistent) {
  TORCH_CHECK(
      inputs.size() == 3,
      "Incorrect size of input expected for Bitwise operator");
  TORCH_CHECK(
      inputs[0].isTensor(),
      "Input arg0 type expected to be a tensor for Bitwise operator");
  TORCH_CHECK(
      inputs[1].isTensor(),
      "Input arg1 type expected to be a tensor for Bitwise operator");
  TORCH_CHECK(
      inputs[2].isTensor() || inputs[2].isScalar(),
      "Input arg2 type expected to be a tensor or scalar");

  auto self = inputs[1].toTensor();
  // Support for other integral types dependent on
  // https://jira.habana-labs.com/browse/SW-36542
  TORCH_CHECK(
      self.scalar_type() == c10::ScalarType::Bool,
      "Bitwise operator supports only Boolean inputs for now");

  auto BitwiseOutOp =
      make_operator<BitwiseOutOperator>(this->p_context_->device_id_, guid_);

  if (inputs[1].isTensor() && inputs[2].isTensor()) {
    BitwiseOutOp->SetSynapseInput(p_context_->syn_inputs_[0]);
    BitwiseOutOp->SetSynapseInput(p_context_->syn_inputs_[1]);
    BitwiseOutOp->SetSynapseInput(p_context_->syn_inputs_[2]);
    BitwiseOutOp->AllocateAndAddSynapseNode(
        graph, inputs, is_output_persistent);
  } else if (inputs[1].isTensor() && inputs[2].isScalar()) {
    auto arg1 = inputs[1].toTensor();
    auto constOp = make_operator<ConstantOperator>(
        this->p_context_->device_id_, arg1.scalar_type());
    auto const_shape_tensor = habana_helpers::createPTTensor(
        arg1, {1}, arg1.options(), at::MemoryFormat::Contiguous, false);
    torch::jit::Stack constInputs = {IValue(const_shape_tensor), inputs[2]};
    constOp->AllocateAndAddSynapseNode(graph, constInputs, false);
    BitwiseOutOp->SetSynapseInput(p_context_->syn_inputs_[0]);
    BitwiseOutOp->SetSynapseInput(p_context_->syn_inputs_[1]);
    BitwiseOutOp->SetSynapseInput(constOp->GetSynOutputs()[0]);
    // replace input scalar with input tensor in the stack
    inputs.pop_back();
    inputs.emplace_back(constOp->GetOutputs()[0]);
    BitwiseOutOp->AllocateAndAddSynapseNode(
        graph, inputs, is_output_persistent);
  }

  p_context_->pt_outputs_.emplace_back(inputs[0].toTensor());
  p_context_->syn_outputs_.emplace_back(
      std::move(BitwiseOutOp->GetSynOutputs()[0]));
}

void BitwiseNotOutOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    bool is_output_persistent) {
  TORCH_CHECK(
      inputs.size() == 2,
      "Incorrect size of input expected for Bitwise not operator");
  TORCH_CHECK(
      inputs[0].isTensor(),
      "Input arg0 type expected to be a tensor for Bitwise operator");
  TORCH_CHECK(
      inputs[1].isTensor() || inputs[1].isScalar(),
      "Input arg1 type expected to be a tensor or scalar");

  auto self = inputs[1].toTensor();
  at::ScalarType scalar_type = self.scalar_type();
  // Support for other integral types dependent on
  // https://jira.habana-labs.com/browse/SW-36542
  TORCH_CHECK(
      scalar_type == c10::ScalarType::Bool,
      "Bitwise operator supports only Boolean inputs for now");

  // Create a constant operator to get a tensor of ones of size self
  auto constOp = make_operator<ConstantOperator>(
      this->p_context_->device_id_, scalar_type);
  auto const_shape_tensor = habana_helpers::createPTTensor(
      self, {1}, self.options(), at::MemoryFormat::Contiguous, false);
  torch::jit::Stack constOp_stack = {
      IValue(const_shape_tensor), IValue(Scalar(1))};
  constOp->AllocateAndAddSynapseNode(graph, constOp_stack, false);

  // Create xor operator
  auto xorOp = make_operator<BitwiseXorOutOperator>(
      this->p_context_->device_id_, scalar_type);
  xorOp->SetSynapseInput(p_context_->syn_inputs_[0]);
  xorOp->SetSynapseInput(p_context_->syn_inputs_[1]);
  xorOp->SetSynapseInput(constOp->GetSynOutputs()[0]);

  inputs.emplace_back(constOp->GetOutputs()[0]);
  xorOp->AllocateAndAddSynapseNode(graph, inputs, is_output_persistent);

  p_context_->pt_outputs_.emplace_back(inputs[0].toTensor());
  p_context_->syn_outputs_.emplace_back(std::move(xorOp->GetSynOutputs()[0]));
}

template <class BitwiseOp>
void process_generic_tensor_bitwise_out_op(
    const std::vector<at::Tensor>& pt_inputs,
    torch::jit::Stack& stack,
    const std::string& node_guid) {
  size_t device_id = pt_inputs[1].device().index();
  at::ScalarType scalar_type = pt_inputs[1].scalar_type();
  std::string node_type =
      node_guid + "_" + habana_helpers::name_suffix_from_type(scalar_type);

  auto& device = synapse_helpers::HPURegistrar::get_device(device_id);
  BitwiseOp Op(device_id, scalar_type);

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

Tensor& bitwise_and_out_hpu(
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

  auto out_shape = BitwiseOutOperator::compute_output_shape(self, other);
  auto out_reshaped = out.unsafeGetTensorImpl();
  if (out.sizes().vec() != out_shape) {
    THHTensor_resizeNd(
        out_reshaped, out_shape.size(), out_shape.data(), nullptr);
  }

  std::vector<at::Tensor> pt_inputs{out, self, other};
  torch::jit::Stack stack{IValue(out), IValue(self), IValue(other)};
  process_generic_tensor_bitwise_out_op<BitwiseAndOutOperator>(
      pt_inputs, stack, "and");

  PT_KERNEL_END;
  return out;
}

Tensor& bitwise_and_out_hpu(Tensor& out, const Tensor& self, Scalar other) {
  auto out_reshaped = out.unsafeGetTensorImpl();
  auto self_shape = self.sizes().vec();
  auto out_shape = out.sizes().vec();
  if (self.dim() == 0) {
    self.unsafeGetTensorImpl()->set_sizes_and_strides({1}, {1});
  }

  if (self_shape != out_shape) {
    THHTensor_resizeNd(
        out_reshaped, self_shape.size(), self_shape.data(), nullptr);
  }

  std::vector<at::Tensor> pt_inputs{out, self};
  torch::jit::Stack stack{IValue(out), IValue(self), IValue(other)};
  process_generic_tensor_bitwise_out_op<BitwiseAndOutOperator>(
      pt_inputs, stack, "and");

  PT_KERNEL_END;
  return out;
}

Tensor& bitwise_or_out_hpu(
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

  auto out_shape = BitwiseOutOperator::compute_output_shape(self, other);
  auto out_reshaped = out.unsafeGetTensorImpl();
  if (out.sizes().vec() != out_shape) {
    THHTensor_resizeNd(
        out_reshaped, out_shape.size(), out_shape.data(), nullptr);
  }

  std::vector<at::Tensor> pt_inputs{out, self, other};
  torch::jit::Stack stack{IValue(out), IValue(self), IValue(other)};
  process_generic_tensor_bitwise_out_op<BitwiseOrOutOperator>(
      pt_inputs, stack, "or");

  PT_KERNEL_END;
  return out;
}

Tensor& bitwise_xor_out_hpu(
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

  auto out_shape = BitwiseOutOperator::compute_output_shape(self, other);
  auto out_reshaped = out.unsafeGetTensorImpl();
  if (out.sizes().vec() != out_shape) {
    THHTensor_resizeNd(
        out_reshaped, out_shape.size(), out_shape.data(), nullptr);
  }

  std::vector<at::Tensor> pt_inputs{out, self, other};
  torch::jit::Stack stack{IValue(out), IValue(self), IValue(other)};
  process_generic_tensor_bitwise_out_op<BitwiseXorOutOperator>(
      pt_inputs, stack, "xor");

  PT_KERNEL_END;
  return out;
}

Tensor& bitwise_not_out_hpu(Tensor& out, const Tensor& self) {
  PT_KERNEL_BEGIN;
  if (self.dim() == 0) {
    self.unsafeGetTensorImpl()->set_sizes_and_strides({1}, {1});
  }

  auto out_shape = self.sizes().vec();
  auto out_reshaped = out.unsafeGetTensorImpl();
  if (out.sizes().vec() != out_shape) {
    THHTensor_resizeNd(
        out_reshaped, out_shape.size(), out_shape.data(), nullptr);
  }

  std::vector<at::Tensor> pt_inputs{out, self};
  torch::jit::Stack stack{IValue(out), IValue(self)};

  size_t device_id = pt_inputs[1].device().index();
  at::ScalarType scalar_type = pt_inputs[1].scalar_type();
  std::string node_type =
      "not_" + habana_helpers::name_suffix_from_type(scalar_type);

  auto& device = synapse_helpers::HPURegistrar::get_device(device_id);
  BitwiseNotOutOperator Op(device_id, scalar_type);

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

  PT_KERNEL_END;
  return out;
}

static auto& KernelRegistry =
    habana::KernelRegistry()
        .add(
            "hpu::bitwise_and_Tensor_out",
            [](const int device_id, c10::ScalarType node_type) {
              return std::make_shared<BitwiseAndOutOperator>(
                  device_id, node_type);
            })
        .add(
            "hpu::bitwise_or_Tensor_out",
            [](const int device_id, c10::ScalarType node_type) {
              return std::make_shared<BitwiseOrOutOperator>(
                  device_id, node_type);
            })
        .add(
            "hpu::bitwise_xor_Tensor_out",
            [](const int device_id, c10::ScalarType node_type) {
              return std::make_shared<BitwiseXorOutOperator>(
                  device_id, node_type);
            })
        .add(
            "hpu::bitwise_not_Tensor_out",
            [](const int device_id, c10::ScalarType node_type) {
              return std::make_shared<BitwiseNotOutOperator>(
                  device_id, node_type);
            });
