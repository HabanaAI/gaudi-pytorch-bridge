/******************************************************************************
 * Copyright (C) 2020 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */
#include <perf_lib_layer_params.h>
#include <torch/script.h>

#include <ATen/ExpandUtils.h>
#include <ATen/InferSize.h>
#include <ATen/WrapDimUtils.h>

#include "habana_device/HPUCheck.h"
#include "habana_device/hpu_cached_devices.h"
#include "habana_helpers/graph.h"
#include "habana_helpers/tensor_utils.h"
#include "habana_kernels/basic_kernels.h"
#include "habana_kernels/binary_composite_kernels.h"
#include "habana_kernels/binary_kernels.h"
#include "habana_kernels/binary_out_kernels.h"
#include "habana_kernels/kernel_utils.h"
#include "habana_kernels/resize.h"
#include "habana_kernels/simple_generic_kernel.h"
#include "habana_kernels/unary_kernels.h"
#include "synapse_helpers/recipe.h"

using namespace torch;
using namespace torch::jit;
using namespace habana;

void UnaryOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    Stack& inputs,
    bool is_output_persistent) {
  TORCH_CHECK(
      inputs.size() == 1, "Incorrect size of inputs expected for operator");
  TORCH_CHECK(inputs[0].isTensor(), "Input type expected to be tensor");

  at::Tensor input = inputs[0].toTensor();
  auto output = habana_helpers::createPTTensor(input, is_output_persistent);
  AllocateSynapseOutput(graph, output, is_output_persistent);
  AddNodeToSynapseGraph(graph, nullptr, 0);
}

void UnaryInplaceOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    Stack& inputs,
    bool is_output_persistent) {
  static_cast<void>(is_output_persistent);

  p_context_->syn_outputs_.emplace_back(
      habana_helpers::duplicate_tensor_in_memory_section(
          p_context_->syn_inputs_[0], graph));
  p_context_->pt_outputs_.emplace_back(inputs[0].toTensor());
  AddNodeToSynapseGraph(graph, nullptr, 0);
}

void UnaryLikeOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    Stack& inputs,
    bool is_output_persistent) {
  if (m_inplace) {
    p_context_->syn_outputs_.emplace_back(
        habana_helpers::duplicate_tensor_in_memory_section(
            p_context_->syn_inputs_[0], graph));
    p_context_->pt_outputs_.emplace_back(inputs[0].toTensor());
  } else {
    auto output = habana_helpers::createPTTensor(
        inputs[0].toTensor(), is_output_persistent);
    AllocateSynapseOutput(graph, output, is_output_persistent);
  }
}

Tensor unary_op_hpu(
    const Tensor& input,
    std::string& node_type,
    UnaryOperator* Op) {
  size_t device_id = input.device().index();
  std::vector<c10::IValue> stack = {IValue(input)};
  std::vector<at::Tensor> pt_inputs{input};
  auto& device = synapse_helpers::HPURegistrar::get_device(device_id);
  size_t key = Op->GetRecipeKey(node_type, stack);

  if (device.get_recipe_handle_cache().isCached(key)) {
    PT_KERNEL_DEBUG("Cache hit key:", key);
    auto output = at::empty(
        input.sizes(), input.options(), input.suggest_memory_format());
    Op->SetPTInputs(pt_inputs);
    Op->SetPTOutput(output);
    Op->Execute(key);
  } else {
    PT_KERNEL_DEBUG("Key:", key);
    //
    // Create Graph
    auto graph = habana_helpers::create_graph(device_id, node_type);

    // Assign Inputs to the Operator
    Op->AllocateSynapseInputs(graph, pt_inputs, true);

    // Build Params for the graph
    Op->AllocateAndAddSynapseNode(graph, stack, true);

    // compile and execute the graph
    Op->Compile(graph);
  }
  std::vector<at::Tensor> out = Op->GetOutputs();
  TORCH_CHECK(out.size() == 1, "Incorrect size of outputs");

  return out.at(0);
}

void unary_inplace_op_hpu(
    const Tensor& self,
    std::string& node_type,
    UnaryInplaceOperator* Op) {
  size_t device_id = self.device().index();
  auto& device = synapse_helpers::HPURegistrar::get_device(device_id);
  std::vector<at::Tensor> pt_inputs{self};
  // Build Params for the graph
  std::vector<c10::IValue> stack = {IValue(self)};

  size_t key = Op->GetRecipeKey(node_type, stack, true);

  if (device.get_recipe_handle_cache().isCached(key)) {
    PT_KERNEL_DEBUG("Cache hit key:", key);
    Op->SetPTInputs(pt_inputs);
    Op->SetPTOutput(pt_inputs[0]);
    Op->Execute(key);
  } else {
    PT_KERNEL_DEBUG("Key:", key);
    // Create Graph
    auto graph = habana_helpers::create_graph(device_id, node_type);

    // Assign Inputs to the Operator
    Op->AllocateSynapseInputs(graph, pt_inputs, true);

    Op->AllocateAndAddSynapseNode(graph, stack, true);

    // compile and execute the graph
    Op->Compile(graph);
  }
}

void UnaryBackwardOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    Stack& inputs,
    bool is_output_persistent) {
  TORCH_CHECK(
      inputs.size() == 2,
      "Incorrect size of inpust expected for UnaryBackward operator");
  TORCH_CHECK(inputs[0].isTensor(), "Input type expected to be tensor");
  TORCH_CHECK(inputs[1].isTensor(), "Input type expected to be tensor");

  at::Tensor grad_in = inputs[0].toTensor();
  at::Tensor input = inputs[1].toTensor();

  TORCH_CHECK(
      grad_in.scalar_type() == input.scalar_type(),
      "Types don't match. grad_in type: ",
      grad_in.scalar_type(),
      " input type: ",
      input.scalar_type());
  TORCH_CHECK(
      (grad_in.sizes() == input.sizes()) ||
          (grad_in.ndimension() == input.ndimension() &&
           std::all_of(
               input.sizes().cbegin(),
               input.sizes().cend(),
               [](auto val) { return val == 1; })),
      "Sizes in elementwise kernel don't match. grad_in sizes: ",
      grad_in.sizes(),
      ", input sizes: ",
      input.sizes());

  auto grad_output =
      habana_helpers::createPTTensor(input, is_output_persistent);
  AllocateSynapseOutput(graph, grad_output, is_output_persistent);
  AddNodeToSynapseGraph(graph, nullptr, 0);
}

Tensor unary_backward_op_hpu(
    const Tensor& grad_in,
    const Tensor& input,
    std::string& node_type,
    UnaryBackwardOperator* Op) {
  TORCH_CHECK(
      grad_in.scalar_type() == input.scalar_type(),
      "Types don't match. grad_in type: ",
      grad_in.scalar_type(),
      " input type: ",
      input.scalar_type());
  TORCH_CHECK(
      (grad_in.sizes() == input.sizes()) ||
          (grad_in.ndimension() == input.ndimension() &&
           std::all_of(
               input.sizes().cbegin(),
               input.sizes().cend(),
               [](auto val) { return val == 1; })),
      "Sizes in elementwise kernel don't match. grad_in sizes: ",
      grad_in.sizes(),
      ", input sizes: ",
      input.sizes());
  size_t device_id = input.device().index();
  std::vector<c10::IValue> stack = {IValue(grad_in), IValue(input)};
  std::vector<at::Tensor> pt_inputs{grad_in, input};
  auto& device = synapse_helpers::HPURegistrar::get_device(device_id);
  size_t key = Op->GetRecipeKey(node_type, stack);

  if (device.get_recipe_handle_cache().isCached(key)) {
    PT_KERNEL_DEBUG("Cache hit key:", key);
    auto output = at::empty(
        input.sizes(), input.options(), input.suggest_memory_format());
    Op->SetPTInputs(pt_inputs);
    Op->SetPTOutput(output);
    Op->Execute(key);
  } else {
    PT_KERNEL_DEBUG("Key:", key);
    //
    // Create Graph
    auto graph = habana_helpers::create_graph(device_id, node_type);

    // Assign Inputs to the Operator
    Op->AllocateSynapseInputs(graph, pt_inputs, true);

    // Build Params for the graph
    Op->AllocateAndAddSynapseNode(graph, stack, true);

    // compile and execute the graph
    Op->Compile(graph);
  }
  std::vector<at::Tensor> out = Op->GetOutputs();
  TORCH_CHECK(out.size() == 1, "Incorrect size of outputs");

  return out.at(0);
}

/*************************************************************************
 * @brief Kernel implementation for output = torch.relu(input)
 * @param [out] output - output tensor, 1-4D, BF16/FP32
 * @param [in] input - input tensor, 1-4D, BF16/FP32
 ************************************************************************/
Tensor relu_hpu(const Tensor& input) {
  PT_KERNEL_BEGIN;
  at::ScalarType scalar_type = input.scalar_type();
  std::string node_type =
      "relu_fwd_" + habana_helpers::name_suffix_from_type(scalar_type);

  // Create the operator
  size_t device_id = input.device().index();
  ReluOperator Op(device_id, scalar_type);

  auto out = unary_op_hpu(input, node_type, &Op);
  PT_KERNEL_END;
  return out;
}

/*************************************************************************
 * @brief Kernel implementation for output = torch.relu_(input)
 * @param [out] output - output tensor, 1-4D, BF16/FP32
 * @param [in] input - input tensor, 1-4D, BF16/FP32
 ************************************************************************/
Tensor& relu_hpu_(Tensor& self) {
  PT_KERNEL_BEGIN;

  at::ScalarType scalar_type = self.scalar_type();
  std::string node_type =
      "relu_fwd_" + habana_helpers::name_suffix_from_type(scalar_type);
  size_t device_id = self.device().index();

  // Create the operator
  ReluInplaceOperator Op(device_id, scalar_type);
  unary_inplace_op_hpu(self, node_type, &Op);
  PT_KERNEL_END;
  return self;
}

/*************************************************************************
 * @brief Kernel implementation for self = torch.leakyrelu_(self)
 * @param [out] self - output tensor, 1-4D, BF16/FP32
 * @param [in] self - input tensor, 1-4D, BF16/FP32
 * @param [in] negative_slope - scalar of type double
 ************************************************************************/
Tensor& leaky_relu_hpu_(Tensor& self, const at::Scalar& negative_slope) {
  PT_KERNEL_BEGIN;
  bool isSelf_0d = false;
  if (self.dim() == 0) {
    self.unsafeGetTensorImpl()->set_sizes_and_strides({1}, {1});
    isSelf_0d = true;
  }
  at::ScalarType scalar_type = self.scalar_type();
  size_t device_id = self.device().index();
  // Create the operator
  // inplace = true
  LeakyReluOperator Op(device_id, scalar_type, true);
  std::string node_type =
      "leakyrelu_fwd_" + habana_helpers::name_suffix_from_type(scalar_type);
  // Assign Inputs to the Operator
  std::vector<at::Tensor> pt_inputs{self};
  std::vector<c10::IValue> stack = {IValue(self), IValue(negative_slope)};

  auto& device = synapse_helpers::HPURegistrar::get_device(device_id);
  size_t key = Op.GetRecipeKey(node_type, stack);

  if (device.get_recipe_handle_cache().isCached(key)) {
    PT_KERNEL_DEBUG("Cache hit key:", key);
    Op.SetPTInputs(pt_inputs);
    Op.SetPTOutput(pt_inputs[0]);
    Op.Execute(key);
  } else {
    PT_KERNEL_DEBUG("key:", key);
    // Create Graph
    auto graph = habana_helpers::create_graph(device_id, node_type);
    Op.AllocateSynapseInputs(graph, pt_inputs, true);
    Op.AllocateAndAddSynapseNode(graph, stack, true);
    Op.Compile(graph);
  }
  if (isSelf_0d) {
    self.unsafeGetTensorImpl()->set_sizes_and_strides({}, {});
  }
  PT_KERNEL_END;
  return self;
}

/*************************************************************************
 * @brief Kernel implementation for self = torch.leakyrelu(self)
 * @param [out] output - output tensor, 1-4D, BF16/FP32
 * @param [in] self - input tensor, 1-4D, BF16/FP32
 * @param [in] negative_slope - scalar of type double
 ************************************************************************/
Tensor leaky_relu_hpu(const Tensor& self, at::Scalar negative_slope) {
  PT_KERNEL_BEGIN;
  bool isSelf_0d = false;
  if (self.dim() == 0) {
    self.unsafeGetTensorImpl()->set_sizes_and_strides({1}, {1});
    isSelf_0d = true;
  }
  at::ScalarType scalar_type = self.scalar_type();
  size_t device_id = self.device().index();
  // Create the operator
  LeakyReluOperator Op(device_id, scalar_type);
  std::string node_type =
      "leakyrelu_fwd_" + habana_helpers::name_suffix_from_type(scalar_type);
  // Assign Inputs to the Operator
  std::vector<at::Tensor> pt_inputs{self};
  std::vector<c10::IValue> stack = {IValue(self), IValue(negative_slope)};

  auto& device = synapse_helpers::HPURegistrar::get_device(device_id);
  size_t key = Op.GetRecipeKey(node_type, stack);

  if (device.get_recipe_handle_cache().isCached(key)) {
    PT_KERNEL_DEBUG("Cache hit key:", key);
    auto output =
        at::empty(self.sizes(), self.options(), self.suggest_memory_format());
    Op.SetPTInputs(pt_inputs);
    Op.SetPTOutput(output);
    Op.Execute(key);
  } else {
    PT_KERNEL_DEBUG("key:", key);
    // Create Graph
    auto graph = habana_helpers::create_graph(device_id, node_type);
    Op.AllocateSynapseInputs(graph, pt_inputs, true);
    Op.AllocateAndAddSynapseNode(graph, stack, true);
    Op.Compile(graph);
  }
  std::vector<at::Tensor> out = Op.GetOutputs();
  TORCH_CHECK(out.size() == 1, "Incorrect size of outputs");
  auto output = out.at(0);
  if (isSelf_0d) {
    self.unsafeGetTensorImpl()->set_sizes_and_strides({}, {});
    output.unsafeGetTensorImpl()->set_sizes_and_strides({}, {});
  }
  PT_KERNEL_END;
  return output;
}

at::Tensor leaky_relu_backward_hpu(
    const at::Tensor& grad_output,
    const at::Tensor& self,
    const at::Scalar& negative_slope,
    bool self_is_result) {
  PT_KERNEL_BEGIN;
  bool isSelf_0d = false;
  bool isGradOutput_0d = false;
  if (self.dim() == 0) {
    self.unsafeGetTensorImpl()->set_sizes_and_strides({1}, {1});
    isSelf_0d = true;
  }
  if (grad_output.dim() == 0) {
    grad_output.unsafeGetTensorImpl()->set_sizes_and_strides({1}, {1});
    isGradOutput_0d = true;
  }
  at::ScalarType scalar_type = self.scalar_type();
  size_t device_id = self.device().index();
  // Create the operator
  LeakyReluBackwardOperator Op(device_id, scalar_type);
  std::string node_type =
      "leakyrelu_bwd_" + habana_helpers::name_suffix_from_type(scalar_type);
  // Assign Inputs to the Operator
  std::vector<at::Tensor> pt_inputs{grad_output, self};
  std::vector<c10::IValue> stack = {
      IValue(grad_output),
      IValue(self),
      IValue(negative_slope),
      IValue(self_is_result)};

  auto& device = synapse_helpers::HPURegistrar::get_device(device_id);
  size_t key = Op.GetRecipeKey(node_type, stack);

  if (device.get_recipe_handle_cache().isCached(key)) {
    PT_KERNEL_DEBUG("Cache hit key:", key);
    auto output =
        at::empty(self.sizes(), self.options(), self.suggest_memory_format());
    Op.SetPTInputs(pt_inputs);
    Op.SetPTOutput(output);
    Op.Execute(key);
  } else {
    PT_KERNEL_DEBUG("key:", key);
    // Create Graph
    auto graph = habana_helpers::create_graph(device_id, node_type);
    Op.AllocateSynapseInputs(graph, pt_inputs, true);
    Op.AllocateAndAddSynapseNode(graph, stack, true);
    Op.Compile(graph);
  }
  std::vector<at::Tensor> out = Op.GetOutputs();
  TORCH_CHECK(out.size() == 1, "Incorrect size of outputs");
  auto output = out.at(0);
  if (isSelf_0d) {
    self.unsafeGetTensorImpl()->set_sizes_and_strides({}, {});
    output.unsafeGetTensorImpl()->set_sizes_and_strides({}, {});
  }
  if (isGradOutput_0d) {
    grad_output.unsafeGetTensorImpl()->set_sizes_and_strides({}, {});
  }
  PT_KERNEL_END;
  return output;
}

/*************************************************************************
 * @brief Kernel implementation for output = torch.sigmoid(input)
 * @param [out] output - output tensor, 1-4D, BF16/FP32
 * @param [in] input - input tensor, 1-4D, BF16/FP32
 ************************************************************************/
Tensor sigmoid_hpu(const Tensor& input) {
  PT_KERNEL_BEGIN;
  at::ScalarType scalar_type = input.scalar_type();
  std::string node_type =
      "sigmoid_fwd_" + habana_helpers::name_suffix_from_type(scalar_type);

  // Create the operator
  size_t device_id = input.device().index();
  SigmoidOperator Op(device_id, scalar_type);

  auto out = unary_op_hpu(input, node_type, &Op);
  PT_KERNEL_END;
  return out;
}

/*************************************************************************
 * @brief Kernel implementation for output = torch.sigmoid(grad_in, input)
 * @param [out] output - output tensor, 1-4D, BF16/FP32
 * @param [in] grad_in - input tensor, 1-4D, BF16/FP32
 * @param [in] input - input tensor, 1-4D, BF16/FP32
 ************************************************************************/
Tensor sigmoid_backward_hpu(const Tensor& grad_in, const Tensor& input) {
  PT_KERNEL_BEGIN;

  at::ScalarType scalar_type = input.scalar_type();
  std::string node_type =
      "sigmoid_bwd_" + habana_helpers::name_suffix_from_type(scalar_type);

  // Create the operator
  size_t device_id = input.device().index();
  SigmoidBackwardOperator Op(device_id, scalar_type);

  auto out = unary_backward_op_hpu(grad_in, input, node_type, &Op);

  PT_KERNEL_END;
  return out;
}

/*************************************************************************
 * @brief Kernel implementation for output = torch.sqrt(input)
 * @param [out] output - output tensor, 1-4D, BF16/FP32
 * @param [in] input - input tensor, 1-4D, BF16/FP32
 ************************************************************************/
Tensor sqrt_hpu(const Tensor& input) {
  PT_KERNEL_BEGIN;

  at::ScalarType scalar_type = input.scalar_type();
  std::string node_type =
      "sqrt_fwd_" + habana_helpers::name_suffix_from_type(scalar_type);

  // Create the operator
  size_t device_id = input.device().index();
  SqrtOperator Op(device_id, scalar_type);

  auto output = unary_op_hpu(input, node_type, &Op);

  PT_KERNEL_END;
  return output;
}

/*************************************************************************
 * @brief Kernel implementation for output = torch.tanh(input)
 * @param [out] output - output tensor, 1-4D, BF16/FP32
 * @param [in] input - input tensor, 1-4D, BF16/FP32
 ************************************************************************/
Tensor tanh_hpu(const Tensor& input) {
  PT_KERNEL_BEGIN;
  at::ScalarType scalar_type = input.scalar_type();
  std::string node_type =
      "tanh_fwd_" + habana_helpers::name_suffix_from_type(scalar_type);

  // Create the operator
  size_t device_id = input.device().index();
  TanhOperator Op(device_id, scalar_type);

  auto output = unary_op_hpu(input, node_type, &Op);
  PT_KERNEL_END;
  return output;
}

/*************************************************************************
 * @brief Kernel implementation for output = a.tanh(input)
 * @param [out] output - output tensor, 1-4D, BF16/FP32
 * @param [in] input - input tensor, 1-4D, BF16/FP32
 ************************************************************************/

Tensor& tanh_hpu_(Tensor& self) {
  PT_KERNEL_BEGIN;
  std::vector<at::Tensor> pt_inputs{self};

  synapse_simple_generic_inplace_kernel(
      pt_inputs, "tanh", nullptr, 0, SynapsePassType::FORWARD_PASS);

  PT_KERNEL_END;
  return self;
}

/*************************************************************************
 * @brief Kernel implementation for torch.tanh(input,out)
 * @param [out] output - output tensor, 1-4D, BF16/FP32
 * @param [in] input - input tensor, 1-4D, BF16/FP32
 ************************************************************************/

Tensor& tanh_out_hpu(Tensor& out, const Tensor& self) {
  PT_KERNEL_BEGIN;
  std::vector<at::Tensor> pt_inputs{self};
  std::vector<at::Tensor> pt_outputs{out};

  synapse_simple_generic_kernel(
      pt_outputs, pt_inputs, "tanh", nullptr, 0, SynapsePassType::FORWARD_PASS);

  PT_KERNEL_END;
  return out;
}

/*************************************************************************
 * @brief Kernel implementation for output = torch.tanh(grad_in, input)
 * @param [out] output - output tensor, 1-4D, BF16/FP32
 * @param [in] grad_in - input tensor, 1-4D, BF16/FP32
 * @param [in] input - input tensor, 1-4D, BF16/FP32
 ************************************************************************/
Tensor tanh_backward_hpu(const Tensor& grad_in, const Tensor& input) {
  PT_KERNEL_BEGIN;

  at::ScalarType scalar_type = input.scalar_type();
  std::string node_type =
      "tanh_bwd_" + habana_helpers::name_suffix_from_type(scalar_type);

  // Create the operator
  size_t device_id = input.device().index();
  TanhBackwardOperator Op(device_id, scalar_type);

  auto grad_output = unary_backward_op_hpu(grad_in, input, node_type, &Op);

  PT_KERNEL_END;
  return grad_output;
}

/*************************************************************************
 * @brief Kernel implementation for output = torch.floor(input)
 * @param [out] output - output tensor, 1-4D, BF16/FP32
 * @param [in] input - input tensor, 1-4D, BF16/FP32
 ************************************************************************/
Tensor floor_hpu(const Tensor& input) {
  PT_KERNEL_BEGIN;
  at::ScalarType scalar_type = input.scalar_type();
  std::string node_type =
      "floor_fwd_" + habana_helpers::name_suffix_from_type(scalar_type);

  // Create the operator
  size_t device_id = input.device().index();
  FloorOperator Op(device_id, scalar_type);

  auto out = unary_op_hpu(input, node_type, &Op);
  PT_KERNEL_END;
  return out;
}

/*************************************************************************
 * @brief Kernel implementation for output = torch.floor_(input)
 * @param [out] output - output tensor, 1-4D, BF16/FP32
 * @param [in] input - input tensor, 1-4D, BF16/FP32
 ************************************************************************/
Tensor& floor_hpu_(Tensor& self) {
  PT_KERNEL_BEGIN;

  at::ScalarType scalar_type = self.scalar_type();
  std::string node_type =
      "floor_fwd_" + habana_helpers::name_suffix_from_type(scalar_type);
  size_t device_id = self.device().index();

  // Create the operator
  FloorInplaceOperator Op(device_id, scalar_type);
  unary_inplace_op_hpu(self, node_type, &Op);

  PT_KERNEL_END;
  return self;
}

/*************************************************************************
 * @brief Kernel implementation for output = torch.log(input)
 * @param [out] output - output tensor, 1-4D, BF16/FP32
 * @param [in] input - input tensor, 1-4D, BF16/FP32
 ************************************************************************/
Tensor log_hpu(const Tensor& input) {
  PT_KERNEL_BEGIN;
  at::ScalarType scalar_type = input.scalar_type();
  std::string node_type =
      "log_fwd_" + habana_helpers::name_suffix_from_type(scalar_type);

  // Create the operator
  size_t device_id = input.device().index();
  LogOperator Op(device_id, scalar_type);

  auto out = unary_op_hpu(input, node_type, &Op);
  PT_KERNEL_END;
  return out;
}

/*************************************************************************
 * @brief Kernel implementation for output = torch.log_(input)
 * @param [out] output - output tensor, 1-4D, BF16/FP32
 * @param [in] input - input tensor, 1-4D, BF16/FP32
 ************************************************************************/
Tensor& log_hpu_(Tensor& self) {
  PT_KERNEL_BEGIN;
  at::ScalarType scalar_type = self.scalar_type();
  std::string node_type =
      "log_fwd_" + habana_helpers::name_suffix_from_type(scalar_type);
  size_t device_id = self.device().index();

  // Create the operator
  LogInplaceOperator Op(device_id, scalar_type);
  unary_inplace_op_hpu(self, node_type, &Op);
  PT_KERNEL_END;
  return self;
}

/*************************************************************************
 * @brief Kernel implementation for output = torch.log2(input)
 * @param [out] output - output tensor, 1-4D, BF16/FP32
 * @param [in] input - input tensor, 1-4D, BF16/FP32
 ************************************************************************/
Tensor log2_hpu(const Tensor& input) {
  PT_KERNEL_BEGIN;
  at::ScalarType scalar_type = input.scalar_type();
  std::string node_type =
      "log2_fwd_" + habana_helpers::name_suffix_from_type(scalar_type);

  // Create the operator
  size_t device_id = input.device().index();
  Log2Operator Op(device_id, scalar_type);

  auto out = unary_op_hpu(input, node_type, &Op);
  PT_KERNEL_END;
  return out;
}

/*************************************************************************
 * @brief Kernel implementation for output = torch.log2_(input)
 * @param [out] output - output tensor, 1-4D, BF16/FP32
 * @param [in] input - input tensor, 1-4D, BF16/FP32
 ************************************************************************/
Tensor& log2_hpu_(Tensor& self) {
  PT_KERNEL_BEGIN;

  at::ScalarType scalar_type = self.scalar_type();
  std::string node_type =
      "log2_fwd_" + habana_helpers::name_suffix_from_type(scalar_type);
  size_t device_id = self.device().index();

  // Create the operator
  Log2InplaceOperator Op(device_id, scalar_type);
  unary_inplace_op_hpu(self, node_type, &Op);
  PT_KERNEL_END;
  return self;
}

void LeakyReluOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    Stack& inputs,
    bool is_output_persistent) {
  TORCH_CHECK(
      inputs.size() == 2,
      std::string("Incorrect size of inputs expected for ") +
          (m_inplace ? "LeakyRelu_" : "LeakyRelu") + " operator");
  TORCH_CHECK(inputs[0].isTensor(), "Input 1 type expected to be tensor");
  TORCH_CHECK(inputs[1].isScalar(), "Input 2 type expected to be scalar");

  ns_LeakyReluKernel::Params param{inputs[1].toScalar().to<double>()};

  UnaryLikeOperator::AllocateAndAddSynapseNode(
      graph, inputs, is_output_persistent);
  AddNodeToSynapseGraph(graph, &param, sizeof(param));
}

void EluOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    Stack& inputs,
    bool is_output_persistent) {
  TORCH_CHECK(
      inputs.size() == 4,
      std::string("Incorrect size of inputs expected for ") +
          (m_inplace ? "Elu_" : "Elu") + " operator");
  TORCH_CHECK(inputs[0].isTensor(), "Input 1 type expected to be a tensor");
  TORCH_CHECK(inputs[1].isScalar(), "Input 2 type expected to be a scalar");
  TORCH_CHECK(inputs[2].isScalar(), "Input 3 type expected to be a scalar");
  TORCH_CHECK(inputs[3].isScalar(), "Input 4 type expected to be a scalar");

  if (inputs[2].toScalar().toFloat() != 1. or
      inputs[3].toScalar().toFloat() != 1.) {
    PT_KERNEL_WARN("Elu supports scale and input_scale as 1.");
  }

  ns_EluKernel::Params param{inputs[1].toScalar().toFloat()};

  UnaryLikeOperator::AllocateAndAddSynapseNode(
      graph, inputs, is_output_persistent);
  AddNodeToSynapseGraph(graph, &param, sizeof(param));
}

void LeakyReluBackwardOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    Stack& inputs,
    bool is_output_persistent) {
  const unsigned short constExpectedNoOfInput = 4;

  TORCH_CHECK(
      inputs.size() == constExpectedNoOfInput,
      std::string("Expected ") + std::to_string(constExpectedNoOfInput) +
          " inputs for LeakyReluBackward operator but received " +
          std::to_string(inputs.size()) + " inputs.");
  ns_LeakyReluKernel::Params param{
      inputs[2].toScalar().to<double>()}; // 3rd input is the Scalar
  auto output = habana_helpers::createPTTensor(
      inputs[0].toTensor(), is_output_persistent);
  AllocateSynapseOutput(graph, output, is_output_persistent);
  AddNodeToSynapseGraph(graph, &param, sizeof(param));
}

void GeluOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    bool is_output_persistent) {
  TORCH_CHECK(
      inputs.size() == 1,
      "Incorrect size of inputs expected for Gelu operator");
  TORCH_CHECK(
      inputs[0].isTensor(),
      "Input arg1 expected to be tensor for Gelu operator");

  auto self = inputs[0].toTensor();

  auto output1 = habana_helpers::createPTTensor(
      self,
      self.sizes(),
      self.options(),
      self.suggest_memory_format(),
      is_output_persistent);

  // TPC kernel expects two outputs first is gelu_fwd second output is tanhz
  // In graph mode we want 2nd output to be non-persistent to reduce memory
  // consumption
  auto output2 = habana_helpers::createPTTensor(
      self, self.sizes(), self.options(), self.suggest_memory_format(), false);

  std::vector<at::Tensor> outputs{output1, output2};
  AllocateSynapseOutputs(
      graph, outputs, {is_output_persistent, false}, {true, false});
  AddNodeToSynapseGraph(graph, nullptr, 0);
}

void HbGeluOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    std::vector<bool> is_output_persistent) {
  TORCH_CHECK(
      inputs.size() == 1,
      "Incorrect size of inputs expected for Gelu operator");
  TORCH_CHECK(
      inputs[0].isTensor(),
      "Input arg1 expected to be tensor for Gelu operator");

  auto self = inputs[0].toTensor();

  auto output1 = habana_helpers::createPTTensor(
      self,
      self.sizes(),
      self.options(),
      self.suggest_memory_format(),
      is_output_persistent[0]);

  // TPC kernel expects two outputs first is gelu_fwd second output is tanhz
  // In graph mode we want 2nd output to be non-persistent to reduce memory
  // consumption
  auto output2 = habana_helpers::createPTTensor(
      self,
      self.sizes(),
      self.options(),
      self.suggest_memory_format(),
      is_output_persistent[1]);

  std::vector<at::Tensor> outputs{output1, output2};
  AllocateSynapseOutputs(graph, outputs, is_output_persistent, {true, true});
  AddNodeToSynapseGraph(graph, nullptr, 0);
}

void IsfiniteOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    bool is_output_persistent) {
  TORCH_CHECK(
      inputs.size() == 1,
      "Incorrect size of inputs expected for IsfiniteOperator operator");
  TORCH_CHECK(
      inputs[0].isTensor(),
      "Input arg1 expected to be tensor for IsfiniteOperator operator");

  auto self = inputs[0].toTensor();

  auto output = habana_helpers::createPTTensor(
      self,
      self.sizes(),
      self.options(),
      self.suggest_memory_format(),
      c10::ScalarType::Bool,
      is_output_persistent);

  AllocateSynapseOutput(graph, output, is_output_persistent);
  AddNodeToSynapseGraph(graph, nullptr, 0);
}

/*************************************************************************
 * @brief Kernel implementation for gelu
 *output = 0.5 * x *(1.0 + tf.tanh(
      (np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3)))))
 * @param [out] output - output tensor, 1-4D, BF16/FP32
 * @param [in] self - input tensor, 1-4D, BF16/FP32
 ************************************************************************/
std::tuple<at::Tensor, at::Tensor> gelu2_hpu(const Tensor& self) {
  PT_KERNEL_BEGIN;

  at::ScalarType scalar_type = self.scalar_type();
  std::string node_type =
      "gelu_fwd_" + habana_helpers::name_suffix_from_type(scalar_type);

  size_t device_id = self.device().index();
  auto& device = synapse_helpers::HPURegistrar::get_device(device_id);

  // create the operator
  HbGeluOperator Op(device_id, scalar_type);

  // Build Params for the graph
  std::vector<c10::IValue> stack = {IValue(self)};
  size_t key = Op.GetRecipeKey(node_type, stack);

  // Assign Inputs to the Operator
  std::vector<at::Tensor> pt_inputs{self};

  if (device.get_recipe_handle_cache().isCached(key)) {
    PT_KERNEL_DEBUG("Cache hit key:", key);
    // TPC kernel expects two outputs first is gelu_fwd second output is tanhz
    auto output1 =
        at::empty(self.sizes(), self.options(), self.suggest_memory_format());
    auto output2 =
        at::empty(self.sizes(), self.options(), self.suggest_memory_format());

    Op.SetPTInputs(pt_inputs);
    std::vector<at::Tensor> v{output1, output2};
    Op.SetPTOutputs(v);
    Op.Execute(key);
  } else {
    PT_KERNEL_DEBUG("Key:", key);
    // Create Graph
    auto graph = habana_helpers::create_graph(device_id, node_type);
    Op.AllocateSynapseInputs(graph, pt_inputs, true);
    Op.AllocateAndAddSynapseNode(graph, stack, {true, true});
    // compile and execute the graph
    Op.Compile(graph);
  }

  std::vector<at::Tensor> out = Op.GetOutputs();
  TORCH_CHECK(out.size() == 2, "Incorrect size of outputs");

  PT_KERNEL_END;
  return std::make_tuple(out[0], out[1]);
}

Tensor gelu_hpu(const Tensor& self) {
  PT_KERNEL_BEGIN;
  auto out = gelu2_hpu(self);
  PT_KERNEL_END;
  return std::get<0>(out);
}

void GeluBackwardOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    bool is_output_persistent) {
  TORCH_CHECK(
      inputs.size() == 2 || inputs.size() == 3,
      "Incorrect size of inputs expected for Gelu Backward operator");
  TORCH_CHECK(
      inputs[0].isTensor(),
      "Input arg1 expected to be tensor for Gelu Backward operator");
  TORCH_CHECK(
      inputs[1].isTensor(),
      "Input arg2 expected to be tensor for Gelu Backward operator");

  auto grad = inputs[0].toTensor();
  auto self = inputs[1].toTensor();
  at::ScalarType scalar_type = self.scalar_type();

  if (inputs.size() == 2) {
    // x^3 implemented as x*x*x. Identity node used to create aliased tensor
    // since GC/TPC does not like giving same tensor as both inputs to a
    // binary op
    auto identityOp = make_operator<IdentityOperator>(
        this->p_context_->device_id_, scalar_type);
    identityOp->SetSynapseInput(p_context_->syn_inputs_[1]);
    torch::jit::Stack stack = {IValue(self)};
    identityOp->AllocateAndAddSynapseNode(graph, stack, false);
    stack.clear();

    auto mulpow1Op =
        make_operator<MulOperator>(this->p_context_->device_id_, scalar_type);
    mulpow1Op->SetSynapseInput(p_context_->syn_inputs_[1]);
    mulpow1Op->SetSynapseInput(identityOp->GetSynOutputs()[0]);
    stack.emplace_back(IValue(self));
    stack.emplace_back(IValue(identityOp->GetOutputs()[0]));
    mulpow1Op->AllocateAndAddSynapseNode(graph, stack, false);
    stack.clear();

    auto mulpow2Op =
        make_operator<MulOperator>(this->p_context_->device_id_, scalar_type);
    mulpow2Op->SetSynapseInput(p_context_->syn_inputs_[1]);
    mulpow2Op->SetSynapseInput(mulpow1Op->GetSynOutputs()[0]);
    stack.emplace_back(IValue(self));
    stack.emplace_back(IValue(mulpow1Op->GetOutputs()[0]));
    mulpow2Op->AllocateAndAddSynapseNode(graph, stack, false);
    stack.clear();

    // Create Add operator
    auto addOp =
        make_operator<AddOperator>(this->p_context_->device_id_, scalar_type);
    addOp->SetSynapseInput(p_context_->syn_inputs_[1]);
    addOp->SetSynapseInput(mulpow2Op->GetSynOutputs()[0]);
    // Build Params for the graph
    Scalar alphaValue = 0.044715;
    stack.emplace_back(IValue(self));
    stack.emplace_back(IValue(mulpow2Op->GetOutputs()[0]));
    stack.emplace_back(IValue(alphaValue));
    addOp->AllocateAndAddSynapseNode(graph, stack, false);
    stack.clear();

    // Create Mul operator
    auto mulOp =
        make_operator<MulOperator>(this->p_context_->device_id_, scalar_type);
    mulOp->SetSynapseInput(addOp->GetSynOutputs()[0]);
    // Build Params for the graph
    Scalar alphaValue_2 = M_2_SQRTPI * M_SQRT1_2;
    stack.emplace_back(IValue(addOp->GetOutputs()[0]));
    stack.emplace_back(IValue(alphaValue_2));
    mulOp->AllocateAndAddSynapseNode(graph, stack, false);
    stack.clear();

    // Create Tanh operator
    auto tanhOp =
        make_operator<TanhOperator>(this->p_context_->device_id_, scalar_type);
    tanhOp->SetSynapseInput(mulOp->GetSynOutputs()[0]);
    // Build Params for the graph
    stack.emplace_back(IValue(mulOp->GetOutputs()[0]));
    tanhOp->AllocateAndAddSynapseNode(graph, stack, false);
    stack.clear();

    auto output = habana_helpers::createPTTensor(
        self,
        self.sizes(),
        self.options(),
        self.suggest_memory_format(),
        is_output_persistent);
    AllocateSynapseOutput(graph, output, is_output_persistent);
    synapse_helpers::tensor& synOutput = p_context_->syn_outputs_[0];

    synapse_helpers::tensor& synInput0 = p_context_->syn_inputs_[0];
    synapse_helpers::tensor& synInput1 = p_context_->syn_inputs_[1];
    synapse_helpers::tensor& synInput2 = tanhOp->GetSynOutputs()[0];

    std::vector<synTensor> syn_in{
        synInput0.get(), synInput1.get(), synInput2.get()};
    std::vector<synTensor> syn_out{synOutput.get()};

    graph.add_node(
        std::move(syn_in), std::move(syn_out), nullptr, 0, std::move(guid_));
  } else {
    auto output = habana_helpers::createPTTensor(
        self,
        self.sizes(),
        self.options(),
        self.suggest_memory_format(),
        is_output_persistent);
    AllocateSynapseOutput(graph, output, is_output_persistent);
    synapse_helpers::tensor& synOutput = p_context_->syn_outputs_[0];

    synapse_helpers::tensor& synInput0 = p_context_->syn_inputs_[0];
    synapse_helpers::tensor& synInput1 = p_context_->syn_inputs_[1];
    synapse_helpers::tensor& synInput2 = p_context_->syn_inputs_[2];

    std::vector<synTensor> syn_in{
        synInput0.get(), synInput1.get(), synInput2.get()};
    std::vector<synTensor> syn_out{synOutput.get()};

    graph.add_node(
        std::move(syn_in), std::move(syn_out), nullptr, 0, std::move(guid_));
  }
}

/*************************************************************************
 * @brief Kernel implementation for gelu_backward
 * @param [out] output - bwd output tensor, 1-4D, BF16/FP32
 * @param [in] grad - bwd input tensor, 1-4D, BF16/FP32
 * @param [in] self - input tensor, 1-4D, BF16/FP32
 ************************************************************************/
Tensor gelu2_backward_hpu(
    const Tensor& grad,
    const Tensor& self,
    const Tensor& saved) {
  PT_KERNEL_BEGIN;

  at::ScalarType scalar_type = self.scalar_type();
  std::string node_type =
      "gelu_bwd_" + habana_helpers::name_suffix_from_type(scalar_type);

  size_t device_id = self.device().index();
  auto& device = synapse_helpers::HPURegistrar::get_device(device_id);

  // create the operator
  GeluBackwardOperator Op(device_id, scalar_type);

  // Build Params for the graph
  std::vector<c10::IValue> stack = {IValue(grad), IValue(self), IValue(saved)};
  size_t key = Op.GetRecipeKey(node_type, stack);

  // Assign Inputs to the Operator
  std::vector<at::Tensor> pt_inputs{grad, self, saved};

  if (device.get_recipe_handle_cache().isCached(key)) {
    PT_KERNEL_DEBUG("Cache hit key:", key);
    auto result =
        at::empty(self.sizes(), self.options(), self.suggest_memory_format());
    Op.SetPTInputs(pt_inputs);
    Op.SetPTOutput(result);
    Op.Execute(key);
  } else {
    PT_KERNEL_DEBUG("Key:", key);
    // Create Graph
    auto graph = habana_helpers::create_graph(device_id, node_type);
    Op.AllocateSynapseInputs(graph, pt_inputs, true);
    Op.AllocateAndAddSynapseNode(graph, stack, true);
    // compile and execute the graph
    Op.Compile(graph);
  }

  std::vector<at::Tensor> out = Op.GetOutputs();
  TORCH_CHECK(out.size() == 1, "Incorrect size of outputs");

  PT_KERNEL_END;
  return out.at(0);
}

Tensor gelu_backward_hpu(const Tensor& grad, const Tensor& self) {
  PT_KERNEL_BEGIN;

  at::ScalarType scalar_type = self.scalar_type();
  std::string node_type =
      "gelu_bwd_" + habana_helpers::name_suffix_from_type(scalar_type);

  size_t device_id = self.device().index();
  auto& device = synapse_helpers::HPURegistrar::get_device(device_id);

  // create the operator
  GeluBackwardOperator Op(device_id, scalar_type);

  // Build Params for the graph
  std::vector<c10::IValue> stack = {IValue(grad), IValue(self)};
  size_t key = Op.GetRecipeKey(node_type, stack);

  // Assign Inputs to the Operator
  std::vector<at::Tensor> pt_inputs{grad, self};

  if (device.get_recipe_handle_cache().isCached(key)) {
    PT_KERNEL_DEBUG("Cache hit key:", key);
    auto result =
        at::empty(self.sizes(), self.options(), self.suggest_memory_format());
    Op.SetPTInputs(pt_inputs);
    Op.SetPTOutput(result);
    Op.Execute(key);
  } else {
    PT_KERNEL_DEBUG("Key:", key);
    // Create Graph
    auto graph = habana_helpers::create_graph(device_id, node_type);
    Op.AllocateSynapseInputs(graph, pt_inputs, true);
    Op.AllocateAndAddSynapseNode(graph, stack, true);
    // compile and execute the graph
    Op.Compile(graph);
  }

  std::vector<at::Tensor> out = Op.GetOutputs();
  TORCH_CHECK(out.size() == 1, "Incorrect size of outputs");

  PT_KERNEL_END;
  return out.at(0);
}

Tensor& idop_hpu_(Tensor& self) {
  PT_KERNEL_BEGIN;
  PT_KERNEL_END;
  return self;
}

Tensor idop_hpu(const Tensor& self) {
  PT_KERNEL_BEGIN;
  auto result = const_cast<Tensor&>(self);
  PT_KERNEL_END;
  return result;
}

/*************************************************************************
 * @brief Kernel implementation for output = torch.erf(input)
 * @param [out] output - output tensor, 1-4D, BF16/FP32
 * @param [in] input - input tensor, 1-4D, BF16/FP32
 ************************************************************************/
Tensor erf_hpu(const Tensor& input) {
  PT_KERNEL_BEGIN;
  at::ScalarType scalar_type = input.scalar_type();
  std::string node_type =
      "erf_fwd_" + habana_helpers::name_suffix_from_type(scalar_type);

  // Create the operator
  size_t device_id = input.device().index();
  ErfOperator Op(device_id, scalar_type);

  auto out = unary_op_hpu(input, node_type, &Op);
  PT_KERNEL_END;
  return out;
}

/*************************************************************************
 * @brief Kernel implementation for output = torch.erf_(input)
 * @param [out] output - output tensor, 1-4D, BF16/FP32
 * @param [in] input - input tensor, 1-4D, BF16/FP32
 ************************************************************************/
Tensor& erf_hpu_(Tensor& self) {
  PT_KERNEL_BEGIN;

  at::ScalarType scalar_type = self.scalar_type();
  std::string node_type =
      "erf_fwd_" + habana_helpers::name_suffix_from_type(scalar_type);
  size_t device_id = self.device().index();

  // Create the operator
  ErfInplaceOperator Op(device_id, scalar_type);
  unary_inplace_op_hpu(self, node_type, &Op);
  PT_KERNEL_END;
  return self;
}

/*************************************************************************
 * @brief Kernel implementation for output = torch.exp(input)
 * @param [out] output - output tensor, 1-4D, BF16/FP32
 * @param [in] input - input tensor, 1-4D, BF16/FP32
 ************************************************************************/
Tensor exp_hpu(const Tensor& input) {
  PT_KERNEL_BEGIN;
  at::ScalarType scalar_type = input.scalar_type();
  std::string node_type =
      "exp_fwd_" + habana_helpers::name_suffix_from_type(scalar_type);

  // Create the operator
  size_t device_id = input.device().index();
  ExpOperator Op(device_id, scalar_type);

  auto out = unary_op_hpu(input, node_type, &Op);
  PT_KERNEL_END;
  return out;
}

/*************************************************************************
 * @brief Kernel implementation for output = torch.exp_(input)
 * @param [out] output - output tensor, 1-4D, BF16/FP32
 * @param [in] input - input tensor, 1-4D, BF16/FP32
 ************************************************************************/
Tensor& exp_hpu_(Tensor& self) {
  PT_KERNEL_BEGIN;

  at::ScalarType scalar_type = self.scalar_type();
  std::string node_type =
      "exp_fwd_" + habana_helpers::name_suffix_from_type(scalar_type);
  size_t device_id = self.device().index();

  // Create the operator
  ExpInplaceOperator Op(device_id, scalar_type);
  unary_inplace_op_hpu(self, node_type, &Op);

  PT_KERNEL_END;
  return self;
}

/*************************************************************************
 * @brief Kernel implementation for torch.neg(input,out)
 * @param [out] out - output tensor, 1-4D, BF16/FP32
 * @param [in] input - input tensor, 1-4D, BF16/FP32
 ************************************************************************/
Tensor& neg_out_hpu(Tensor& result, const Tensor& input) {
  PT_KERNEL_BEGIN;

  // Resize result to correct size (if required)
  auto shape = DimVector(input.sizes());
  auto tht_result = result.unsafeGetTensorImpl();
  THHTensor_resizeNd(tht_result, shape.size(), shape.data(), nullptr);

  std::vector<at::Tensor> pt_outputs{result};
  std::vector<at::Tensor> pt_inputs{input};

  synapse_simple_generic_kernel(
      pt_outputs, pt_inputs, "neg", nullptr, 0, SynapsePassType::FORWARD_PASS);

  PT_KERNEL_END;
  return result;
}

/*************************************************************************
 * @brief Kernel implementation for inplace torch.reciprocal_(self)
 * @param [in] self - input tensor, 1-4D, BF16/FP32
 ************************************************************************/
Tensor& reciprocal_hpu_(Tensor& self) {
  PT_KERNEL_BEGIN;
  std::vector<at::Tensor> pt_inputs{self};

  synapse_simple_generic_inplace_kernel(
      pt_inputs, "reciprocal", nullptr, 0, SynapsePassType::FORWARD_PASS);

  PT_KERNEL_END;
  return self;
}

void ReciprocalOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    bool is_output_persistent) {
  TORCH_CHECK(
      inputs.size() == 1,
      "Incorrect size of inputs expected for Reciprocal operator");
  TORCH_CHECK(
      inputs[0].isTensor(),
      "Input arg1 expected to be tensor for Reciprocal operator");

  auto self = inputs[0].toTensor();
  auto result = habana_helpers::createPTTensor(self, is_output_persistent);
  inputs.insert(inputs.begin(), IValue(result));

  ReciprocalOutOperator::AllocateAndAddSynapseNode(
      graph, inputs, is_output_persistent);
}

/*************************************************************************
 * @brief Kernel implementation for output = torch.reciprocal(self)
 * @param [out] output - output tensor, 1-4D, BF16/FP32
 * @param [in] self - input tensor, 1-4D, BF16/FP32
 ************************************************************************/
Tensor reciprocal_hpu(const Tensor& self) {
  PT_KERNEL_BEGIN;

  at::ScalarType scalar_type = self.scalar_type();
  std::string node_type =
      "reciprocal_fwd_" + habana_helpers::name_suffix_from_type(scalar_type);

  size_t device_id = self.device().index();
  auto& device = synapse_helpers::HPURegistrar::get_device(device_id);

  // create the operator
  ReciprocalOperator Op(device_id, scalar_type);

  // Build Params for the graph
  std::vector<c10::IValue> stack = {IValue(self)};
  size_t key = Op.GetRecipeKey(node_type, stack);

  // Assign Inputs to the Operator
  std::vector<at::Tensor> pt_inputs{self};

  if (device.get_recipe_handle_cache().isCached(key)) {
    PT_KERNEL_DEBUG("Cache hit key:", key);
    auto result =
        at::empty(self.sizes(), self.options(), self.suggest_memory_format());
    Op.SetPTInputs(pt_inputs);
    Op.SetPTOutput(result);
    Op.Execute(key);
  } else {
    PT_KERNEL_DEBUG("Key:", key);
    // Create Graph
    auto graph = habana_helpers::create_graph(device_id, node_type);
    Op.AllocateSynapseInputs(graph, pt_inputs, true);
    Op.AllocateAndAddSynapseNode(graph, stack, true);
    // compile and execute the graph
    Op.Compile(graph);
  }

  std::vector<at::Tensor> out = Op.GetOutputs();
  TORCH_CHECK(out.size() == 1, "Incorrect size of outputs");

  PT_KERNEL_END;
  return out.at(0);
}

void ReciprocalOutOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    bool is_output_persistent) {
  TORCH_CHECK(
      inputs.size() == 2,
      "Incorrect size of inputs expected for ReciprocalOut operator");
  TORCH_CHECK(
      inputs[0].isTensor(),
      "Input arg1 expected to be tensor for ReciprocalOut operator");
  TORCH_CHECK(
      inputs[1].isTensor(),
      "Input arg2 expected to be tensor for ReciprocalOut operator");

  auto result = inputs[0].toTensor();
  auto self = inputs[1].toTensor();

  auto shape = DimVector(self.sizes());
  auto tht_result = result.unsafeGetTensorImpl();
  THHTensor_resizeNd(tht_result, shape.size(), shape.data(), nullptr);

  AllocateSynapseOutput(graph, result, is_output_persistent);
  AddNodeToSynapseGraph(graph, nullptr, 0);
}

/*************************************************************************
 * @brief Kernel implementation for torch.reciprocal(self,out)
 * @param [out] out - output tensor, 1-4D, BF16/FP32
 * @param [in] self - input tensor, 1-4D, BF16/FP32
 ************************************************************************/
Tensor& reciprocal_out_hpu(Tensor& result, const Tensor& self) {
  PT_KERNEL_BEGIN;

  at::ScalarType scalar_type = self.scalar_type();
  std::string node_type =
      "reciprocal_fwd_" + habana_helpers::name_suffix_from_type(scalar_type);

  size_t device_id = self.device().index();
  auto& device = synapse_helpers::HPURegistrar::get_device(device_id);

  ReciprocalOutOperator Op(device_id, scalar_type);

  // Build Params for the graph
  std::vector<c10::IValue> stack = {IValue(result), IValue(self)};
  size_t key = Op.GetRecipeKey(node_type, stack);

  // Assign Inputs to the Operator
  std::vector<at::Tensor> pt_inputs{self};

  if (device.get_recipe_handle_cache().isCached(key)) {
    PT_KERNEL_DEBUG("Cache hit key:", key);
    Op.SetPTInputs(pt_inputs);
    Op.SetPTOutput(result);
    Op.Execute(key);
  } else {
    PT_KERNEL_DEBUG("Key:", key);
    // Create Graph
    auto graph = habana_helpers::create_graph(device_id, node_type);
    Op.AllocateSynapseInputs(graph, pt_inputs, true);
    Op.AllocateAndAddSynapseNode(graph, stack, true);
    // compile and execute the graph
    Op.Compile(graph);
  }

  PT_KERNEL_END;
  return result;
}

void ClampOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    Stack& inputs,
    bool is_output_persistent) {
  TORCH_CHECK(
      inputs.size() == 3,
      "Incorrect size of inputs expected for Clamp operator");
  TORCH_CHECK(inputs[0].isTensor(), "Input type expected to be tensor");

  auto self = inputs[0].toTensor();
  auto min = inputs[1].isScalar() ? inputs[1].toScalar()
                                  : inputs[1].toOptional<Scalar>();
  auto max = inputs[2].isScalar() ? inputs[2].toScalar()
                                  : inputs[2].toOptional<Scalar>();

  ns_ClampKernel::Params param;
  param.upperBound.f = max.has_value() ? max.value().to<float>()
                                       : std::numeric_limits<float>::max();
  param.lowerBound.f = min.has_value() ? min.value().to<float>()
                                       : -std::numeric_limits<float>::max();

  if (self.scalar_type() == c10::ScalarType::Int) {
    // Guid needs to be updated since TPC only supports F32/BF16
    SetGuid("clamp_fwd_f32");
    // Cast Input tensor to Float tensor
    std::string node_type = "cast_i32_to_f32";

    // Create the operator
    auto intToFloatOp =
        make_operator<CastOperator>(this->p_context_->device_id_, node_type);
    intToFloatOp->SetSynapseInput(p_context_->syn_inputs_[0]);

    // Build Params for the graph
    std::vector<c10::IValue> stack{
        IValue(self), IValue(c10::ScalarType::Float)};
    intToFloatOp->AllocateAndAddSynapseNode(graph, stack, false);

    synapse_helpers::tensor& float_syn_tensor =
        intToFloatOp->GetSynOutputs()[0];
    auto output_float = intToFloatOp->GetOutputs()[0];
    stack.clear();

    AllocateSynapseOutput(graph, output_float, false);
    synapse_helpers::tensor& synOutput = p_context_->syn_outputs_[0];

    std::vector<synTensor> syn_in{float_syn_tensor.get()};
    std::vector<synTensor> syn_out{synOutput.get()};

    node_type = "clamp_fwd_f32";
    graph.add_node(
        std::move(syn_in),
        std::move(syn_out),
        &param,
        sizeof(param),
        std::move(node_type));

    node_type = "cast_f32_to_i32";
    // Create cast operator
    auto floatToIntOp =
        make_operator<CastOperator>(this->p_context_->device_id_, node_type);
    floatToIntOp->SetSynapseInput(p_context_->syn_outputs_[0]);
    floatToIntOp->SetOutputMetadata(output_metadata_);

    // Build Params for the graph
    stack = {IValue(p_context_->pt_outputs_[0]), IValue(c10::ScalarType::Int)};

    floatToIntOp->AllocateAndAddSynapseNode(graph, stack, is_output_persistent);
    p_context_->syn_outputs_[0] = std::move(floatToIntOp->GetSynOutputs()[0]);
    p_context_->pt_outputs_[0] = std::move(floatToIntOp->GetOutputs()[0]);

  } else {
    auto output = habana_helpers::createPTTensor(self, is_output_persistent);
    AllocateSynapseOutput(graph, output, is_output_persistent);
    AddNodeToSynapseGraph(graph, &param, sizeof(param));
  }
}

void ClampMinOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    Stack& inputs,
    bool is_output_persistent) {
  TORCH_CHECK(
      inputs.size() == 2,
      "Incorrect size of inputs expected for Clamp operator");
  TORCH_CHECK(inputs[0].isTensor(), "Input type expected to be tensor");

  auto self = inputs[0].toTensor();
  auto min = inputs[1].isScalar() ? inputs[1].toScalar()
                                  : inputs[1].toOptional<Scalar>();

  ns_ClampKernel::Params param;
  param.upperBound.f = std::numeric_limits<float>::max();
  param.lowerBound.f = min.has_value() ? min.value().to<float>()
                                       : -std::numeric_limits<float>::max();

  if (self.scalar_type() == c10::ScalarType::Int) {
    // Guid needs to be updated since TPC only supports F32/BF16
    SetGuid("clamp_fwd_f32");
    // Cast Input tensor to Float tensor
    std::string node_type = "cast_i32_to_f32";

    // Create the operator
    auto intToFloatOp =
        make_operator<CastOperator>(this->p_context_->device_id_, node_type);
    intToFloatOp->SetSynapseInput(p_context_->syn_inputs_[0]);

    // Build Params for the graph
    std::vector<c10::IValue> stack{
        IValue(self), IValue(c10::ScalarType::Float)};
    intToFloatOp->AllocateAndAddSynapseNode(graph, stack, false);

    synapse_helpers::tensor& float_syn_tensor =
        intToFloatOp->GetSynOutputs()[0];
    auto output_float = intToFloatOp->GetOutputs()[0];
    stack.clear();

    AllocateSynapseOutput(graph, output_float, false);
    synapse_helpers::tensor& synOutput = p_context_->syn_outputs_[0];

    std::vector<synTensor> syn_in{float_syn_tensor.get()};
    std::vector<synTensor> syn_out{synOutput.get()};

    node_type = "clamp_fwd_f32";
    graph.add_node(
        std::move(syn_in),
        std::move(syn_out),
        &param,
        sizeof(param),
        std::move(node_type));

    node_type = "cast_f32_to_i32";
    // Create cast operator
    auto floatToIntOp =
        make_operator<CastOperator>(this->p_context_->device_id_, node_type);
    floatToIntOp->SetSynapseInput(p_context_->syn_outputs_[0]);
    floatToIntOp->SetOutputMetadata(output_metadata_);
    // Build Params for the graph
    stack = {IValue(p_context_->pt_outputs_[0]), IValue(c10::ScalarType::Int)};

    floatToIntOp->AllocateAndAddSynapseNode(graph, stack, is_output_persistent);
    p_context_->syn_outputs_[0] = std::move(floatToIntOp->GetSynOutputs()[0]);
    p_context_->pt_outputs_[0] = std::move(floatToIntOp->GetOutputs()[0]);

  } else {
    auto output = habana_helpers::createPTTensor(self, is_output_persistent);
    AllocateSynapseOutput(graph, output, is_output_persistent);
    AddNodeToSynapseGraph(graph, &param, sizeof(param));
  }
}

/** @brief This function implements torch.clamp_min()
 * @param self (bf16, fp32 tensor) Input tensor
 * @param min (int, float) Minimum value at which input will be clamped
 */
Tensor clamp_min_hpu(const Tensor& self, Scalar min) {
  PT_KERNEL_BEGIN;
  at::ScalarType scalar_type = self.scalar_type();
  std::string node_type =
      "clamp_fwd_" + habana_helpers::name_suffix_from_type(scalar_type);

  size_t device_id = self.device().index();

  ClampOperator Op(device_id, scalar_type);

  // Create Graph
  auto graph = habana_helpers::create_graph(device_id, node_type);

  // Assign Inputs to the Operator
  std::vector<at::Tensor> pt_inputs{self};
  Op.AllocateSynapseInputs(graph, pt_inputs, true);

  // Build Params for the graph
  std::vector<c10::IValue> stack = {
      IValue(self),
      IValue(min),
      IValue(Scalar(std::numeric_limits<float>::max()))};
  Op.AllocateAndAddSynapseNode(graph, stack, true);

  // compile and execute the graph
  Op.Compile(graph);

  std::vector<at::Tensor> out = Op.GetOutputs();
  TORCH_CHECK(out.size() == 1, "Incorrect size of outputs");

  PT_KERNEL_END;
  return out.at(0);
}

Tensor clamp_hpu(
    const Tensor& self,
    const c10::optional<Scalar>& min,
    const c10::optional<Scalar>& max) {
  PT_KERNEL_BEGIN;
  at::ScalarType scalar_type = self.scalar_type();
  std::string node_type =
      "clamp_fwd_" + habana_helpers::name_suffix_from_type(scalar_type);
  size_t device_id = self.device().index();
  auto& device = synapse_helpers::HPURegistrar::get_device(device_id);
  ClampOperator Op(device_id, scalar_type);

  std::vector<at::Tensor> pt_inputs;
  std::vector<c10::IValue> stack;
  if (self.scalar_type() == ScalarType::Long) {
    auto self_i32 = habana_helpers::cast_tensor_to_integer(self);
    pt_inputs = {self_i32};
    stack = {IValue(self_i32), IValue(min), IValue(max)};
  } else {
    pt_inputs = {self};
    stack = {IValue(self), IValue(min), IValue(max)};
  }

  size_t key = Op.GetRecipeKey(node_type, stack);
  if (device.get_recipe_handle_cache().isCached(key)) {
    PT_KERNEL_DEBUG("Cache hit key:", key);
    Op.SetPTInputs(pt_inputs);

    if (self.scalar_type() == ScalarType::Long) {
      auto output = at::empty(
          self.sizes(),
          self.options().dtype(c10::ScalarType::Int),
          self.suggest_memory_format());
      Op.SetPTOutput(output);
    } else {
      auto output =
          at::empty(self.sizes(), self.options(), self.suggest_memory_format());
      Op.SetPTOutput(output);
    }
    Op.Execute(key);
  } else {
    PT_KERNEL_DEBUG("Key:", key);
    // Create Graph
    auto graph = habana_helpers::create_graph(device_id, node_type);
    Op.AllocateSynapseInputs(graph, pt_inputs, true);
    Op.AllocateAndAddSynapseNode(graph, stack, true);
    // compile and execute the graph
    Op.Compile(graph);
  }

  std::vector<at::Tensor> out = Op.GetOutputs();
  TORCH_CHECK(out.size() == 1, "Incorrect size of outputs");
  if (self.scalar_type() == ScalarType::Long) {
    auto output = habana_helpers::cast_tensor_to_long(out.at(0));
    PT_KERNEL_END;
    return output;
  } else {
    PT_KERNEL_END;
    return out.at(0);
  }
}

void ClampInplaceOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    Stack& inputs,
    bool is_output_persistent) {
  static_cast<void>(is_output_persistent);
  TORCH_CHECK(
      inputs.size() == 3,
      "Incorrect size of inputs expected for Clamp operator");
  TORCH_CHECK(inputs[0].isTensor(), "Input type expected to be tensor");

  auto input = inputs[0].toTensor();
  auto min = inputs[1].isScalar() ? inputs[1].toScalar()
                                  : inputs[1].toOptional<Scalar>();
  auto max = inputs[2].isScalar() ? inputs[2].toScalar()
                                  : inputs[2].toOptional<Scalar>();

  ns_ClampKernel::Params param;
  param.upperBound.f = max.has_value() ? max.value().to<float>()
                                       : std::numeric_limits<float>::max();
  param.lowerBound.f = min.has_value() ? min.value().to<float>()
                                       : -std::numeric_limits<float>::max();
  if (p_context_->pt_inputs_.size() == 0)
    p_context_->pt_inputs_.emplace_back(inputs[0].toTensor());
  AllocateSynapseInplaceOutput(graph);
  AddNodeToSynapseGraph(graph, &param, sizeof(param));
}

Tensor& clamp_hpu_(
    Tensor& self,
    const c10::optional<Scalar>& min,
    const c10::optional<Scalar>& max) {
  PT_KERNEL_BEGIN;

  if ((self.scalar_type() == ScalarType::Long) ||
      (self.scalar_type() == ScalarType::Int)) {
    self.copy_(clamp_hpu(self, min, max));
    PT_KERNEL_END;
    return self;
  }
  at::ScalarType scalar_type = self.scalar_type();
  std::string node_type =
      "clamp_fwd_" + habana_helpers::name_suffix_from_type(scalar_type);

  size_t device_id = self.device().index();
  auto& device = synapse_helpers::HPURegistrar::get_device(device_id);
  ClampInplaceOperator Op(device_id, scalar_type);
  // Assign Inputs to the Operator
  std::vector<at::Tensor> pt_inputs{self};
  // Build Params for the graph
  std::vector<c10::IValue> stack = {IValue(self), IValue(min), IValue(max)};

  size_t key = Op.GetRecipeKey(node_type, stack, true);
  if (device.get_recipe_handle_cache().isCached(key)) {
    PT_KERNEL_DEBUG("Cache hit key:", key);
    Op.SetPTInputs(pt_inputs);
    Op.SetPTOutput(self);
    Op.Execute(key);
  } else {
    PT_KERNEL_DEBUG("Key:", key);
    // Create Graph
    auto graph = habana_helpers::create_graph(device_id, node_type);
    Op.AllocateSynapseInputs(graph, pt_inputs, true);
    Op.AllocateAndAddSynapseNode(graph, stack, true);
    // compile and execute the graph
    Op.Compile(graph);
  }

  PT_KERNEL_END;
  return self;
}

/*************************************************************************
 * @brief Kernel implementation for output = torch.abs(self)
 * @param [out] output - output tensor, 1-4D, BF16/FP32
 * @param [in] self - input tensor, 1-4D, BF16/FP32
 ************************************************************************/
Tensor abs_hpu(const Tensor& self) {
  PT_KERNEL_BEGIN;

  at::ScalarType scalar_type = self.scalar_type();
  std::string node_type =
      "abs_fwd_" + habana_helpers::name_suffix_from_type(scalar_type);

  // Create the operator
  size_t device_id = self.device().index();
  AbsOperator Op(device_id, scalar_type);

  auto out = unary_op_hpu(self, node_type, &Op);

  PT_KERNEL_END;
  return out;
}

/*************************************************************************
 * @brief Kernel implementation for output = torch.abs_(input)
 * @param [out] output - output tensor, 1-4D, BF16/FP32
 * @param [in] input - input tensor, 1-4D, BF16/FP32
 ************************************************************************/
Tensor& abs_hpu_(Tensor& self) {
  PT_KERNEL_BEGIN;

  at::ScalarType scalar_type = self.scalar_type();
  std::string node_type =
      "abs_fwd_" + habana_helpers::name_suffix_from_type(scalar_type);
  size_t device_id = self.device().index();

  // Create the operator
  AbsInplaceOperator Op(device_id, scalar_type);
  unary_inplace_op_hpu(self, node_type, &Op);

  PT_KERNEL_END;
  return self;
}

/*************************************************************************
 * @brief Kernel implementation for output = torch.round(self)
 * @param [out] output - output tensor, 1-4D, BF16/FP32
 * @param [in] self - input tensor, 1-4D, BF16/FP32
 ************************************************************************/
Tensor round_hpu(const Tensor& self) {
  PT_KERNEL_BEGIN;

  at::ScalarType scalar_type = self.scalar_type();
  std::string node_type =
      "round_fwd_" + habana_helpers::name_suffix_from_type(scalar_type);

  // Create the operator
  size_t device_id = self.device().index();
  RoundOperator Op(device_id, scalar_type);

  auto out = unary_op_hpu(self, node_type, &Op);

  PT_KERNEL_END;
  return out;
}

/*************************************************************************
 * @brief Kernel implementation for output = torch.round_(input)
 * @param [out] output - output tensor, 1-4D, BF16/FP32
 * @param [in] input - input tensor, 1-4D, BF16/FP32
 ************************************************************************/
Tensor& round_hpu_(Tensor& self) {
  PT_KERNEL_BEGIN;

  at::ScalarType scalar_type = self.scalar_type();
  std::string node_type =
      "round_fwd_" + habana_helpers::name_suffix_from_type(scalar_type);
  size_t device_id = self.device().index();

  // Create the operator
  RoundInplaceOperator Op(device_id, scalar_type);
  unary_inplace_op_hpu(self, node_type, &Op);

  PT_KERNEL_END;
  return self;
}

/*************************************************************************
 * @brief Kernel implementation for output = torch.rsqrt(self)
 * @param [out] output - output tensor, 1-4D, BF16/FP32
 * @param [in] self - input tensor, 1-4D, BF16/FP32
 ************************************************************************/
Tensor rsqrt_hpu(const Tensor& self) {
  PT_KERNEL_BEGIN;

  at::ScalarType scalar_type = self.scalar_type();
  std::string node_type =
      "rsqrt_fwd_" + habana_helpers::name_suffix_from_type(scalar_type);

  // Create the operator
  size_t device_id = self.device().index();
  RsqrtOperator Op(device_id, scalar_type);

  auto out = unary_op_hpu(self, node_type, &Op);

  PT_KERNEL_END;
  return out;
}

/*************************************************************************
 * @brief Kernel implementation for output = torch.rsqrt_(input)
 * @param [out] output - output tensor, 1-4D, BF16/FP32
 * @param [in] input - input tensor, 1-4D, BF16/FP32
 ************************************************************************/
Tensor& rsqrt_hpu_(Tensor& self) {
  PT_KERNEL_BEGIN;

  at::ScalarType scalar_type = self.scalar_type();
  std::string node_type =
      "rsqrt_fwd_" + habana_helpers::name_suffix_from_type(scalar_type);
  size_t device_id = self.device().index();

  // Create the operator
  RsqrtInplaceOperator Op(device_id, scalar_type);
  unary_inplace_op_hpu(self, node_type, &Op);

  PT_KERNEL_END;
  return self;
}

/*************************************************************************
 * @brief Kernel implementation for output = torch.isfinite(self)
 * @param [out] output - output tensor, 1-4D, BF16/FP32
 * @param [in] self - input tensor, 1-4D, BF16/FP32
 ************************************************************************/
Tensor isfinite_hpu(const Tensor& self) {
  PT_KERNEL_BEGIN;

  at::ScalarType scalar_type = self.scalar_type();
  std::string node_type =
      "isfinite_fwd_" + habana_helpers::name_suffix_from_type(scalar_type);

  // Create the operator
  size_t device_id = self.device().index();
  IsfiniteOperator Op(device_id, scalar_type);

  std::vector<c10::IValue> stack = {IValue(self)};
  std::vector<at::Tensor> pt_inputs{self};
  auto& device = synapse_helpers::HPURegistrar::get_device(device_id);
  size_t key = Op.GetRecipeKey(node_type, stack);

  if (device.get_recipe_handle_cache().isCached(key)) {
    PT_KERNEL_DEBUG("Cache hit key:", key);
    auto output = at::empty(
        self.sizes(),
        self.options().dtype(c10::ScalarType::Bool),
        self.suggest_memory_format());
    Op.SetPTInputs(pt_inputs);
    Op.SetPTOutput(output);
    Op.Execute(key);
  } else {
    PT_KERNEL_DEBUG("Key:", key);
    auto graph = habana_helpers::create_graph(device_id, node_type);
    Op.AllocateSynapseInputs(graph, pt_inputs, true);
    Op.AllocateAndAddSynapseNode(graph, stack, true);
    Op.Compile(graph);
  }
  std::vector<at::Tensor> out = Op.GetOutputs();
  TORCH_CHECK(out.size() == 1, "Incorrect size of outputs");

  PT_KERNEL_END;
  return out.at(0);
}

/*************************************************************************
 * @brief Kernel implementation for output = torch.neg(self)
 * @param [out] output - output tensor, 1-4D, BF16/FP32
 * @param [in] self - input tensor, 1-4D, BF16/FP32
 ************************************************************************/
Tensor neg_hpu(const Tensor& self) {
  PT_KERNEL_BEGIN;

  at::ScalarType scalar_type = self.scalar_type();
  std::string node_type =
      "neg_fwd_" + habana_helpers::name_suffix_from_type(scalar_type);

  // Create the operator
  size_t device_id = self.device().index();
  NegOperator Op(device_id, scalar_type);

  auto out = unary_op_hpu(self, node_type, &Op);

  PT_KERNEL_END;
  return out;
}

/*************************************************************************
 * @brief Kernel implementation for output = torch.sin(self)
 * @param [out] output - output tensor, 1-5D, BF16/FP32
 * @param [in] self - input tensor, 1-5D, BF16/FP32
 ************************************************************************/
Tensor sin_hpu(const Tensor& self) {
  PT_KERNEL_BEGIN;

  bool isSelf_0d = false;
  if (self.dim() == 0) {
    self.unsafeGetTensorImpl()->set_sizes_and_strides({1}, {1});
    isSelf_0d = true;
  }

  at::ScalarType scalar_type = self.scalar_type();
  std::string node_type =
      "sin_fwd_" + habana_helpers::name_suffix_from_type(scalar_type);

  // Create the operator
  size_t device_id = self.device().index();
  SinOperator Op(device_id, scalar_type);

  auto out = unary_op_hpu(self, node_type, &Op);
  if (isSelf_0d) {
    self.unsafeGetTensorImpl()->set_sizes_and_strides({}, {});
    out.unsafeGetTensorImpl()->set_sizes_and_strides({}, {});
  }
  PT_KERNEL_END;
  return out;
}

/*************************************************************************
 * @brief Kernel implementation for output = torch.cos(self)
 * @param [out] output - output tensor, 1-5D, BF16/FP32
 * @param [in] self - input tensor, 1-5D, BF16/FP32
 ************************************************************************/
Tensor cos_hpu(const Tensor& self) {
  PT_KERNEL_BEGIN;

  bool isSelf_0d = false;
  if (self.dim() == 0) {
    self.unsafeGetTensorImpl()->set_sizes_and_strides({1}, {1});
    isSelf_0d = true;
  }
  at::ScalarType scalar_type = self.scalar_type();
  std::string node_type =
      "cos_fwd_" + habana_helpers::name_suffix_from_type(scalar_type);

  // Create the operator
  size_t device_id = self.device().index();
  CosOperator Op(device_id, scalar_type);

  auto out = unary_op_hpu(self, node_type, &Op);

  if (isSelf_0d) {
    self.unsafeGetTensorImpl()->set_sizes_and_strides({}, {});
    out.unsafeGetTensorImpl()->set_sizes_and_strides({}, {});
  }
  PT_KERNEL_END;
  return out;
}

void HardsigmoidOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    Stack& inputs,
    bool is_output_persistent) {
  const unsigned short constExpectedNoOfInput = 1;
  // hardsigmoid (x) = 0          if x <= -3
  //                   1          if x >= +3
  //                   x/6 + 1/2  otherwise
  constexpr float alpha = 1 / 6.0f;
  constexpr float beta = 1 / 2.0f;

  TORCH_CHECK(
      inputs.size() == constExpectedNoOfInput,
      std::string("Expected ") + std::to_string(constExpectedNoOfInput) +
          " input for " +
          (m_inplace ? "HardsigmoidOperator_" : "HardsigmoidOperator") +
          " operator but received " + std::to_string(inputs.size()) +
          " inputs.");
  TORCH_CHECK(inputs[0].isTensor(), "Input type expected to be tensor");

  ns_HardSigmoidKernel::Params param{alpha, beta};
  UnaryLikeOperator::AllocateAndAddSynapseNode(
      graph, inputs, is_output_persistent);
  AddNodeToSynapseGraph(graph, &param, sizeof(param));
}

void HardsigmoidBackwardOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    Stack& inputs,
    bool is_output_persistent) {
  const unsigned short constExpectedNoOfInput = 2;
  constexpr float alpha = 1 / 6.0f;
  constexpr float beta = 1 / 2.0f;
  TORCH_CHECK(
      inputs.size() == constExpectedNoOfInput,
      std::string("Expected ") + std::to_string(constExpectedNoOfInput) +
          " inputs for HardsigmoidOperator operator" + " but received " +
          std::to_string(inputs.size()) + " inputs.");
  ns_HardSigmoidKernel::Params param{alpha, beta};
  auto output = habana_helpers::createPTTensor(
      inputs[0].toTensor(), is_output_persistent);
  AllocateSynapseOutput(graph, output, is_output_persistent);
  AddNodeToSynapseGraph(graph, &param, sizeof(param));
}

void SiluOutOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    bool is_output_persistent) {
  const int64_t correctInputSize = 2;
  TORCH_CHECK(
      inputs.size() == correctInputSize,
      "Incorrect size ",
      inputs.size(),
      " provided as input, while expected size is ",
      correctInputSize,
      " for SiluOutOperator");
  TORCH_CHECK(
      inputs[0].isTensor(),
      "Input arg1 expected to be tensor for SiluOutOperator");

  auto self = inputs[0].toTensor();
  auto out = inputs[1].toTensor();

  at::ScalarType scalar_type = self.scalar_type();
  size_t device_id = self.device().index();

  SigmoidOperator Op(device_id, scalar_type);

  Op.SetSynapseInput(p_context_->syn_inputs_[0]);

  std::vector<c10::IValue> stack{IValue(self)};
  Op.AllocateAndAddSynapseNode(graph, stack, false);

  auto output_sigmoid = Op.GetOutputs()[0];

  stack.clear();

  // Create MulOut operator
  MulOutOperator mulOutOp(this->p_context_->device_id_, scalar_type);
  mulOutOp.SetSynapseInput(p_context_->syn_inputs_[1]);
  mulOutOp.SetSynapseInput(p_context_->syn_inputs_[0]);
  mulOutOp.SetSynapseInput(Op.GetSynOutputs()[0]);

  stack.emplace_back(IValue(out));
  stack.emplace_back(IValue(self));
  stack.emplace_back(IValue(Op.GetOutputs()[0]));

  mulOutOp.AllocateAndAddSynapseNode(graph, stack, is_output_persistent);
  stack.clear();

  p_context_->syn_outputs_.emplace_back(std::move(mulOutOp.GetSynOutputs()[0]));
  p_context_->pt_outputs_.emplace_back(std::move(mulOutOp.GetOutputs()[0]));
}

// TODO:use SiluOutOperator to implement it - avoid duplication of code
void SiluOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    bool is_output_persistent) {
  const int64_t correctInputSize = 1;
  TORCH_CHECK(
      inputs.size() == correctInputSize,
      "Incorrect size " + std::to_string(inputs.size()) +
          " provided as input, while expected size is " +
          std::to_string(correctInputSize) + " for SiluOperator");
  TORCH_CHECK(
      inputs[0].isTensor(),
      "Input arg1 expected to be tensor for SiluOperator");

  auto self = inputs[0].toTensor();
  at::ScalarType scalar_type = self.scalar_type();
  size_t device_id = self.device().index();

  SigmoidOperator Op(device_id, scalar_type);

  Op.SetSynapseInput(p_context_->syn_inputs_[0]);

  std::vector<c10::IValue> stack{IValue(self)};
  Op.AllocateAndAddSynapseNode(graph, stack, false);

  stack.clear();

  // Create Mul operator
  MulOperator mulOp(this->p_context_->device_id_, scalar_type);
  mulOp.SetSynapseInput(p_context_->syn_inputs_[0]);
  mulOp.SetSynapseInput(Op.GetSynOutputs()[0]);

  stack.emplace_back(IValue(self));
  stack.emplace_back(IValue(Op.GetOutputs()[0]));

  mulOp.AllocateAndAddSynapseNode(graph, stack, is_output_persistent);
  stack.clear();

  p_context_->syn_outputs_.emplace_back(std::move(mulOp.GetSynOutputs()[0]));
  p_context_->pt_outputs_.emplace_back(std::move(mulOp.GetOutputs()[0]));
}

/*************************************************************************
 * @brief Kernel implementation for output = torch.silu(input)
 * @param [in] input - input tensor, 1-5D, BF16/FP32
 ************************************************************************/
Tensor silu_hpu(const Tensor& self) {
  PT_KERNEL_BEGIN;
  bool isSelf_0d = false;
  if (self.dim() == 0) {
    self.unsafeGetTensorImpl()->set_sizes_and_strides({1}, {1});
    isSelf_0d = true;
  }
  at::ScalarType scalar_type = self.scalar_type();
  std::string node_type =
      "silu_fwd_" + habana_helpers::name_suffix_from_type(scalar_type);

  // Create the operator
  size_t device_id = self.device().index();
  SiluOperator Op(device_id, scalar_type);

  auto out = unary_op_hpu(self, node_type, &Op);
  if (isSelf_0d) {
    self.unsafeGetTensorImpl()->set_sizes_and_strides({}, {});
    out.unsafeGetTensorImpl()->set_sizes_and_strides({}, {});
  }
  PT_KERNEL_END;
  return out;
}

void SiluBackwardOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    Stack& inputs,
    bool is_output_persistent) {
  const unsigned short constExpectedNoOfInput = 2;

  TORCH_CHECK(
      inputs.size() == constExpectedNoOfInput,
      "Expected ",
      constExpectedNoOfInput,
      " inputs for SiluBackwardOperator but received ",
      inputs.size(),
      " inputs.");
  TORCH_CHECK(
      inputs[0].isTensor(),
      "Input arg1 expected to be Tensor for SiluBackwardOperator");
  TORCH_CHECK(
      inputs[1].isTensor(),
      "Input arg2 expected to be Tensor for SiluBackwardOperator");

  auto grad = inputs[0].toTensor();
  auto self = inputs[1].toTensor();

  auto scalar_type = self.scalar_type();

  // Sigmoid
  SigmoidOperator sigmoidOp(this->p_context_->device_id_, scalar_type);
  sigmoidOp.SetSynapseInput(p_context_->syn_inputs_[1]);
  torch::jit::Stack stack = {IValue(self)};
  sigmoidOp.AllocateAndAddSynapseNode(graph, stack, false);
  stack.clear();

  // Do G*S
  habana::MulOperator mulOp(this->p_context_->device_id_, scalar_type);

  mulOp.SetSynapseInput(p_context_->syn_inputs_[0]); // grad
  mulOp.SetSynapseInput(sigmoidOp.GetSynOutputs()[0]); // sigmoid

  stack.emplace_back(IValue(grad));
  stack.emplace_back(IValue(sigmoidOp.GetOutputs()[0]));
  mulOp.AllocateAndAddSynapseNode(graph, stack, false);

  stack.clear();

  // Do G*S*Self
  habana::MulOperator mulOp1(this->p_context_->device_id_, scalar_type);

  mulOp1.SetSynapseInput(p_context_->syn_inputs_[1]); // self
  mulOp1.SetSynapseInput(mulOp.GetSynOutputs()[0]); // sigmoid

  stack.emplace_back(IValue(self));
  stack.emplace_back(IValue(mulOp.GetOutputs()[0]));
  mulOp1.AllocateAndAddSynapseNode(graph, stack, false);

  stack.clear();

  // Do G*S*Self*S
  habana::MulOperator mulOp2(this->p_context_->device_id_, scalar_type);

  mulOp2.SetSynapseInput(sigmoidOp.GetSynOutputs()[0]); // sigmoid
  mulOp2.SetSynapseInput(mulOp1.GetSynOutputs()[0]); // G*S*Self

  stack.emplace_back(IValue(sigmoidOp.GetOutputs()[0]));
  stack.emplace_back(IValue(mulOp1.GetOutputs()[0]));
  mulOp2.AllocateAndAddSynapseNode(graph, stack, false);

  stack.clear();

  // Do G*S + G*self*S
  habana::AddOperator addOp(this->p_context_->device_id_, scalar_type);

  addOp.SetSynapseInput(mulOp.GetSynOutputs()[0]); // G*S
  addOp.SetSynapseInput(mulOp1.GetSynOutputs()[0]); // G*self*S

  stack.emplace_back(IValue(mulOp.GetOutputs()[0]));
  stack.emplace_back(IValue(mulOp1.GetOutputs()[0]));
  stack.emplace_back(IValue(1.0));
  addOp.AllocateAndAddSynapseNode(graph, stack, false);

  stack.clear();

  // Do G*S + G*self*S - G*self*S*S
  habana::SubOperator subOp(this->p_context_->device_id_, scalar_type);

  subOp.SetSynapseInput(addOp.GetSynOutputs()[0]); // G*S + G*self*S
  subOp.SetSynapseInput(mulOp2.GetSynOutputs()[0]); // G*self*S*S

  stack.emplace_back(IValue(addOp.GetOutputs()[0]));
  stack.emplace_back(IValue(mulOp2.GetOutputs()[0]));
  stack.emplace_back(IValue(1.0));
  subOp.AllocateAndAddSynapseNode(graph, stack, is_output_persistent);

  stack.clear();

  // Output
  p_context_->syn_outputs_.emplace_back(std::move(subOp.GetSynOutputs()[0]));
  p_context_->pt_outputs_.emplace_back(subOp.GetOutputs()[0]);
}

void IsnanOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    bool is_output_persistent) {
  const int64_t correctInputSize = 1;
  TORCH_CHECK(
      inputs.size() == correctInputSize,
      "Incorrect size " + std::to_string(inputs.size()) +
          " provided as input, while expected size is " +
          std::to_string(correctInputSize) + " for IsnanOperator");
  TORCH_CHECK(
      inputs[0].isTensor(),
      "Input arg1 expected to be Tensor for IsnanOperator");

  auto self = inputs[0].toTensor();

  Tensor output = habana_helpers::createPTTensor(
      self,
      self.sizes(),
      self.options(),
      self.suggest_memory_format(),
      c10::ScalarType::Bool,
      is_output_persistent);
  AllocateSynapseOutput(graph, output, is_output_persistent);
  AddNodeToSynapseGraph(graph, nullptr, 0);
}

/*************************************************************************
 * @brief Kernel implementation for output = torch.isnan(self)
 * @param [out] output - output tensor, 1-4D, Bool
 * @param [in] self - input tensor, 1-4D, BF16/FP32
 ************************************************************************/
Tensor isnan_hpu(const Tensor& self) {
  PT_KERNEL_BEGIN;
  bool isSelf_0d = false;
  if (self.dim() == 0) {
    self.unsafeGetTensorImpl()->set_sizes_and_strides({1}, {1});
    isSelf_0d = true;
  }
  at::ScalarType scalar_type = self.scalar_type();
  std::string node_type =
      "isnan_fwd_" + habana_helpers::name_suffix_from_type(scalar_type);
  // Create the operator
  size_t device_id = self.device().index();
  IsnanOperator Op(device_id, scalar_type);

  std::vector<c10::IValue> stack = {IValue(self)};
  std::vector<at::Tensor> pt_inputs{self};
  auto& device = synapse_helpers::HPURegistrar::get_device(device_id);
  size_t key = Op.GetRecipeKey(node_type, stack);

  if (device.get_recipe_handle_cache().isCached(key)) {
    PT_KERNEL_DEBUG("Cache hit key:", key);
    auto output = at::empty(
        self.sizes(),
        self.options().dtype(c10::ScalarType::Bool),
        self.suggest_memory_format());
    Op.SetPTInputs(pt_inputs);
    Op.SetPTOutput(output);
    Op.Execute(key);
  } else {
    PT_KERNEL_DEBUG("Key:", key);
    auto graph = habana_helpers::create_graph(device_id, node_type);
    Op.AllocateSynapseInputs(graph, pt_inputs, true);
    Op.AllocateAndAddSynapseNode(graph, stack, true);
    Op.Compile(graph);
  }
  std::vector<at::Tensor> out = Op.GetOutputs();
  TORCH_CHECK(out.size() == 1, "Incorrect size of outputs");
  auto output = out[0];
  if (isSelf_0d) {
    self.unsafeGetTensorImpl()->set_sizes_and_strides({}, {});
    output.unsafeGetTensorImpl()->set_sizes_and_strides({}, {});
  }
  PT_KERNEL_END;
  return output;
}

void CumsumOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    bool is_output_persistent) {
  const int64_t correctInputSize = 3;
  TORCH_CHECK(
      inputs.size() == correctInputSize,
      "Incorrect size ",
      inputs.size(),
      " provided as input, while expected size is ",
      correctInputSize,
      " for IsnanOperator");
  TORCH_CHECK(
      inputs[0].isTensor(),
      "Input arg1 expected to be Tensor for CumsumOperator");
  TORCH_CHECK(
      inputs[1].isInt(), "Input arg2 expected to be Int for CumsumOperator");

  auto self = inputs[0].toTensor();
  auto dim = inputs[1].toInt();
  // TODO::Handle 3rd arument dtype, in lazy mode
  dim = at::maybe_wrap_dim(dim, self.dim(), /*wrap_scalar=*/true);

  int tpcAxis = (self.sizes().vec().size() - 1) - dim;

  ns_CumSumKernel::Params param{
      tpcAxis, // cumsum along this
      0, // 0 =>inclusive
      0 // 0=> no reverse
  };

  UnaryLikeOperator::AllocateAndAddSynapseNode(
      graph, inputs, is_output_persistent);
  AddNodeToSynapseGraph(graph, &param, sizeof(param));
}

/*************************************************************************
 * @brief Kernel implementation for output = torch.cumsum(input, axis, dtype)
 * @param [in] input - input tensor, 1-4D, BF16/FP32
 * @param [in] dim -   reduction axis , int64_t
 * @param [in] dtype - optional target datatype, ScalarType
 ************************************************************************/
Tensor cumsum_hpu(
    const Tensor& self,
    int64_t dim,
    c10::optional<ScalarType> dtype) {
  PT_KERNEL_BEGIN;
  CONVERT_0D_TO_1D(self)
  at::ScalarType scalar_type = self.scalar_type();
  at::Tensor self_updated_dtype = self;
  if (dtype.has_value() && (dtype.value() != self.scalar_type())) {
    self_updated_dtype = self.to(dtype.value());
  }
  scalar_type = self_updated_dtype.scalar_type();

  std::string node_type =
      "cumsum_fwd_" + habana_helpers::name_suffix_from_type(scalar_type);

  // Create the operator
  size_t device_id = self_updated_dtype.device().index();
  CumsumOperator Op(device_id, scalar_type);

  // Assign Inputs to the Operator
  std::vector<c10::IValue> stack = {
      IValue(self_updated_dtype), IValue(dim), IValue(dtype)};
  std::vector<at::Tensor> pt_inputs{self_updated_dtype};

  auto& device = synapse_helpers::HPURegistrar::get_device(device_id);
  size_t key = Op.GetRecipeKey(node_type, stack);

  if (device.get_recipe_handle_cache().isCached(key)) {
    PT_KERNEL_DEBUG("Cache hit key:", key);
    auto output = at::empty(
        self_updated_dtype.sizes(),
        self_updated_dtype.options(),
        self_updated_dtype.suggest_memory_format());
    Op.SetPTInputs(pt_inputs);
    Op.SetPTOutput(output);
    Op.Execute(key);
  } else {
    PT_KERNEL_DEBUG("key:", key);
    // Create Graph
    auto graph = habana_helpers::create_graph(device_id, node_type);
    Op.AllocateSynapseInputs(graph, pt_inputs, true);
    Op.AllocateAndAddSynapseNode(graph, stack, true);
    Op.Compile(graph);
  }
  std::vector<at::Tensor> out = Op.GetOutputs();
  TORCH_CHECK(out.size() == 1, "Incorrect size of outputs");
  CONVERT_1D_TO_0D(self, out.at(0))
  PT_KERNEL_END;
  return out.at(0);
}

static auto& KernelRegistry =
    habana::KernelRegistry()
        .add("aten::elu", KERNEL_FN(EluOperator))
        .add("aten::elu_", KERNEL_FN(EluOperator))
        .add("aten::leaky_relu", KERNEL_FN(LeakyReluOperator))
        .add("aten::leaky_relu_", KERNEL_FN_ARG(LeakyReluOperator, true))
        .add("aten::leaky_relu_backward", KERNEL_FN(LeakyReluBackwardOperator))
        .add("aten::hardsigmoid", KERNEL_FN(HardsigmoidOperator))
        .add("aten::hardsigmoid_", KERNEL_FN_ARG(HardsigmoidOperator, true))
        .add(
            "aten::hardsigmoid_backward",
            KERNEL_FN(HardsigmoidBackwardOperator))
        .add("aten::abs", KERNEL_FN(AbsOperator))
        .add("aten::abs_", KERNEL_FN(AbsInplaceOperator))
        .add("aten::isfinite", KERNEL_FN(IsfiniteOperator))
        .add("aten::neg", KERNEL_FN(NegOperator))
        .add("aten::reciprocal", KERNEL_FN(ReciprocalOperator))
        .add("aten::reciprocal_", KERNEL_FN(ReciprocalInplaceOperator))
        .add("aten::gelu", KERNEL_FN(GeluOperator))
        .add("aten::gelu_backward", KERNEL_FN(GeluBackwardOperator))
        .add("aten::hbgelu2", KERNEL_FN(HbGeluOperator))
        .add("aten::hbgelu2_backward", KERNEL_FN(GeluBackwardOperator))
        .add("aten::exp", KERNEL_FN(ExpOperator))
        .add("aten::exp_", KERNEL_FN(ExpInplaceOperator))
        .add("aten::isnan", KERNEL_FN(IsnanOperator))
        .add("aten::silu_backward", KERNEL_FN(SiluBackwardOperator))
        .add("aten::cumsum", KERNEL_FN(CumsumOperator));
