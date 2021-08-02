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
#include <torch/script.h>
#include <memory>
#include <vector>

#include "habana_device/HPUCheck.h"
#include "habana_device/hpu_cached_devices.h"
#include "habana_helpers/graph.h"
#include "habana_helpers/tensor_utils.h"
#include "habana_kernels/binary_kernels.h"
#include "habana_kernels/kernel_utils.h"
#include "habana_kernels/resize.h"
#include "habana_kernels/simple_generic_kernel.h"
#include "habana_kernels/tensor_shape_kernels.h"

using namespace torch;

/** @brief if the tensor is in CPU push it to HPU. Further if the CPU tensor is
 *of double dtype typecast to float.
 **/
inline Tensor get_hpu_tensor(Tensor input) {
  Tensor output;
  if (input.device().type() == c10::DeviceType::CPU) {
    if (input.scalar_type() == c10::ScalarType::Double) {
      output = input.to(c10::ScalarType::Float).to(c10::DeviceType::HPU);
    } else {
      output = input.to(c10::DeviceType::HPU);
    }
  } else {
    output = input;
  }

  return output;
}

std::vector<int64_t> habana::BinaryOperator::compute_output_shape(
    const Tensor& arg1,
    const Tensor& arg2) {
  auto out_size = habana_helpers::compute_broadcast_shape(arg1, arg2);
  return out_size;
}

/************************************************************************
 * @brief This function implements synapse node addition for
 * binary operators where both inputs are tensors. Mismatch in input
 * tensor dims is also taken care of using reshape nodes.
 ************************************************************************/
void habana::BinaryOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    bool is_output_persistent) {
  // this check is for stack during graph execution
  TORCH_CHECK(
      inputs.size() == 2,
      "Incorrect size of input expected for Binary operator");
  TORCH_CHECK(inputs[0].isTensor(), "Input type expected to be tensor");
  TORCH_CHECK(inputs[1].isTensor(), "Input type expected to be tensor");
  Tensor arg1 = inputs[0].toTensor();
  Tensor arg2 = inputs[1].toTensor();

  // This if block is introduced to support bool tensors for mult
  // TPC does not support bool for mult operation
  // cast node is added before and after the tpc call to support bool
  // This change is done as an WA to avoid any script change
  // as a part of <https://jira.habana-labs.com/browse/SW-48605>
  if (guid_.substr(0, 4) == "mult" &&
      arg1.scalar_type() == c10::ScalarType::Bool &&
      arg2.scalar_type() == c10::ScalarType::Bool) {
    // Cast Input tensor to Int tensor
    std::string node_type = "cast_i8_to_i32";

    // Create the operator
    auto boolToIntOp1 =
        make_operator<CastOperator>(this->p_context_->device_id_, node_type);
    boolToIntOp1->SetSynapseInput(p_context_->syn_inputs_[0]);

    // Build Params for the graph
    std::vector<c10::IValue> stack{IValue(arg1), IValue(c10::ScalarType::Int)};
    boolToIntOp1->AllocateAndAddSynapseNode(graph, stack, false);
    stack.clear();

    // Create the operator
    auto boolToIntOp2 =
        make_operator<CastOperator>(this->p_context_->device_id_, node_type);
    boolToIntOp2->SetSynapseInput(p_context_->syn_inputs_[1]);

    // Build Params for the graph
    stack.emplace_back(IValue(arg2));
    stack.emplace_back(IValue(c10::ScalarType::Int));
    boolToIntOp2->AllocateAndAddSynapseNode(graph, stack, false);
    stack.clear();

    // Add the Mult node
    auto output_mult = habana_helpers::createPTTensor(
        arg1,
        arg1.sizes(),
        arg1.options(),
        arg1.suggest_memory_format(),
        c10::ScalarType::Int,
        false);

    AllocateSynapseOutput(graph, output_mult, false);
    synapse_helpers::tensor& synOutput = p_context_->syn_outputs_[0];
    synapse_helpers::tensor& synInput1 = boolToIntOp1->GetSynOutputs()[0];
    synapse_helpers::tensor& synInput2 = boolToIntOp2->GetSynOutputs()[0];

    std::vector<synTensor> syn_in{synInput1.get(), synInput2.get()};
    std::vector<synTensor> syn_out{synOutput.get()};

    guid_ = "mult_fwd_" +
        habana_helpers::name_suffix_from_type(c10::ScalarType::Int);
    graph.add_node(
        std::move(syn_in), std::move(syn_out), nullptr, 0, std::move(guid_));

    // Cast Int tensor to Bool tensor
    node_type = "cast_i32_to_i8";

    // Create the operator
    auto intToBoolOp =
        make_operator<CastOperator>(this->p_context_->device_id_, node_type);
    intToBoolOp->SetSynapseInput(std::move(p_context_->syn_outputs_[0]));

    // Build Params for the graph
    stack.emplace_back(IValue(output_mult));
    stack.emplace_back(IValue(c10::ScalarType::Bool));
    intToBoolOp->AllocateAndAddSynapseNode(graph, stack, is_output_persistent);
    p_context_->syn_outputs_.pop_back();
    p_context_->pt_outputs_.pop_back();

    p_context_->syn_outputs_.emplace_back(
        std::move(intToBoolOp->GetSynOutputs()[0]));
    p_context_->pt_outputs_.emplace_back(intToBoolOp->GetOutputs()[0]);
    return;
  }

  synapse_helpers::tensor& arg1_syn_tensor = p_context_->syn_inputs_[0];
  synapse_helpers::tensor& arg2_syn_tensor = p_context_->syn_inputs_[1];

  std::vector<synTensor> syn_inputs;
  syn_inputs.push_back(arg1_syn_tensor.get());

  auto memory_format = at::MemoryFormat::Contiguous;
  if ((arg1.suggest_memory_format() == at::MemoryFormat::ChannelsLast) ||
      (arg2.suggest_memory_format() == at::MemoryFormat::ChannelsLast)) {
    memory_format = at::MemoryFormat::ChannelsLast;
  }

  // For cases where 2nd argument to a binary op is a scalar (e.g. a = b + 5),
  // but gets converted to a tensor (by dispatcher or bridge) before reaching
  // kernel we need to cast 2nd argument to same type as 1st argument.
  std::string node_type = "cast_f32_to_bf16";
  auto castOp =
      make_operator<CastOperator>(this->p_context_->device_id_, node_type);
  if (arg1.dtype() == c10::ScalarType::BFloat16 &&
      arg1.dtype() != arg2.dtype()) {
    castOp->SetSynapseInput(arg2_syn_tensor);
    torch::jit::Stack stack = {IValue(arg2), IValue(c10::ScalarType::BFloat16)};
    castOp->AllocateAndAddSynapseNode(graph, stack, false);
    synapse_helpers::tensor& syn_tensor = std::move(castOp->GetSynOutputs()[0]);
    syn_inputs.push_back(syn_tensor.get());
    auto out_shape = BinaryOperator::compute_output_shape(arg1, arg2);
    auto output = habana_helpers::createPTTensor(
        arg1,
        IntArrayRef(out_shape.data(), out_shape.size()),
        arg1.options(),
        memory_format,
        c10::ScalarType::BFloat16,
        is_output_persistent);
    AllocateSynapseOutput(graph, output, is_output_persistent);
  } else {
    syn_inputs.push_back(arg2_syn_tensor.get());
    auto out_shape = BinaryOperator::compute_output_shape(arg1, arg2);
    auto output = habana_helpers::createPTTensor(
        arg1,
        IntArrayRef(out_shape.data(), out_shape.size()),
        arg1.options(),
        memory_format,
        is_output_persistent);
    AllocateSynapseOutput(graph, output, is_output_persistent);
  }

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
 * @brief This function implements synapse node addition for Binary OPs
 * with 2 input arguments (where 1st input is always a tensor whereas
 * 2nd input can be a tensor or a scalar)
 ************************************************************************/
void habana::BinaryWrapperOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    bool is_output_persistent) {
  // this check is for stack during graph execution
  TORCH_CHECK(
      inputs.size() == 2,
      "Incorrect size of input expected for Binary operator");
  TORCH_CHECK(
      inputs[0].isTensor() || inputs[1].isTensor(),
      "At least one of the inputs arg1 or arg2 expected to be a tensor");
  // Note that pow has a (Scalar, Tensor) variant in native_functions.yaml
  // although mul and div do not
  TORCH_CHECK(
      inputs[0].isTensor() || inputs[0].isScalar(),
      "Input arg1 type expected to be a tensor or scalar");
  TORCH_CHECK(
      inputs[1].isTensor() || inputs[1].isScalar(),
      "Input arg2 type expected to be a tensor or scalar");

  auto binaryOp = make_operator<BinaryOperator>(
      this->p_context_->device_id_, guid_, this->scalarType_);

  if (inputs[0].isTensor() && inputs[1].isTensor()) { // Both inputs are tensors
    binaryOp->SetSynapseInput(p_context_->syn_inputs_[0]);
    binaryOp->SetSynapseInput(p_context_->syn_inputs_[1]);
    binaryOp->AllocateAndAddSynapseNode(graph, inputs, is_output_persistent);

  } else if (inputs[0].isTensor() && inputs[1].isScalar()) { // 2nd input is a
                                                             // scalar
    auto arg1 = inputs[0].toTensor();
    // add constant node to convert 2nd input to tensor
    auto constOp = make_operator<ConstantOperator>(
        this->p_context_->device_id_, this->scalarType_);
    auto const_shape_tensor = habana_helpers::createPTTensor(
        arg1, {1}, arg1.options(), at::MemoryFormat::Contiguous, false);
    torch::jit::Stack constOp_stack = {IValue(const_shape_tensor), inputs[1]};
    constOp->AllocateAndAddSynapseNode(graph, constOp_stack, false);
    binaryOp->SetSynapseInput(p_context_->syn_inputs_[0]);
    binaryOp->SetSynapseInput(constOp->GetSynOutputs()[0]);
    // replace input scalar with input tensor in the stack
    inputs.pop_back();
    inputs.emplace_back(constOp->GetOutputs()[0]);
    binaryOp->AllocateAndAddSynapseNode(graph, inputs, is_output_persistent);
  } else { // 1st input is a scalar
    auto arg2 = inputs[1].toTensor();
    // add constant node to convert 1st input to tensor
    auto constOp = make_operator<ConstantOperator>(
        this->p_context_->device_id_, this->scalarType_);
    auto const_shape_tensor = habana_helpers::createPTTensor(
        arg2, {1}, arg2.options(), at::MemoryFormat::Contiguous, false);
    torch::jit::Stack constOp_stack = {IValue(const_shape_tensor), inputs[0]};
    constOp->AllocateAndAddSynapseNode(graph, constOp_stack, false);
    binaryOp->SetSynapseInput(constOp->GetSynOutputs()[0]);
    binaryOp->SetSynapseInput(p_context_->syn_inputs_[0]);
    // replace input scalar with input tensor in the stack
    inputs.erase(inputs.cbegin());
    inputs.emplace(inputs.cbegin(), constOp->GetOutputs()[0]);
    binaryOp->AllocateAndAddSynapseNode(graph, inputs, is_output_persistent);
  }

  p_context_->pt_outputs_.emplace_back(binaryOp->GetOutputs()[0]);
  p_context_->syn_outputs_.emplace_back(
      std::move(binaryOp->GetSynOutputs()[0]));
}

void habana::BinaryWrapperOperator::SetPTOutputs(torch::jit::Stack& inputs) {
  Tensor output;
  if (inputs[0].isTensor() && inputs[1].isTensor()) {
    auto out_shape = BinaryOperator::compute_output_shape(
        inputs[0].toTensor(), inputs[1].toTensor());
    output = at::empty(
        IntArrayRef(out_shape.data(), out_shape.size()),
        inputs[0].toTensor().options(),
        inputs[0].toTensor().suggest_memory_format());
  } else if (inputs[0].isTensor()) {
    auto operand = inputs[0].toTensor();
    output = at::empty(
        operand.sizes(), operand.options(), operand.suggest_memory_format());
  } else {
    auto operand = inputs[1].toTensor();
    output = at::empty(
        operand.sizes(), operand.options(), operand.suggest_memory_format());
  }
  std::vector<at::Tensor> v{output};
  HabanaOperator::SetPTOutputs(v);
}

/************************************************************************
 * @brief This function implements synapse node addition for
 * binary operators where 2 inputs are tensors & 3rd input is a scalar.
 * Mismatch in input tensor dims is taken care of using reshape nodes,
 * whereas scalar to tensor conversion is done using constant node.
 ************************************************************************/
void habana::BinaryOperatorWithAlpha::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    bool is_output_persistent) {
  TORCH_CHECK(
      inputs.size() == 3, "Incorrect size of input expected for add operator");
  TORCH_CHECK(inputs[0].isTensor(), "Input 0 type expected to be tensor");
  TORCH_CHECK(inputs[1].isTensor(), "Input 1 type expected to be tensor");
  Tensor arg1 = inputs[0].toTensor();
  Tensor arg2 = inputs[1].toTensor();

  auto memory_format = at::MemoryFormat::Contiguous;
  if ((arg1.suggest_memory_format() == at::MemoryFormat::ChannelsLast) ||
      (arg2.suggest_memory_format() == at::MemoryFormat::ChannelsLast)) {
    memory_format = at::MemoryFormat::ChannelsLast;
  }

  if (inputs[2].toScalar().toFloat() != 1.0) {
    // Multiplication between arg2 and alpha is required
    auto mulOp = make_operator<habana::MulOperator>(
        this->p_context_->device_id_, this->scalarType_);
    mulOp->SetSynapseInput(p_context_->syn_inputs_[1]);
    torch::jit::Stack mulOp_stack{inputs[1], inputs[2]};

    mulOp->AllocateAndAddSynapseNode(graph, mulOp_stack, false);
    synapse_helpers::tensor_or_ref& mulOp_out = mulOp->GetSynOutputs()[0];

    auto out_shape = BinaryOperator::compute_output_shape(arg1, arg2);
    auto output = habana_helpers::createPTTensor(
        arg1,
        IntArrayRef(out_shape.data(), out_shape.size()),
        arg1.options(),
        memory_format,
        is_output_persistent);

    AllocateSynapseOutput(graph, output, is_output_persistent);

    synapse_helpers::tensor& arg1_syn_tensor = p_context_->syn_inputs_[0];
    synapse_helpers::tensor& arg2_syn_tensor = mulOp_out;

    std::vector<synTensor> syn_inputs{
        arg1_syn_tensor.get(), arg2_syn_tensor.get()};

    synapse_helpers::tensor& output_syn_tensor = p_context_->syn_outputs_[0];
    std::vector<synTensor> syn_outputs{output_syn_tensor.get()};

    graph.add_node(
        std::move(syn_inputs),
        std::move(syn_outputs),
        nullptr,
        0,
        std::move(guid_));
  } else {
    auto out_shape = BinaryOperator::compute_output_shape(arg1, arg2);
    auto output = habana_helpers::createPTTensor(
        arg1,
        IntArrayRef(out_shape.data(), out_shape.size()),
        arg1.options(),
        memory_format,
        is_output_persistent);
    AllocateSynapseOutput(graph, output, is_output_persistent);

    synapse_helpers::tensor& arg1_syn_tensor = p_context_->syn_inputs_[0];
    synapse_helpers::tensor& arg2_syn_tensor = p_context_->syn_inputs_[1];

    std::vector<synTensor> syn_inputs{
        arg1_syn_tensor.get(), arg2_syn_tensor.get()};

    synapse_helpers::tensor& output_syn_tensor = p_context_->syn_outputs_[0];
    std::vector<synTensor> syn_outputs{output_syn_tensor.get()};

    graph.add_node(
        std::move(syn_inputs),
        std::move(syn_outputs),
        nullptr,
        0,
        std::move(guid_));
  }
}

/************************************************************************
 * @brief This function implements synapse node addition for Binary OPs
 * with 3 input arguments (where 1st input is always a tensor whereas
 * 2nd and 3rd inputs can be a tensor or a scalar)
 ************************************************************************/
void habana::BinaryWrapperOperatorWithAlpha::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    bool is_output_persistent) {
  TORCH_CHECK(
      inputs.size() == 3, "Incorrect size of input expected for add operator");
  TORCH_CHECK(
      inputs[0].isTensor() || inputs[1].isTensor(),
      "At least one of the inputs arg1 or arg2 expected to be a tensor");
  TORCH_CHECK(
      inputs[0].isTensor() || inputs[0].isScalar(),
      "Input arg1 type expected to be a tensor or scalar");
  TORCH_CHECK(
      inputs[1].isTensor() || inputs[1].isScalar(),
      "Input arg2 type expected to be a tensor or scalar");
  TORCH_CHECK(inputs[2].isScalar(), "Input arg3 type expected to be scalar");

  auto binaryOp = make_operator<BinaryOperatorWithAlpha>(
      this->p_context_->device_id_, guid_, this->scalarType_);

  if (inputs[0].isTensor() &&
      inputs[1].isTensor()) { // First 2 inputs are both tensors
    binaryOp->SetSynapseInput(p_context_->syn_inputs_[0]);
    binaryOp->SetSynapseInput(p_context_->syn_inputs_[1]);
    binaryOp->AllocateAndAddSynapseNode(graph, inputs, is_output_persistent);
  } else if (inputs[0].isTensor() && inputs[1].isScalar()) { // 2nd input is a
                                                             // scalar
    auto arg1 = inputs[0].toTensor();
    // add node to convert scalar to tensor
    auto constOp = make_operator<ConstantOperator>(
        this->p_context_->device_id_, this->scalarType_);
    auto const_shape_tensor = habana_helpers::createPTTensor(
        arg1, {1}, arg1.options(), at::MemoryFormat::Contiguous, false);
    torch::jit::Stack constOp_stack = {IValue(const_shape_tensor), inputs[1]};
    constOp->AllocateAndAddSynapseNode(graph, constOp_stack, false);
    binaryOp->SetSynapseInput(p_context_->syn_inputs_[0]);
    binaryOp->SetSynapseInput(constOp->GetSynOutputs()[0]);
    // replace 2nd scalar input with a tensor in stack
    inputs.erase(inputs.cbegin() + 1);
    inputs.emplace(inputs.cbegin() + 1, constOp->GetOutputs()[0]);
    binaryOp->AllocateAndAddSynapseNode(graph, inputs, is_output_persistent);
  } else { // 1st input is a scalar
    auto arg2 = inputs[1].toTensor();
    // add node to convert scalar to tensor
    auto constOp = make_operator<ConstantOperator>(
        this->p_context_->device_id_, this->scalarType_);
    auto const_shape_tensor = habana_helpers::createPTTensor(
        arg2, {1}, arg2.options(), at::MemoryFormat::Contiguous, false);
    torch::jit::Stack constOp_stack = {IValue(const_shape_tensor), inputs[0]};
    constOp->AllocateAndAddSynapseNode(graph, constOp_stack, false);
    binaryOp->SetSynapseInput(constOp->GetSynOutputs()[0]);
    binaryOp->SetSynapseInput(p_context_->syn_inputs_[0]);

    // replace 1st scalar input with a tensor in stack
    inputs.erase(inputs.cbegin());
    inputs.emplace(inputs.cbegin(), constOp->GetOutputs()[0]);
    binaryOp->AllocateAndAddSynapseNode(graph, inputs, is_output_persistent);
  }

  p_context_->pt_outputs_.emplace_back(binaryOp->GetOutputs()[0]);
  p_context_->syn_outputs_.emplace_back(
      std::move(binaryOp->GetSynOutputs()[0]));
}

void habana::BinaryWrapperOperatorWithAlpha::SetPTOutputs(
    torch::jit::Stack& inputs) {
  Tensor output;
  if (inputs[0].isTensor() && inputs[1].isTensor()) {
    auto out_shape = BinaryOperator::compute_output_shape(
        inputs[0].toTensor(), inputs[1].toTensor());
    output = at::empty(
        IntArrayRef(out_shape.data(), out_shape.size()),
        inputs[0].toTensor().options(),
        inputs[0].toTensor().suggest_memory_format());
  } else if (inputs[0].isTensor()) {
    auto operand = inputs[0].toTensor();
    output = at::empty(
        operand.sizes(), operand.options(), operand.suggest_memory_format());
  } else {
    auto operand = inputs[1].toTensor();
    output = at::empty(
        operand.sizes(), operand.options(), operand.suggest_memory_format());
  }
  std::vector<at::Tensor> v{output};
  HabanaOperator::SetPTOutputs(v);
}

/************************************************************************
 * @brief Generic wrapper for all Eager mode Binary OP invocations that
 * are "not" .out or inplace
 ************************************************************************/
template <class BinaryOp>
Tensor process_generic_tensor_binary_op(
    std::vector<at::Tensor>& pt_inputs,
    torch::jit::Stack& stack,
    const std::string& node_guid) {
  for (auto i = 0u; i < stack.size(); i++) {
    if (stack[i].isTensor()) {
      if (stack[i].toTensor().scalar_type() == c10::ScalarType::Long) {
        auto dst = habana_helpers::cast_tensor_to_integer(stack[i].toTensor());
        // overwrite original tensor with corresponding casted tensor
        pt_inputs[i] = dst;
        stack[i] = IValue(dst);
      }
    }
  }

  // If dtypes of input tensors differ we need to cast one of them to larger
  // dtype.
  int pos = -1;
  c10::ScalarType dst_dtype = c10::ScalarType::Float;
  habana_helpers::type_promotion_for_two_tensor_inputs(stack, pos, dst_dtype);
  if (pos != -1) {
    auto dst = habana_helpers::hpu_cast_tensor(
        stack[pos].toTensor(), at::scalarTypeToTypeMeta(dst_dtype));
    // overwrite original tensor with corresponding casted tensor
    pt_inputs[pos] = dst;
    stack[pos] = IValue(dst);
  }

  size_t device_id = pt_inputs[0].device().index();
  at::ScalarType scalar_type = pt_inputs[0].scalar_type();
  std::string node_type =
      node_guid + "_fwd_" + habana_helpers::name_suffix_from_type(scalar_type);

  auto& device = synapse_helpers::HPURegistrar::get_device(device_id);
  BinaryOp Op(device_id, scalar_type);

  size_t key = Op.GetRecipeKey(node_type, stack);

  if (device.get_recipe_handle_cache().isCached(key)) {
    PT_KERNEL_DEBUG("Cache hit key:", key);
    Op.SetPTInputs(pt_inputs);
    Op.SetPTOutputs(stack);
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
  std::vector<at::Tensor> out = Op.GetOutputs();
  TORCH_CHECK(out.size() == 1, "Incorrect size of outputs");
  return out[0];
}

/*************************************************************************
 * @brief Kernel implementation for out = torch.add(self, alpha, other)
 * @param self - first input
 * @param other - second input
 * @param alpha - optional input
 * out = self + alpha * other
 ************************************************************************/
Tensor add_tensor_hpu(const Tensor& self, const Tensor& other, const Scalar& alpha) {
  PT_KERNEL_BEGIN;
  Tensor output;
  if (self.dim() == 0) {
    self.unsafeGetTensorImpl()->set_sizes_and_strides({1}, {1});
  }
  if (other.dim() == 0) {
    other.unsafeGetTensorImpl()->set_sizes_and_strides({1}, {1});
  }
  auto self_hpu = get_hpu_tensor(self);
  auto other_hpu = get_hpu_tensor(other);
  std::vector<at::Tensor> pt_inputs{self_hpu, other_hpu};
  torch::jit::Stack stack{IValue(self_hpu), IValue(other_hpu), IValue(alpha)};
  output = process_generic_tensor_binary_op<habana::AddOperator>(
      pt_inputs, stack, "add");
  PT_KERNEL_END;
  return output;
}

/*************************************************************************
 * @brief Kernel implementation for out = add.Scalar(self, alpha, other)
 * @param self - first input
 * @param other - second input
 * @param alpha - optional input
 * out = self + alpha * other
 ************************************************************************/
Tensor add_scalar_hpu(const Tensor& self, const Scalar& other, const Scalar& alpha) {
  PT_KERNEL_BEGIN;

  if (self.dim() == 0) {
    self.unsafeGetTensorImpl()->set_sizes_and_strides({1}, {1});
  }
  auto self_hpu = get_hpu_tensor(self);
  std::vector<at::Tensor> pt_inputs{self_hpu};
  torch::jit::Stack stack{IValue(self_hpu), IValue(other), IValue(alpha)};
  auto output = process_generic_tensor_binary_op<habana::AddOperator>(
      pt_inputs, stack, "add");

  PT_KERNEL_END;
  return output;
}

/*************************************************************************
 * @brief Kernel implementation for out = torch.sub(self, alpha, other)
 * @param self - first input
 * @param other - second input
 * @param alpha - optional input
 * out = self - alpha * other
 ************************************************************************/
Tensor sub_tensor_hpu(const Tensor& self, const Tensor& other, const Scalar& alpha) {
  PT_KERNEL_BEGIN;

  if (self.dim() == 0) {
    self.unsafeGetTensorImpl()->set_sizes_and_strides({1}, {1});
  }
  if (other.dim() == 0) {
    other.unsafeGetTensorImpl()->set_sizes_and_strides({1}, {1});
  }
  auto self_hpu = get_hpu_tensor(self);
  auto other_hpu = get_hpu_tensor(other);
  std::vector<at::Tensor> pt_inputs{self_hpu, other_hpu};
  torch::jit::Stack stack{IValue(self_hpu), IValue(other_hpu), IValue(alpha)};
  auto output = process_generic_tensor_binary_op<habana::SubOperator>(
      pt_inputs, stack, "sub");

  PT_KERNEL_END;
  return output;
}

/*************************************************************************
 * @brief Kernel implementation for inplace Scalar torch.sub_(self, alpha,
 *other)
 * @param self - first input
 * @param other - second input
 * @param alpha - optional input
 * self -= alpha * other
 ************************************************************************/
Tensor sub_scalar_hpu(
    const Tensor& self,
    const Scalar& other,
    const Scalar& alpha) { // TODO: No way to test this yet from python
  PT_KERNEL_BEGIN;

  if (self.dim() == 0) {
    self.unsafeGetTensorImpl()->set_sizes_and_strides({1}, {1});
  }
  auto self_hpu = get_hpu_tensor(self);
  std::vector<at::Tensor> pt_inputs{self_hpu};
  torch::jit::Stack stack{IValue(self_hpu), IValue(other), IValue(alpha)};
  auto output = process_generic_tensor_binary_op<habana::SubOperator>(
      pt_inputs, stack, "sub");

  PT_KERNEL_END;
  return output;
}

void habana::RsubOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    bool is_output_persistent) {
  TORCH_CHECK(
      inputs.size() == 3, "Incorrect size of input expected for add operator");
  TORCH_CHECK(inputs[0].isTensor(), "Input arg1 type expected to be tensor");
  TORCH_CHECK(
      inputs[1].isTensor() || inputs[1].isScalar(),
      "Input arg2 type expected to be a tensor or scalar");
  TORCH_CHECK(inputs[2].isScalar(), "Input arg3 type expected to be scalar");

  // Swap the first and second members of inputs
  inputs = {inputs[1], inputs[0], inputs[2]};
  // Now invoke normal SubOperator
  SubOperator::AllocateAndAddSynapseNode(graph, inputs, is_output_persistent);
}

/*************************************************************************
 * @brief Kernel implementation for rsub Scalar torch.rsub(self, other,
 *  alpha)
 * @param self - first input tensor, 1-4D, FP32/BF16
 * @param other - second input Scalar
 * @param alpha - optional input Scalar, default = 1
 * output = other - self * alpha
 ************************************************************************/
Tensor rsub_scalar_hpu(const Tensor& self, const Scalar& other, const Scalar& alpha) {
  PT_KERNEL_BEGIN;

  if (self.dim() == 0) {
    self.unsafeGetTensorImpl()->set_sizes_and_strides({1}, {1});
  }
  auto self_hpu = get_hpu_tensor(self);
  std::vector<at::Tensor> pt_inputs{self_hpu};
  torch::jit::Stack stack{IValue(self_hpu), IValue(other), IValue(alpha)};
  auto output = process_generic_tensor_binary_op<habana::RsubOperator>(
      pt_inputs, stack, "sub");

  PT_KERNEL_END;
  return output;
}

/*************************************************************************
 * @brief Kernel implementation for output = torch.mul(self, other)
 * @param self - first input
 * @param other - second input
 * output = self * other
 ************************************************************************/
Tensor mul_tensor_hpu(const Tensor& self, const Tensor& other) {
  PT_KERNEL_BEGIN;

  // TODO: Add pow operator for graph mode
  if (self.is_same(other) && self.scalar_type() != c10::ScalarType::Bool &&
      other.scalar_type() != c10::ScalarType::Bool) {
    auto tensor = at::pow(self, 2.0);
    PT_KERNEL_END;
    return tensor;
  }

  if (self.dim() == 0) {
    self.unsafeGetTensorImpl()->set_sizes_and_strides({1}, {1});
  }
  if (other.dim() == 0) {
    other.unsafeGetTensorImpl()->set_sizes_and_strides({1}, {1});
  }
  auto self_hpu = get_hpu_tensor(self);
  auto other_hpu = get_hpu_tensor(other);
  std::vector<at::Tensor> pt_inputs{self_hpu, other_hpu};
  torch::jit::Stack stack{IValue(self_hpu), IValue(other_hpu)};
  auto output = process_generic_tensor_binary_op<habana::MulOperator>(
      pt_inputs, stack, "mult");

  PT_KERNEL_END;
  return output;
}

/*************************************************************************
 * @brief Kernel implementation for output = torch.mul(self, Scalar other)
 * @param self - first input
 * @param other - second input
 * output = self * other
 ************************************************************************/
Tensor mul_scalar_hpu(const Tensor& self, const Scalar& other) {
  PT_KERNEL_BEGIN;

  if (self.dim() == 0) {
    self.unsafeGetTensorImpl()->set_sizes_and_strides({1}, {1});
  }
  auto self_hpu = get_hpu_tensor(self);
  std::vector<at::Tensor> pt_inputs{self_hpu};
  torch::jit::Stack stack{IValue(self_hpu), IValue(other)};
  auto output = process_generic_tensor_binary_op<habana::MulOperator>(
      pt_inputs, stack, "mult");

  PT_KERNEL_END;
  return output;
}

/*************************************************************************
 * @brief Kernel implementation for out = torch.div(self,other)
 * @param self - first input
 * @param other - second input
 ************************************************************************/
Tensor div_tensor_hpu(const Tensor& self, const Tensor& other) {
  PT_KERNEL_BEGIN;

  if (self.dim() == 0) {
    self.unsafeGetTensorImpl()->set_sizes_and_strides({1}, {1});
  }
  if (other.dim() == 0) {
    other.unsafeGetTensorImpl()->set_sizes_and_strides({1}, {1});
  }
  auto self_hpu = get_hpu_tensor(self);
  auto other_hpu = get_hpu_tensor(other);
  std::vector<at::Tensor> pt_inputs{self_hpu, other_hpu};
  torch::jit::Stack stack{IValue(self_hpu), IValue(other_hpu)};
  auto output = process_generic_tensor_binary_op<habana::DivOperator>(
      pt_inputs, stack, "div");

  PT_KERNEL_END;
  return output;
}

/*************************************************************************
 * @brief Kernel implementation for inplace div.Scalar(self,other)
 * @param self - first input
 * @param other - second input of scalar type
 ************************************************************************/
Tensor div_scalar_hpu(
    const Tensor& self,
    const Scalar& other) { // TODO: Add test by using an extension module for new op
                    // at python level
  PT_KERNEL_BEGIN;

  if (self.dim() == 0) {
    self.unsafeGetTensorImpl()->set_sizes_and_strides({1}, {1});
  }
  auto self_hpu = get_hpu_tensor(self);
  std::vector<at::Tensor> pt_inputs{self_hpu};
  torch::jit::Stack stack{IValue(self_hpu), IValue(other)};
  auto output = process_generic_tensor_binary_op<habana::DivOperator>(
      pt_inputs, stack, "div");

  PT_KERNEL_END;
  return output;
}

/*************************************************************************
 * @brief Kernel implementation for out = torch.pow(self,other)
 * @param self - first input
 * @param other - second input
 ************************************************************************/
Tensor pow_tensor_tensor_hpu(const Tensor& self, const Tensor& other) {
  PT_KERNEL_BEGIN;

  if (self.dim() == 0) {
    self.unsafeGetTensorImpl()->set_sizes_and_strides({1}, {1});
  }
  if (other.dim() == 0) {
    other.unsafeGetTensorImpl()->set_sizes_and_strides({1}, {1});
  }
  auto self_hpu = get_hpu_tensor(self);
  auto other_hpu = get_hpu_tensor(other);
  std::vector<Tensor> pt_inputs{self_hpu, other_hpu};
  torch::jit::Stack stack{IValue(self_hpu), IValue(other_hpu)};
  auto output = process_generic_tensor_binary_op<habana::PowOperator>(
      pt_inputs, stack, "pow");

  PT_KERNEL_END;
  return output;
}

/*************************************************************************
 * @brief Kernel implementation for out = self.pow(,other)
 * @param self [in,out]- Tensor 1D bf16/FP32
 * @param other [in] - Scalar
 ************************************************************************/
Tensor pow_tensor_scalar_hpu(const Tensor& self, const Scalar& other) {
  PT_KERNEL_BEGIN;

  if (self.dim() == 0) {
    self.unsafeGetTensorImpl()->set_sizes_and_strides({1}, {1});
  }
  auto self_hpu = get_hpu_tensor(self);
  std::vector<at::Tensor> pt_inputs{self_hpu};
  torch::jit::Stack stack{IValue(self_hpu), IValue(other)};
  auto output = process_generic_tensor_binary_op<habana::PowOperator>(
      pt_inputs, stack, "pow");

  PT_KERNEL_END;
  return output;
}

/***************************************************************************
 * @brief Kernel implementation for out = torch.pow(other,self) = other^self
 * @param other [in] - Scalar
 * @param self [in,out]- Tensor 1D bf16/FP32
 ****************************************************************************/
Tensor pow_scalar_tensor_hpu(Scalar other, const Tensor& self) {
  PT_KERNEL_BEGIN;

  if (self.dim() == 0) {
    self.unsafeGetTensorImpl()->set_sizes_and_strides({1}, {1});
  }
  auto self_hpu = get_hpu_tensor(self);
  std::vector<at::Tensor> pt_inputs{self_hpu};
  torch::jit::Stack stack{IValue(other), IValue(self_hpu)};
  auto output = process_generic_tensor_binary_op<habana::PowOperator>(
      pt_inputs, stack, "pow");

  PT_KERNEL_END;
  return output;
}

/***************************************************************************
 * @brief Kernel implementation for aten::_masked_scale(Tensor self, Tensor
 *mask, float scale) -> Tensor
 * @param self [in]- Tensor 1D bf16/FP32
 * @param mask [in]- Tensor 1D bf16/FP32
 * @param scale - float
 * Implements: grad_input = grad_output * mask / p1m
 ****************************************************************************/
Tensor masked_scale_hpu(const Tensor& self, const Tensor& mask, double scale) {
  PT_OTHER_OPS_BEGIN; // this macro is used because this kernel is used
                      // within Lazy kernel tests
  // scale changed to support dropout backward based on what we pass for dropout
  scale = 1.0 / (1.0 - 1.0 / scale);
  auto tt_mul_out = at::mul(
      self,
      (self.dtype() != mask.dtype())
          ? habana_helpers::hpu_cast_tensor(mask, self.dtype())
          : mask);
  auto output = at::mul(tt_mul_out, Scalar(scale));
  PT_KERNEL_END;
  return output;
}

/*************************************************************************
 * @brief Kernel implementation for output = torch.maximum(self, other)
 * @param self - first input
 * @param other - second input
 * output[i] = self[i] > other[i] ? self[i] : other[i]
 ************************************************************************/
Tensor maximum_hpu(const Tensor& self, const Tensor& other) {
  PT_KERNEL_BEGIN;

  if (self.dim() == 0) {
    self.unsafeGetTensorImpl()->set_sizes_and_strides({1}, {1});
  }
  if (other.dim() == 0) {
    other.unsafeGetTensorImpl()->set_sizes_and_strides({1}, {1});
  }
  auto self_hpu = get_hpu_tensor(self);
  auto other_hpu = get_hpu_tensor(other);
  std::vector<at::Tensor> pt_inputs{self_hpu, other_hpu};
  torch::jit::Stack stack{IValue(self_hpu), IValue(other_hpu)};
  auto output = process_generic_tensor_binary_op<habana::MaximumOperator>(
      pt_inputs, stack, "max");

  PT_KERNEL_END;
  return output;
}

/*************************************************************************
 * @brief Kernel implementation for output = torch.minimum(self, other)
 * @param self - first input
 * @param other - second input
 * output[i] = self[i] < other[i] ? self[i] : other[i]
 ************************************************************************/
Tensor minimum_hpu(const Tensor& self, const Tensor& other) {
  PT_KERNEL_BEGIN;

  if (self.dim() == 0) {
    self.unsafeGetTensorImpl()->set_sizes_and_strides({1}, {1});
  }
  if (other.dim() == 0) {
    other.unsafeGetTensorImpl()->set_sizes_and_strides({1}, {1});
  }
  auto self_hpu = get_hpu_tensor(self);
  auto other_hpu = get_hpu_tensor(other);
  std::vector<at::Tensor> pt_inputs{self_hpu, other_hpu};
  torch::jit::Stack stack{IValue(self_hpu), IValue(other_hpu)};
  auto output = process_generic_tensor_binary_op<habana::MinimumOperator>(
      pt_inputs, stack, "min");

  PT_KERNEL_END;
  return output;
}

void habana::RemainderWrapperOperator::SetPTOutputs(torch::jit::Stack& inputs) {
  Tensor quotient, remainder;
  if (inputs[0].isTensor() && inputs[1].isTensor()) {
    auto self = inputs[0].toTensor();
    auto other = inputs[1].toTensor();
    auto out_shape = BinaryOperator::compute_output_shape(self, other);
    remainder = habana_helpers::createPTTensor(
        self,
        IntArrayRef(out_shape.data(), out_shape.size()),
        self.options(),
        self.suggest_memory_format(),
        true);
  } else {
    auto self = inputs[0].toTensor();
    remainder = habana_helpers::createPTTensor(self, true);
  }
  std::vector<at::Tensor> v{remainder};
  HabanaOperator::SetPTOutputs(v);
}

void habana::RemainderWrapperOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    bool is_output_persistent) {
  TORCH_CHECK(
      inputs.size() == 2,
      "Incorrect size of input expected for Remainder operator");
  // Note that there is no (Scalar, Tensor) version for remainder ops
  // in native_functions.yaml
  TORCH_CHECK(inputs[0].isTensor(), "Input arg1 type expected to be tensor");
  TORCH_CHECK(
      inputs[1].isTensor() || inputs[1].isScalar(),
      "Input arg2 type expected to be a tensor or scalar");

  auto remainderOp = make_operator<RemainderOperator>(
      this->p_context_->device_id_, this->scalarType_);

  if (inputs[1].isTensor()) { // Both inputs are tensors
    remainderOp->SetSynapseInput(p_context_->syn_inputs_[0]);
    remainderOp->SetSynapseInput(p_context_->syn_inputs_[1]);
    remainderOp->AllocateAndAddSynapseNode(graph, inputs, is_output_persistent);
  } else { // 2nd input is a scalar
    // add constant node to convert 2nd input to tensor
    auto arg1 = inputs[0].toTensor();
    auto constOp = make_operator<ConstantOperator>(
        this->p_context_->device_id_, this->scalarType_);
    auto const_shape_tensor = habana_helpers::createPTTensor(
        arg1, {1}, arg1.options(), at::MemoryFormat::Contiguous, false);
    torch::jit::Stack constOp_stack = {IValue(const_shape_tensor), inputs[1]};
    constOp->AllocateAndAddSynapseNode(graph, constOp_stack, false);
    remainderOp->SetSynapseInput(p_context_->syn_inputs_[0]);
    remainderOp->SetSynapseInput(constOp->GetSynOutputs()[0]);
    // replace 2nd scalar input with a tensor in stack
    inputs.erase(inputs.cbegin() + 1);
    inputs.emplace(inputs.cbegin() + 1, constOp->GetOutputs()[0]);
    remainderOp->AllocateAndAddSynapseNode(graph, inputs, is_output_persistent);
  }

  p_context_->pt_outputs_.emplace_back(remainderOp->GetOutputs()[1]);
  p_context_->syn_outputs_.emplace_back(
      std::move(remainderOp->GetSynOutputs()[1]));
}

void habana::RemainderOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    bool is_output_persistent) {
  TORCH_CHECK(
      inputs.size() == 2,
      "Incorrect size of input expected for remainder operator");
  TORCH_CHECK(inputs[0].isTensor(), "Input type expected to be tensor");
  TORCH_CHECK(inputs[1].isTensor(), "Input type expected to be tensor");

  at::Tensor self = inputs[0].toTensor();
  at::Tensor other = inputs[1].toTensor();

  // Python div_mod is enabled where remainder returns the same sign of the
  // divisor, except for the zero remainder
  ns_DivModKernel::Params params{true};

  p_context_->params_.emplace<ns_DivModKernel::Params>(params);
  p_context_->params_size_ = sizeof(params);

  // tpc kernel returns quotient and remainder as tuple
  auto out_shape = BinaryOperator::compute_output_shape(self, other);
  auto quotient = habana_helpers::createPTTensor(
      self,
      IntArrayRef(out_shape.data(), out_shape.size()),
      self.options(),
      self.suggest_memory_format(),
      false);
  auto remainder = habana_helpers::createPTTensor(
      self,
      IntArrayRef(out_shape.data(), out_shape.size()),
      self.options(),
      self.suggest_memory_format(),
      is_output_persistent);
  AllocateSynapseOutputs(
      graph, {quotient, remainder}, {false, is_output_persistent});
  AddNodeToSynapseGraph(graph, &params, sizeof(params));
}

Tensor remainder_tensor_hpu(const Tensor& self, const Tensor& other) {
  PT_KERNEL_BEGIN;

  bool isSelf_0d = false;
  bool isOther_0d = false;
  if (self.dim() == 0) {
    self.unsafeGetTensorImpl()->set_sizes_and_strides({1}, {1});
    isSelf_0d = true;
  }
  if (other.dim() == 0) {
    other.unsafeGetTensorImpl()->set_sizes_and_strides({1}, {1});
    isOther_0d = true;
  }

  size_t device_id = self.device().index();
  auto& device = synapse_helpers::HPURegistrar::get_device(device_id);
  at::ScalarType scalar_type = self.scalar_type();
  std::string node_type =
      "div_mod_fwd_" + habana_helpers::name_suffix_from_type(scalar_type);

  // create the operator
  habana::RemainderWrapperOperator Op(device_id, scalar_type);
  std::vector<c10::IValue> stack = {IValue(self), IValue(other)};
  size_t key = Op.GetRecipeKey(node_type, stack);

  std::vector<at::Tensor> pt_inputs{self, other};
  if (device.get_recipe_handle_cache().isCached(key)) {
    PT_KERNEL_DEBUG("Cache hit key:", key);

    Op.SetPTInputs(pt_inputs);
    Op.SetPTOutputs(stack);
    Op.Execute(key);
  } else {
    PT_KERNEL_DEBUG("Key:", key);
    // create graph
    auto graph = habana_helpers::create_graph(device_id, node_type);

    // Assign Inputs to the Operator
    Op.AllocateSynapseInputs(graph, pt_inputs, true);

    // Build Params for the graph
    Op.AllocateAndAddSynapseNode(graph, stack, true);

    // compile and execute the graph
    Op.Compile(graph);
  }
  std::vector<at::Tensor> out = Op.GetOutputs();
  TORCH_CHECK(out.size() == 1, "Incorrect size of outputs");

  auto output = out.at(0);
  if (isSelf_0d && isOther_0d) {
    output.unsafeGetTensorImpl()->set_sizes_and_strides({}, {});
  }
  if (isSelf_0d) {
    self.unsafeGetTensorImpl()->set_sizes_and_strides({}, {});
  }
  if (isOther_0d) {
    other.unsafeGetTensorImpl()->set_sizes_and_strides({}, {});
  }

  PT_KERNEL_END;
  return output;
}

void habana::RemainderInplaceOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    std::vector<bool> is_output_persistent) {
  // this check is for stack during graph execution
  TORCH_CHECK(
      inputs.size() == 2,
      "Incorrect size of input expected for Binary operator");
  TORCH_CHECK(inputs[0].isTensor(), "Input type expected to be tensor");
  TORCH_CHECK(inputs[1].isTensor(), "Input type expected to be tensor");
  Tensor self = inputs[0].toTensor();
  Tensor other = inputs[1].toTensor();

  auto out_shape = BinaryOperator::compute_output_shape(self, other);
  auto quotient = habana_helpers::createPTTensor(
      self,
      IntArrayRef(out_shape.data(), out_shape.size()),
      self.options(),
      self.suggest_memory_format(),
      is_output_persistent[0]);
  AllocateSynapseOutput(graph, quotient, is_output_persistent[0]);

  // Note here we are using input[0] to store output[1]
  p_context_->syn_outputs_.emplace_back(
      habana_helpers::duplicate_tensor_in_memory_section(
          p_context_->syn_inputs_[0], graph));
  p_context_->pt_outputs_.emplace_back(self);

  synapse_helpers::tensor& arg1_syn_tensor = p_context_->syn_inputs_[0];
  synapse_helpers::tensor& arg2_syn_tensor = p_context_->syn_inputs_[1];

  std::vector<synTensor> syn_inputs{
      arg1_syn_tensor.get(), arg2_syn_tensor.get()};

  synapse_helpers::tensor& output1_syn_tensor = p_context_->syn_outputs_[0];
  synapse_helpers::tensor& output2_syn_tensor = p_context_->syn_outputs_[1];
  std::vector<synTensor> syn_outputs{
      output1_syn_tensor.get(), output2_syn_tensor.get()};

  // Python div_mod is enabled where remainder returns the same sign of the
  // divisor, except for the zero remainder
  ns_DivModKernel::Params params{true};

  p_context_->params_.emplace<ns_DivModKernel::Params>(params);
  p_context_->params_size_ = sizeof(params);
  graph.add_node(
      std::move(syn_inputs),
      std::move(syn_outputs),
      &params,
      sizeof(params),
      std::move(this->guid_));
}

void habana::RemainderInplaceWrapperOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    bool is_output_persistent) {
  TORCH_CHECK(
      inputs.size() == 2,
      "Incorrect size of input expected for Remainder operator");
  // Note that there is no (Scalar, Tensor) version for remainder ops
  // in native_functions.yaml
  TORCH_CHECK(inputs[0].isTensor(), "Input arg1 type expected to be tensor");
  TORCH_CHECK(
      inputs[1].isTensor() || inputs[1].isScalar(),
      "Input arg2 type expected to be a tensor or scalar");

  auto remainderOp = make_operator<RemainderInplaceOperator>(
      this->p_context_->device_id_, this->scalarType_);

  if (inputs[1].isTensor()) { // Both inputs are tensors
    remainderOp->SetSynapseInput(p_context_->syn_inputs_[0]);
    remainderOp->SetSynapseInput(p_context_->syn_inputs_[1]);
    remainderOp->AllocateAndAddSynapseNode(
        graph, inputs, {false, is_output_persistent});
  } else { // 2nd input is a scalar
    // add constant node to convert 2nd input to tensor
    auto arg1 = inputs[0].toTensor();
    auto constOp = make_operator<ConstantOperator>(
        this->p_context_->device_id_, this->scalarType_);
    auto const_shape_tensor = habana_helpers::createPTTensor(
        arg1, {1}, arg1.options(), at::MemoryFormat::Contiguous, false);
    torch::jit::Stack constOp_stack = {IValue(const_shape_tensor), inputs[1]};
    constOp->AllocateAndAddSynapseNode(graph, constOp_stack, false);
    remainderOp->SetSynapseInput(p_context_->syn_inputs_[0]);
    remainderOp->SetSynapseInput(constOp->GetSynOutputs()[0]);
    // replace 2nd scalar input with a tensor in stack
    inputs.erase(inputs.cbegin() + 1);
    inputs.emplace(inputs.cbegin() + 1, constOp->GetOutputs()[0]);
    remainderOp->AllocateAndAddSynapseNode(
        graph, inputs, {false, is_output_persistent});
  }

  p_context_->pt_outputs_.emplace_back(remainderOp->GetOutputs()[1]);
  p_context_->syn_outputs_.emplace_back(
      std::move(remainderOp->GetSynOutputs()[1]));
}
Tensor& remainder_tensor_hpu_(Tensor& self, const Tensor& other) {
  PT_KERNEL_BEGIN;

  bool isSelf_0d = false;
  bool isOther_0d = false;
  if (self.dim() == 0) {
    self.unsafeGetTensorImpl()->set_sizes_and_strides({1}, {1});
    isSelf_0d = true;
  }
  if (other.dim() == 0) {
    other.unsafeGetTensorImpl()->set_sizes_and_strides({1}, {1});
    isOther_0d = true;
  }
  size_t device_id = self.device().index();
  auto& device = synapse_helpers::HPURegistrar::get_device(device_id);
  at::ScalarType scalar_type = self.scalar_type();
  std::string node_type =
      "div_mod_fwd_" + habana_helpers::name_suffix_from_type(scalar_type);

  // create the operator
  habana::RemainderInplaceWrapperOperator Op(device_id, scalar_type);
  std::vector<c10::IValue> stack = {IValue(self), IValue(other)};
  size_t key = Op.GetRecipeKey(node_type, stack);

  std::vector<at::Tensor> pt_inputs{self, other};
  if (device.get_recipe_handle_cache().isCached(key)) {
    PT_KERNEL_DEBUG("Cache hit key:", key);
    Op.SetPTInputs(pt_inputs);
    Op.SetPTOutput(self);
    Op.Execute(key);
  } else {
    PT_KERNEL_DEBUG("Key:", key);
    // create graph
    auto graph = habana_helpers::create_graph(device_id, node_type);

    // Assign Inputs to the Operator
    Op.AllocateSynapseInputs(graph, pt_inputs, true);

    // Build Params for the graph
    Op.AllocateAndAddSynapseNode(graph, stack, true);

    // compile and execute the graph
    Op.Compile(graph);
  }
  std::vector<at::Tensor> out = Op.GetOutputs();
  TORCH_CHECK(out.size() == 1, "Incorrect size of outputs");

  if (isSelf_0d && isOther_0d) {
    self.unsafeGetTensorImpl()->set_sizes_and_strides({}, {});
  }
  if (isOther_0d) {
    other.unsafeGetTensorImpl()->set_sizes_and_strides({}, {});
  }
  PT_KERNEL_END;

  return self;
}

Tensor remainder_scalar_hpu(const Tensor& self, const at::Scalar& other) {
  PT_KERNEL_BEGIN;
  bool isSelf_0d = false;
  if (self.dim() == 0) {
    self.unsafeGetTensorImpl()->set_sizes_and_strides({1}, {1});
    isSelf_0d = true;
  }
  size_t device_id = self.device().index();
  auto& device = synapse_helpers::HPURegistrar::get_device(device_id);
  at::ScalarType scalar_type = self.scalar_type();
  std::string node_type =
      "div_mod_fwd_" + habana_helpers::name_suffix_from_type(scalar_type);

  // create the operator
  habana::RemainderWrapperOperator Op(device_id, scalar_type);
  std::vector<c10::IValue> stack = {IValue(self), IValue(other)};
  size_t key = Op.GetRecipeKey(node_type, stack);

  std::vector<at::Tensor> pt_inputs{self};
  if (device.get_recipe_handle_cache().isCached(key)) {
    PT_KERNEL_DEBUG("Cache hit key:", key);

    Op.SetPTInputs(pt_inputs);
    Op.SetPTOutputs(stack);
    Op.Execute(key);
  } else {
    PT_KERNEL_DEBUG("Key:", key);
    // create graph
    auto graph = habana_helpers::create_graph(device_id, node_type);

    // Assign Inputs to the Operator
    Op.AllocateSynapseInputs(graph, pt_inputs, true);

    // Build Params for the graph
    Op.AllocateAndAddSynapseNode(graph, stack, true);

    // compile and execute the graph
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

Tensor& remainder_scalar_hpu_(Tensor& self, const at::Scalar& other) {
  PT_KERNEL_BEGIN;

  bool isSelf_0d = false;
  if (self.dim() == 0) {
    self.unsafeGetTensorImpl()->set_sizes_and_strides({1}, {1});
    isSelf_0d = true;
  }
  size_t device_id = self.device().index();
  auto& device = synapse_helpers::HPURegistrar::get_device(device_id);
  at::ScalarType scalar_type = self.scalar_type();
  std::string node_type =
      "div_mod_fwd_" + habana_helpers::name_suffix_from_type(scalar_type);

  // create the operator
  habana::RemainderInplaceWrapperOperator Op(device_id, scalar_type);
  std::vector<c10::IValue> stack = {IValue(self), IValue(other)};
  size_t key = Op.GetRecipeKey(node_type, stack);

  std::vector<at::Tensor> pt_inputs{self};
  if (device.get_recipe_handle_cache().isCached(key)) {
    PT_KERNEL_DEBUG("Cache hit key:", key);
    Op.SetPTInputs(pt_inputs);
    Op.SetPTOutput(self);
    Op.Execute(key);
  } else {
    PT_KERNEL_DEBUG("Key:", key);
    // create graph
    auto graph = habana_helpers::create_graph(device_id, node_type);

    // Assign Inputs to the Operator
    Op.AllocateSynapseInputs(graph, pt_inputs, true);

    // Build Params for the graph
    Op.AllocateAndAddSynapseNode(graph, stack, true);

    // compile and execute the graph
    Op.Compile(graph);
  }
  std::vector<at::Tensor> out = Op.GetOutputs();
  TORCH_CHECK(out.size() == 1, "Incorrect size of outputs");

  if (isSelf_0d) {
    self.unsafeGetTensorImpl()->set_sizes_and_strides({}, {});
  }
  PT_KERNEL_END;

  return self;
}

void habana::RemainderOutOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    bool is_output_persistent) {
  static_cast<void>(is_output_persistent);
  TORCH_CHECK(
      inputs.size() == 3,
      "Incorrect size of inputs expected for topk operator");
  TORCH_CHECK(
      inputs[0].isTensor(),
      "Input arg1 expected to be tensor for topk operator");
  Tensor self = inputs[0].toTensor();
  auto other = inputs[1].toTensor();
  auto remainder = inputs[2].toTensor();
  auto out_shape = BinaryOperator::compute_output_shape(self, other);
  auto quotient = habana_helpers::createPTTensor(
      self,
      IntArrayRef(out_shape.data(), out_shape.size()),
      self.options(),
      self.suggest_memory_format(),
      false);

  AllocateSynapseOutput(graph, quotient, false);
  synapse_helpers::tensor& arg1_syn_tensor = p_context_->syn_inputs_[0];
  synapse_helpers::tensor& arg2_syn_tensor = p_context_->syn_inputs_[1];

  p_context_->syn_outputs_.emplace_back(std::move(p_context_->syn_inputs_[2]));
  p_context_->pt_outputs_.emplace_back(remainder);

  std::vector<synTensor> syn_inputs;
  syn_inputs.push_back(arg1_syn_tensor.get());
  syn_inputs.push_back(arg2_syn_tensor.get());

  synapse_helpers::tensor& output1_syn_tensor = p_context_->syn_outputs_[0];
  synapse_helpers::tensor& output2_syn_tensor = p_context_->syn_outputs_[1];
  std::vector<synTensor> syn_outputs{
      output1_syn_tensor.get(), output2_syn_tensor.get()};

  graph.add_node(
      std::move(syn_inputs),
      std::move(syn_outputs),
      nullptr,
      0,
      std::move(guid_));
}

void habana::RemainderOutWrapperOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    bool is_output_persistent) {
  TORCH_CHECK(
      inputs.size() == 3,
      "Incorrect size of input expected for Remainder operator");
  // Note that there is no (Scalar, Tensor) version for remainder ops
  // in native_functions.yaml
  TORCH_CHECK(inputs[0].isTensor(), "Input arg1 type expected to be tensor");
  TORCH_CHECK(
      inputs[1].isTensor() || inputs[1].isScalar(),
      "Input arg2 type expected to be a tensor or scalar");
  TORCH_CHECK(inputs[2].isTensor(), "Input arg3 type expected to be tensor");

  auto remainderOp = make_operator<RemainderOutOperator>(
      this->p_context_->device_id_, this->scalarType_);

  if (inputs[1].isTensor()) { // Both inputs are tensors
    remainderOp->SetSynapseInput(p_context_->syn_inputs_[0]);
    remainderOp->SetSynapseInput(p_context_->syn_inputs_[1]);
    remainderOp->SetSynapseInput(p_context_->syn_inputs_[2]);
    remainderOp->AllocateAndAddSynapseNode(graph, inputs, is_output_persistent);
  } else { // 2nd input is a scalar
    // add constant node to convert 2nd input to tensor
    auto arg1 = inputs[0].toTensor();
    auto constOp = make_operator<ConstantOperator>(
        this->p_context_->device_id_, this->scalarType_);
    auto const_shape_tensor = habana_helpers::createPTTensor(
        arg1, {1}, arg1.options(), at::MemoryFormat::Contiguous, false);
    torch::jit::Stack constOp_stack = {IValue(const_shape_tensor), inputs[1]};
    constOp->AllocateAndAddSynapseNode(graph, constOp_stack, false);
    remainderOp->SetSynapseInput(p_context_->syn_inputs_[0]);
    remainderOp->SetSynapseInput(constOp->GetSynOutputs()[0]);
    remainderOp->SetSynapseInput(p_context_->syn_inputs_[1]);
    // replace 2nd scalar input with a tensor in stack
    inputs.erase(inputs.cbegin() + 1);
    inputs.emplace(inputs.cbegin() + 1, constOp->GetOutputs()[0]);
    remainderOp->AllocateAndAddSynapseNode(graph, inputs, is_output_persistent);
  }

  p_context_->pt_outputs_.emplace_back(remainderOp->GetOutputs()[1]);
  p_context_->syn_outputs_.emplace_back(
      std::move(remainderOp->GetSynOutputs()[1]));
}

Tensor& remainder_tensor_hpu_out(
    const Tensor& self,
    const Tensor& other,
    Tensor& result) {
  PT_KERNEL_BEGIN;

  bool isSelf_0d = false;
  bool isOther_0d = false;
  if (self.dim() == 0) {
    self.unsafeGetTensorImpl()->set_sizes_and_strides({1}, {1});
    isSelf_0d = true;
  }
  if (other.dim() == 0) {
    other.unsafeGetTensorImpl()->set_sizes_and_strides({1}, {1});
    isOther_0d = true;
  }
  if (result.dim() == 0) {
    result.unsafeGetTensorImpl()->set_sizes_and_strides({1}, {1});
  }

  auto out_shape = habana::BinaryOperator::compute_output_shape(self, other);
  auto out_reshaped = result.unsafeGetTensorImpl();
  if (result.sizes().vec() != out_shape) {
    THHTensor_resizeNd(
        out_reshaped, out_shape.size(), out_shape.data(), nullptr);
  }

  at::ScalarType scalar_type = self.scalar_type();
  std::string node_type =
      "div_mod_fwd_" + habana_helpers::name_suffix_from_type(scalar_type);

  // Create the operator
  size_t device_id = self.device().index();
  auto& device = synapse_helpers::HPURegistrar::get_device(device_id);
  std::vector<c10::IValue> stack = {
      IValue(self), IValue(other), IValue(result)};
  habana::RemainderOutWrapperOperator Op(device_id, scalar_type);
  size_t key = Op.GetRecipeKey(node_type, stack);

  std::vector<at::Tensor> pt_inputs{self, other, result};
  if (device.get_recipe_handle_cache().isCached(key)) {
    PT_KERNEL_DEBUG("Cache hit key:", key);
    Op.SetPTInputs(pt_inputs);
    Op.SetPTOutput(result);
    Op.Execute(key);
  } else {
    PT_KERNEL_DEBUG("Key:", key);
    // Create Graph
    auto graph = habana_helpers::create_graph(device_id, node_type);

    // Assign Inputs to the Operator
    Op.AllocateSynapseInputs(graph, pt_inputs, true);

    // Build Params for the graph
    Op.AllocateAndAddSynapseNode(graph, stack, true);

    // compile and execute the graph
    Op.Compile(graph);
  }

  std::vector<at::Tensor> out = Op.GetOutputs();
  TORCH_CHECK(out.size() == 1, "Incorrect size of outputs");

  if (isSelf_0d && isOther_0d) {
    result.unsafeGetTensorImpl()->set_sizes_and_strides({}, {});
  }
  if (isSelf_0d) {
    self.unsafeGetTensorImpl()->set_sizes_and_strides({}, {});
  }
  if (isOther_0d) {
    other.unsafeGetTensorImpl()->set_sizes_and_strides({}, {});
  }
  PT_KERNEL_END;
  return result;
}

Tensor& remainder_scalar_hpu_out(
    const Tensor& self,
    at::Scalar other,
    Tensor& result) {
  PT_KERNEL_BEGIN;

  bool isSelf_0d = false;
  if (self.dim() == 0) {
    self.unsafeGetTensorImpl()->set_sizes_and_strides({1}, {1});
    isSelf_0d = true;
  }
  if (result.dim() == 0) {
    result.unsafeGetTensorImpl()->set_sizes_and_strides({1}, {1});
  }

  auto out_shape = self.sizes().vec();
  auto out_reshaped = result.unsafeGetTensorImpl();
  if (result.sizes().vec() != out_shape) {
    THHTensor_resizeNd(
        out_reshaped, out_shape.size(), out_shape.data(), nullptr);
  }

  at::ScalarType scalar_type = self.scalar_type();
  std::string node_type =
      "div_mod_fwd_" + habana_helpers::name_suffix_from_type(scalar_type);

  // Create the operator
  size_t device_id = self.device().index();
  auto& device = synapse_helpers::HPURegistrar::get_device(device_id);
  std::vector<c10::IValue> stack = {
      IValue(self), IValue(other), IValue(result)};
  habana::RemainderOutWrapperOperator Op(device_id, scalar_type);
  size_t key = Op.GetRecipeKey(node_type, stack);

  std::vector<at::Tensor> pt_inputs{self, result};
  if (device.get_recipe_handle_cache().isCached(key)) {
    PT_KERNEL_DEBUG("Cache hit key:", key);
    Op.SetPTInputs(pt_inputs);
    Op.SetPTOutput(result);
    Op.Execute(key);
  } else {
    PT_KERNEL_DEBUG("Key:", key);
    // Create Graph
    auto graph = habana_helpers::create_graph(device_id, node_type);

    // Assign Inputs to the Operator
    Op.AllocateSynapseInputs(graph, pt_inputs, true);

    // Build Params for the graph
    Op.AllocateAndAddSynapseNode(graph, stack, true);

    // compile and execute the graph
    Op.Compile(graph);
  }

  std::vector<at::Tensor> out = Op.GetOutputs();
  TORCH_CHECK(out.size() == 1, "Incorrect size of outputs");

  if (isSelf_0d) {
    self.unsafeGetTensorImpl()->set_sizes_and_strides({}, {});
    result.unsafeGetTensorImpl()->set_sizes_and_strides({}, {});
  }
  PT_KERNEL_END;
  return result;
}

static auto& KernelRegistry =
    habana::KernelRegistry()
        .add(
            "aten::add.Tensor",
            [](const int device_id, c10::ScalarType node_type) {
              return std::make_shared<habana::AddOperator>(
                  device_id, node_type);
            })
        .add(
            "aten::add.Scalar",
            [](const int device_id, c10::ScalarType node_type) {
              return std::make_shared<habana::AddOperator>(
                  device_id, node_type);
            })
        .add(
            "aten::sub.Tensor",
            [](const int device_id, c10::ScalarType node_type) {
              return std::make_shared<habana::SubOperator>(
                  device_id, node_type);
            })
        .add(
            "aten::sub.Scalar",
            [](const int device_id, c10::ScalarType node_type) {
              return std::make_shared<habana::SubOperator>(
                  device_id, node_type);
            })
        .add(
            "hpu::rsub.Tensor",
            [](const int device_id, c10::ScalarType node_type) {
              return std::make_shared<habana::RsubOperator>(
                  device_id, node_type);
            })
        .add(
            "aten::rsub.Scalar",
            [](const int device_id, c10::ScalarType node_type) {
              return std::make_shared<habana::RsubOperator>(
                  device_id, node_type);
            })
        .add(
            "aten::mul.Tensor",
            [](const int device_id, c10::ScalarType node_type) {
              return std::make_shared<habana::MulOperator>(
                  device_id, node_type);
            })
        .add(
            "aten::mul.Scalar",
            [](const int device_id, c10::ScalarType node_type) {
              return std::make_shared<habana::MulOperator>(
                  device_id, node_type);
            })
        .add(
            "aten::div.Tensor",
            [](const int device_id, c10::ScalarType node_type) {
              return std::make_shared<habana::DivOperator>(
                  device_id, node_type);
            })
        .add(
            "aten::div.Scalar",
            [](const int device_id, c10::ScalarType node_type) {
              return std::make_shared<habana::DivOperator>(
                  device_id, node_type);
            })
        .add(
            "aten::pow.Tensor_Scalar",
            [](const int device_id, c10::ScalarType node_type) {
              return std::make_shared<habana::PowOperator>(
                  device_id, node_type);
            })
        .add(
            "aten::pow.Tensor_Tensor",
            [](const int device_id, c10::ScalarType node_type) {
              return std::make_shared<habana::PowOperator>(
                  device_id, node_type);
            })
        .add(
            "aten::pow.Scalar",
            [](const int device_id, c10::ScalarType node_type) {
              return std::make_shared<habana::PowOperator>(
                  device_id, node_type);
            })
        .add(
            "aten::maximum",
            [](const int device_id, c10::ScalarType node_type) {
              return std::make_shared<habana::MaximumOperator>(
                  device_id, node_type);
            })
        .add(
            "aten::minimum",
            [](const int device_id, c10::ScalarType node_type) {
              return std::make_shared<habana::MinimumOperator>(
                  device_id, node_type);
            })
        .add(
            "aten::remainder.Tensor",
            [](const int device_id, c10::ScalarType node_type) {
              return std::make_shared<habana::RemainderWrapperOperator>(
                  device_id, node_type);
            })
        .add(
            "aten::remainder_.Tensor",
            [](const int device_id, c10::ScalarType node_type) {
              return std::make_shared<habana::RemainderInplaceWrapperOperator>(
                  device_id, node_type);
            })
        .add(
            "aten::remainder.Scalar",
            [](const int device_id, c10::ScalarType node_type) {
              return std::make_shared<habana::RemainderWrapperOperator>(
                  device_id, node_type);
            })
        .add(
            "aten::remainder_.Scalar",
            [](const int device_id, c10::ScalarType node_type) {
              return std::make_shared<habana::RemainderInplaceWrapperOperator>(
                  device_id, node_type);
            })
        .add(
            "aten::remainder.Tensor_out",
            [](const int device_id, c10::ScalarType node_type) {
              return std::make_shared<habana::RemainderOutWrapperOperator>(
                  device_id, node_type);
            })
        .add(
            "aten::remainder.Scalar_out",
            [](const int device_id, c10::ScalarType node_type) {
              return std::make_shared<habana::RemainderOutWrapperOperator>(
                  device_id, node_type);
            });
