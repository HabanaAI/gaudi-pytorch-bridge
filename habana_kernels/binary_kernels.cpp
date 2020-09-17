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

#include "habana_device/HPUCheck.h"
#include "habana_device/hpu_cached_devices.h"
#include "habana_helpers/graph.h"
#include "habana_helpers/tensor_utils.h"
#include "habana_kernels/binary_kernels.h"
#include "habana_kernels/kernel_utils.h"
#include "habana_kernels/simple_generic_kernel.h"
#include "habana_kernels/tensor_shape_kernels.h"

using namespace torch;

void check_ew_kernel_constraints(const Tensor& arg1, const Tensor& arg2) {
  TORCH_CHECK(
      arg1.scalar_type() == arg2.scalar_type(),
      "Types don't match. arg1 type: ",
      arg1.scalar_type(),
      " arg2 type: ",
      arg2.scalar_type());
  // Since binary ops are required to broadcast, we don't check tensor sizes
}

// if the tensor is in CPU push it to HPU. Further if the CPU tensor is of
// double dtype typecast to float. This workaround needed if
// one the binary operand of torch op is scalar. TODO: [SW-9849]
static inline Tensor get_hpu_tensor(Tensor input) {
  Tensor output;
  if (input.device().type() == c10::DeviceType::CPU) {
    if (input.scalar_type() == c10::ScalarType::Double) {
      output = input.to(c10::ScalarType::Float).to(c10::DeviceType::HABANA);
    } else {
      output = input.to(c10::DeviceType::HABANA);
    }
  } else {
    output = input;
  }

  return output;
}

inline Tensor convert_scalar_to_tensor_using_self(
    const Tensor& self,
    Scalar other) {
  if (self.dim() == 0) {
    self.unsafeGetTensorImpl()->set_sizes_and_strides({1}, {1});
  }
  auto self_hpu = get_hpu_tensor(self);
  auto result = habana_helpers::scalar_to_device_tensor(
      other, self_hpu, self_hpu.ndimension());

  return result;
}

// helper that finally interfaces with synapse generic kernel
static inline Tensor& do_binary_op(
    Tensor& out,
    const Tensor& operand1,
    const Tensor& operand2,
    const std::string& op,
    SynapsePassType pass_type) {
  std::vector<at::Tensor> pt_inputs;
  pt_inputs.push_back(operand1);
  pt_inputs.push_back(operand2);
  std::vector<at::Tensor> pt_outputs;
  pt_outputs.push_back(out);
  synapse_simple_generic_kernel(
      pt_outputs, pt_inputs, op, nullptr, 0, pass_type);
  return out;
}

// helper that finally interfaces with synapse generic inplace kernel
static inline Tensor& do_binary_inplace_op(
    Tensor& self,
    const Tensor& other,
    const std::string& op,
    SynapsePassType pass_type) {
  std::vector<at::Tensor> pt_inputs;
  pt_inputs.push_back(self);
  pt_inputs.push_back(other);
  // Build Params for the graph
  std::vector<c10::IValue> stack = {IValue(self), IValue(other)};
  std::string node_type = (SynapsePassType::NO_PASS == pass_type) ? op
                                                                  : op +
          std::string((SynapsePassType::FORWARD_PASS == pass_type) ? "_fwd_"
                                                                   : "_bwd_") +
          habana_helpers::name_suffix_from_type(pt_inputs[0].scalar_type());
  const auto device_id = pt_inputs[0].device().index();
  auto& device = synapse_helpers::HPURegistrar::get_device(device_id);
  size_t key = habana_helpers::getRecipeKey(node_type, stack, true);
  if (device.get_recipe_handle_cache().isCached(key)) {
    PT_KERNEL_DEBUG("Cache hit key:", key);
    synapse_execute_cached_inplace_kernel(pt_inputs, device_id, key);
  } else {
    PT_KERNEL_DEBUG("key:", key);
    synapse_execute_inplace_kernel(
        pt_inputs, node_type, nullptr, 0, device_id, key);
  }
  return self;
}

// generic binary tensor op interface that takes care of broadcasting
// semantics requirements
static inline void do_generic_tensor_binary_op_inplace(
    Tensor& self,
    const Tensor& operand2,
    const std::string& op,
    SynapsePassType pass_type) {
  auto operand2_hpu = get_hpu_tensor(operand2);
  check_ew_kernel_constraints(self, operand2_hpu);
  TORCH_CHECK(
      self.ndimension() >= operand2.ndimension(),
      "Binary inplace ops shouldn't get self.ndimension() < other.ndimension()");
  auto out_dims = self.ndimension();
  // Make sure that we give tensors that match dims to Synapse
  auto operand2_sizes = operand2.sizes().vec();
  // Create view_sizes initialized to part which has size=1 for upper dims
  auto view_sizes = std::vector<int64_t>(out_dims - operand2.ndimension(), 1);
  // and append the smaller tensor dims
  view_sizes.insert(
      view_sizes.end(), operand2_sizes.begin(), operand2_sizes.end());
  auto expanded_operand2_tensor = operand2_hpu.view(view_sizes);
  do_binary_inplace_op(self, expanded_operand2_tensor, op, pass_type);
  return;
}

// generic binary tensor op interface that takes care of broadcasting
// semantics requirements
inline void do_generic_tensor_binary_op_out(
    Tensor& output,
    const Tensor& operand1,
    const Tensor& operand2,
    const std::string& op,
    SynapsePassType pass_type) {
  auto operand1_hpu = get_hpu_tensor(operand1);
  auto operand2_hpu = get_hpu_tensor(operand2);

  check_ew_kernel_constraints(operand1_hpu, operand2_hpu);

  auto out_sizes = output.sizes().vec();
  auto out_dims = output.ndimension();
  // Make sure that we give tensors that match dims to Synapse
  if (operand1.ndimension() > operand2.ndimension()) {
    auto operand2_sizes = operand2.sizes().vec();
    // Create view_sizes initialized to part which has size=1 for upper dims
    auto view_sizes = std::vector<int64_t>(out_dims - operand2.ndimension(), 1);
    // and append the smaller tensor dims
    view_sizes.insert(
        view_sizes.end(), operand2_sizes.begin(), operand2_sizes.end());
    auto expanded_operand2_tensor = operand2_hpu.view(view_sizes);
    output = do_binary_op(
        output, operand1_hpu, expanded_operand2_tensor, op, pass_type);
  } else {
    auto operand1_sizes = operand1.sizes().vec();
    // Create view_sizes initialized to part which has size=1 for upper dims
    auto view_sizes = std::vector<int64_t>(out_dims - operand1.ndimension(), 1);
    view_sizes.insert(
        view_sizes.end(), operand1_sizes.begin(), operand1_sizes.end());
    auto operand1_expanded = operand1_hpu.view(view_sizes);
    output =
        do_binary_op(output, operand1_expanded, operand2_hpu, op, pass_type);
  }
}

// scalar*tensor helper
static inline Tensor do_tensor_scalar_mul(const Tensor& tensor, Scalar alpha) {
  if (alpha.toFloat() == 1.0)
    return tensor;
  auto alpha_tensor = habana_helpers::scalar_to_device_tensor(
      alpha, tensor, tensor.ndimension());
  auto out_mul = at::mul(tensor, alpha_tensor);
  return out_mul;
}

// scalar*scalar helper
static inline Tensor do_scalar_scalar_mul(
    const Tensor& self,
    Scalar other,
    Scalar alpha) {
  if (alpha.toFloat() == 1.0) {
    return habana_helpers::scalar_to_device_tensor(
        other, self, self.ndimension());
  }
  if (other.toFloat() == 1.0) {
    return habana_helpers::scalar_to_device_tensor(
        alpha, self, self.ndimension());
  }
  auto other_tensor =
      habana_helpers::scalar_to_device_tensor(other, self, self.ndimension());
  auto alpha_tensor =
      habana_helpers::scalar_to_device_tensor(alpha, self, self.ndimension());
  auto out_mul = at::mul(other_tensor, alpha_tensor);

  return out_mul;
}

inline Tensor get_correct_input_tensor(const Tensor& arg1, const Tensor& arg2) {
  auto arg_final = arg1.ndimension() > arg2.ndimension()
      ? arg1
      : arg1.numel() > arg2.numel() ? arg1 : arg2;
  return arg_final;
}

void habana::BinaryOperator::insert_reshape_op(
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

  bool isArg1modified = false, isArg2modified = false;
  std::vector<synapse_helpers::tensor_or_ref> reshape_syn_output;
  auto out_dims = arg1.ndimension() > arg2.ndimension() ? arg1.ndimension()
                                                        : arg2.ndimension();

  ReshapeOperator reshapeOp(this->p_context_->device_id_, this->scalarType_);
  // Make sure that we give tensors that match dims to Synapse
  if (arg1.ndimension() > arg2.ndimension()) {
    isArg2modified = true;
    insert_reshape_op(graph, reshapeOp, arg2, 1, out_dims);
    reshape_syn_output.push_back(std::move(reshapeOp.GetSynOutputs()[0]));
  } else if (arg1.ndimension() < arg2.ndimension()) {
    isArg1modified = true;
    insert_reshape_op(graph, reshapeOp, arg1, 0, out_dims);
    reshape_syn_output.push_back(std::move(reshapeOp.GetSynOutputs()[0]));
  }

  auto operand = get_correct_input_tensor(arg1, arg2);
  auto output = habana_helpers::createPTTensor(operand, is_output_persistent);
  AllocateSynapseOutput(graph, output, is_output_persistent);
  synapse_helpers::tensor& arg1_syn_tensor =
      isArg1modified ? reshape_syn_output[0] : p_context_->syn_inputs_[0];
  synapse_helpers::tensor& arg2_syn_tensor =
      isArg2modified ? reshape_syn_output[0] : p_context_->syn_inputs_[1];

  std::vector<synTensor> syn_inputs{arg1_syn_tensor.get(),
                                    arg2_syn_tensor.get()};

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
  TORCH_CHECK(inputs[0].isTensor() || inputs[1].isTensor(),
      "At least one of the inputs arg1 or arg2 expected to be a tensor");
  // Note that pow has a (Scalar, Tensor) variant in native_functions.yaml
  // although mul and div do not
  TORCH_CHECK(inputs[0].isTensor() || inputs[0].isScalar(),
      "Input arg1 type expected to be a tensor or scalar");
  TORCH_CHECK(inputs[1].isTensor() || inputs[1].isScalar(),
      "Input arg2 type expected to be a tensor or scalar");

  BinaryOperator binaryOp(
      this->p_context_->device_id_, guid_, this->scalarType_);

  if (inputs[0].isTensor() && inputs[1].isTensor()) { // Both inputs are tensors
    auto& syn_arg1 =
        binaryOp.SetSynapseInput(std::move(p_context_->syn_inputs_[0]));
    auto& syn_arg2 =
        binaryOp.SetSynapseInput(std::move(p_context_->syn_inputs_[1]));
    binaryOp.AllocateAndAddSynapseNode(graph, inputs, is_output_persistent);
    p_context_->syn_inputs_[0] = std::move(syn_arg1);
    p_context_->syn_inputs_[1] = std::move(syn_arg2);

  } else if (inputs[0].isTensor() && inputs[1].isScalar()) { // 2nd input is a scalar
    // add constant node to convert 2nd input to tensor
    ConstantOperator constOp(this->p_context_->device_id_, this->scalarType_);
    constOp.AllocateAndAddSynapseNode(graph, inputs, false);
    auto& syn_arg1 =
        binaryOp.SetSynapseInput(std::move(p_context_->syn_inputs_[0]));
    UNUSED auto& syn_arg2 =
        binaryOp.SetSynapseInput(std::move(constOp.GetSynOutputs()[0]));
    // replace input scalar with input tensor in the stack
    inputs.pop_back();
    inputs.emplace_back(constOp.GetOutputs()[0]);
    binaryOp.AllocateAndAddSynapseNode(graph, inputs, is_output_persistent);
    p_context_->syn_inputs_[0] = std::move(syn_arg1);

  } else { // 1st input is a scalar
    // add constant node to convert 1st input to tensor
    ConstantOperator constOp(this->p_context_->device_id_, this->scalarType_);
    torch::jit::Stack constOp_stack = {inputs[1], inputs[0]};
    constOp.AllocateAndAddSynapseNode(graph, constOp_stack, false);
    UNUSED auto& syn_arg1 =
        binaryOp.SetSynapseInput(std::move(constOp.GetSynOutputs()[0]));
    auto& syn_arg2 =
        binaryOp.SetSynapseInput(std::move(p_context_->syn_inputs_[0]));
    // replace input scalar with input tensor in the stack
    inputs.erase(inputs.cbegin());
    inputs.emplace(inputs.cbegin(), constOp.GetOutputs()[0]);
    binaryOp.AllocateAndAddSynapseNode(graph, inputs, is_output_persistent);
    p_context_->syn_inputs_[0] = std::move(syn_arg2);
  }

  p_context_->pt_outputs_.emplace_back(binaryOp.GetOutputs()[0]);
  p_context_->syn_outputs_.emplace_back(std::move(binaryOp.GetSynOutputs()[0]));
}

void habana::BinaryWrapperOperator::SetPTOutputs(torch::jit::Stack& inputs) {
  Tensor output;
  if (inputs[0].isTensor() && inputs[1].isTensor()) {
    auto operand =
        get_correct_input_tensor(inputs[0].toTensor(), inputs[1].toTensor());
    output = at::empty(
        operand.sizes(), operand.options(), operand.suggest_memory_format());
  } else if (inputs[0].isTensor()) {
    auto operand = inputs[0].toTensor();
    output = at::empty(
        operand.sizes(), operand.options(), operand.suggest_memory_format());
  } else {
    auto operand = inputs[1].toTensor();
    output = at::empty(
        operand.sizes(), operand.options(), operand.suggest_memory_format());
  }
  HabanaOperator::SetPTOutputs({output});
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

  if (inputs[2].toScalar().toFloat() != 1.0) {
    // Multiplication between arg2 and alpha is required
    habana::MulOperator mulOp(this->p_context_->device_id_, this->scalarType_);
    auto& arg2_syn =
        mulOp.SetSynapseInput(std::move(p_context_->syn_inputs_[1]));
    torch::jit::Stack mulOp_stack{inputs[1], inputs[2]};

    mulOp.AllocateAndAddSynapseNode(graph, mulOp_stack, false);
    synapse_helpers::tensor_or_ref& mulOp_out = mulOp.GetSynOutputs()[0];

    bool isArg1modified = false, isArg2modified = false;
    std::vector<synapse_helpers::tensor_or_ref> reshape_syn_output;
    auto out_dims = arg1.ndimension() > arg2.ndimension() ? arg1.ndimension()
                                                          : arg2.ndimension();
    ReshapeOperator reshapeOp(this->p_context_->device_id_, this->scalarType_);
    // Make sure that we give tensors that match dims to Synapse
    if (arg1.ndimension() > arg2.ndimension()) {
      isArg2modified = true;
      // Moving the output of mulOp to syn_inputs_[1] prior to reshape
      p_context_->syn_inputs_[1] = std::move(mulOp_out);
      insert_reshape_op(graph, reshapeOp, arg2, 1, out_dims);
      reshape_syn_output.push_back(std::move(reshapeOp.GetSynOutputs()[0]));
    } else if (arg1.ndimension() < arg2.ndimension()) {
      isArg1modified = true;
      insert_reshape_op(graph, reshapeOp, arg1, 0, out_dims);
      reshape_syn_output.push_back(std::move(reshapeOp.GetSynOutputs()[0]));
    }
    // Restoring arg2_syn to syn_inputs[1] in case it was earlier assigned to
    // mulOp output
    p_context_->syn_inputs_[1] = std::move(arg2_syn);

    auto operand = get_correct_input_tensor(arg1, arg2);
    auto output = habana_helpers::createPTTensor(operand, is_output_persistent);

    AllocateSynapseOutput(graph, output, is_output_persistent);

    synapse_helpers::tensor& arg1_syn_tensor =
        isArg1modified ? reshape_syn_output[0] : p_context_->syn_inputs_[0];
    synapse_helpers::tensor& arg2_syn_tensor =
        isArg2modified ? reshape_syn_output[0] : mulOp_out;

    std::vector<synTensor> syn_inputs{arg1_syn_tensor.get(),
                                      arg2_syn_tensor.get()};

    synapse_helpers::tensor& output_syn_tensor = p_context_->syn_outputs_[0];
    std::vector<synTensor> syn_outputs{output_syn_tensor.get()};

    graph.add_node(
        std::move(syn_inputs),
        std::move(syn_outputs),
        nullptr,
        0,
        std::move(guid_));
  } else {
    bool isArg1modified = false, isArg2modified = false;
    std::vector<synapse_helpers::tensor_or_ref> reshape_syn_output;
    auto out_dims = arg1.ndimension() > arg2.ndimension() ? arg1.ndimension()
                                                          : arg2.ndimension();
    ReshapeOperator reshapeOp(this->p_context_->device_id_, this->scalarType_);
    // Make sure that we give tensors that match dims to Synapse
    if (arg1.ndimension() > arg2.ndimension()) {
      isArg2modified = true;
      insert_reshape_op(graph, reshapeOp, arg2, 1, out_dims);
      reshape_syn_output.push_back(std::move(reshapeOp.GetSynOutputs()[0]));
    } else if (arg1.ndimension() < arg2.ndimension()) {
      isArg1modified = true;
      insert_reshape_op(graph, reshapeOp, arg1, 0, out_dims);
      reshape_syn_output.push_back(std::move(reshapeOp.GetSynOutputs()[0]));
    }

    auto operand = get_correct_input_tensor(arg1, arg2);
    auto output = habana_helpers::createPTTensor(operand, is_output_persistent);
    AllocateSynapseOutput(graph, output, is_output_persistent);

    synapse_helpers::tensor& arg1_syn_tensor =
        isArg1modified ? reshape_syn_output[0] : p_context_->syn_inputs_[0];
    synapse_helpers::tensor& arg2_syn_tensor =
        isArg2modified ? reshape_syn_output[0] : p_context_->syn_inputs_[1];

    std::vector<synTensor> syn_inputs{arg1_syn_tensor.get(),
                                      arg2_syn_tensor.get()};

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
  TORCH_CHECK(inputs[0].isTensor() || inputs[1].isTensor(),
      "At least one of the inputs arg1 or arg2 expected to be a tensor");
  TORCH_CHECK(
      inputs[0].isTensor() || inputs[0].isScalar(),
      "Input arg1 type expected to be a tensor or scalar");
  TORCH_CHECK(
      inputs[1].isTensor() || inputs[1].isScalar(),
      "Input arg2 type expected to be a tensor or scalar");
  TORCH_CHECK(inputs[2].isScalar(), "Input arg3 type expected to be scalar");

  BinaryOperatorWithAlpha binaryOp(
      this->p_context_->device_id_, guid_, this->scalarType_);

  if (inputs[0].isTensor() && inputs[1].isTensor()) { // First 2 inputs are both tensors
    auto& syn_arg1 =
        binaryOp.SetSynapseInput(std::move(p_context_->syn_inputs_[0]));
    auto& syn_arg2 =
        binaryOp.SetSynapseInput(std::move(p_context_->syn_inputs_[1]));
    binaryOp.AllocateAndAddSynapseNode(graph, inputs, is_output_persistent);
    p_context_->syn_inputs_[0] = std::move(syn_arg1);
    p_context_->syn_inputs_[1] = std::move(syn_arg2);

  } else if (inputs[0].isTensor() && inputs[1].isScalar()) { // 2nd input is a scalar
    // add node to convert scalar to tensor
    ConstantOperator constOp(this->p_context_->device_id_, this->scalarType_);
    constOp.AllocateAndAddSynapseNode(graph, inputs, false);
    auto& syn_arg1 =
        binaryOp.SetSynapseInput(std::move(p_context_->syn_inputs_[0]));
    UNUSED auto& syn_arg2 =
        binaryOp.SetSynapseInput(std::move(constOp.GetSynOutputs()[0]));
    // replace 2nd scalar input with a tensor in stack
    inputs.erase(inputs.cbegin() + 1);
    inputs.emplace(inputs.cbegin() + 1, constOp.GetOutputs()[0]);
    binaryOp.AllocateAndAddSynapseNode(graph, inputs, is_output_persistent);
    p_context_->syn_inputs_[0] = std::move(syn_arg1);

  } else { // 1st input is a scalar
    // add node to convert scalar to tensor
    ConstantOperator constOp(this->p_context_->device_id_, this->scalarType_);
    torch::jit::Stack constOp_stack = {inputs[1], inputs[0]};
    constOp.AllocateAndAddSynapseNode(graph, constOp_stack, false);
    UNUSED auto& syn_arg1 =
        binaryOp.SetSynapseInput(std::move(constOp.GetSynOutputs()[0]));
    auto& syn_arg2 =
        binaryOp.SetSynapseInput(std::move(p_context_->syn_inputs_[0]));
    // replace 1st scalar input with a tensor in stack
    inputs.erase(inputs.cbegin());
    inputs.emplace(inputs.cbegin(), constOp.GetOutputs()[0]);
    binaryOp.AllocateAndAddSynapseNode(graph, inputs, is_output_persistent);
    p_context_->syn_inputs_[0] = std::move(syn_arg2);

  }

  p_context_->pt_outputs_.emplace_back(binaryOp.GetOutputs()[0]);
  p_context_->syn_outputs_.emplace_back(std::move(binaryOp.GetSynOutputs()[0]));
}

void habana::BinaryWrapperOperatorWithAlpha::SetPTOutputs(
    torch::jit::Stack& inputs) {
  Tensor output;
  if (inputs[0].isTensor() && inputs[1].isTensor()) {
    auto operand =
        get_correct_input_tensor(inputs[0].toTensor(), inputs[1].toTensor());
    output = at::empty(
        operand.sizes(), operand.options(), operand.suggest_memory_format());
  } else if (inputs[0].isTensor()) {
    auto operand = inputs[0].toTensor();
    output = at::empty(
        operand.sizes(), operand.options(), operand.suggest_memory_format());
  } else {
    auto operand = inputs[1].toTensor();
    output = at::empty(
        operand.sizes(), operand.options(), operand.suggest_memory_format());
  }
  HabanaOperator::SetPTOutputs({output});
}

/************************************************************************
 * @brief Generic wrapper for all Eager mode Binary OP invocations that
 * are "not" .out or inplace
 ************************************************************************/
template <class BinaryOp>
Tensor process_generic_tensor_binary_op(
    const std::vector<at::Tensor>& pt_inputs,
    torch::jit::Stack& stack,
    const std::string& node_guid) {
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
Tensor add_tensor_hpu(const Tensor& self, const Tensor& other, Scalar alpha) {
  PT_KERNEL_BEGIN;
  Tensor output;
  if ((other.dim() == 0) && (other.scalar_type() == c10::ScalarType::Long) &&
      (other.device().type() == c10::DeviceType::CPU)) {
    /*Fix for BN copy kernel issue. This is getting generated from unused code
    in pytorch when momentum is configured. For now return w/o addition
    // Ref:
    https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/batchnorm.py,
    line - 446 should ideally be placed within if condition
    TPC kernel are not invoked because add and mul kernels do not support
    integer tensors. */

    auto output_cpu = self;
    output = output_cpu.to(c10::DeviceType::HABANA);
    PT_KERNEL_WARN("Unsupported long int addition");
  } else {
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
  }

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
Tensor add_scalar_hpu(const Tensor& self, Scalar other, Scalar alpha) {
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
 * @brief Kernel implementation for inplace Scalar self.add_(other)
 * output = self + alpha * other
 * @param self - first input
 * @param other - second input
 * @param alpha - optional input
 ************************************************************************/
Tensor& add_scalar_hpu_(
    Tensor& self,
    Scalar other) { // TODO: Add test by using an extension module for new op
                    // at python level
  PT_KERNEL_BEGIN;
  auto other_tensor = convert_scalar_to_tensor_using_self(self, other);
  self.add_(other_tensor, 1);
  PT_KERNEL_END;
  return self;
}

// self += alpha * other
Tensor& add_tensor_hpu_(Tensor& self, const Tensor& other, Scalar alpha) {
  PT_KERNEL_BEGIN;
  auto out_mul = do_tensor_scalar_mul(other, alpha);

  do_generic_tensor_binary_op_inplace(
      self, out_mul, "add", SynapsePassType::FORWARD_PASS);

  PT_KERNEL_END;
  return self;
}

/*************************************************************************
 * @brief Kernel implementation for out = torch.sub(self, alpha, other)
 * @param self - first input
 * @param other - second input
 * @param alpha - optional input
 * out = self - alpha * other
 ************************************************************************/
Tensor sub_tensor_hpu(const Tensor& self, const Tensor& other, Scalar alpha) {
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
 * @brief Kernel implementation for inplace torch.sub_(self, alpha, other)
 * @param self - first input
 * @param other - second input
 * @param alpha - optional input
 * self -= alpha * other
 ************************************************************************/
Tensor& sub_tensor_hpu_(Tensor& self, const Tensor& other, Scalar alpha) {
  PT_KERNEL_BEGIN;
  auto out_mul = do_tensor_scalar_mul(other, alpha);
  do_generic_tensor_binary_op_inplace(
      self, out_mul, "sub", SynapsePassType::FORWARD_PASS);
  PT_KERNEL_END;
  return self;
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
    Scalar other,
    Scalar alpha) { // TODO: No way to test this yet from python
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

/*************************************************************************
 * @brief Kernel implementation for inplace Scalar torch.sub_(self, alpha,
 *other)
 * @param self - first input
 * @param other - second input
 * @param alpha - optional input
 * self -= alpha * other
 ************************************************************************/
Tensor& sub_scalar_hpu_(
    Tensor& self,
    Scalar other,
    Scalar alpha) { // TODO: No way to test this yet from python
  PT_KERNEL_BEGIN;
  auto out_mul = do_scalar_scalar_mul(self, other, alpha);
  do_generic_tensor_binary_op_inplace(
      self, out_mul, "sub", SynapsePassType::FORWARD_PASS);
  PT_KERNEL_END;
  return self;
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
Tensor rsub_scalar_hpu(const Tensor& self, Scalar other, Scalar alpha) {
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

// Elementwise multiplication
// self *= other
/*************************************************************************
 * @brief Kernel implementation for inplace torch.mul_(self, other)
 * @param self - first input
 * @param other - second input
 ************************************************************************/
Tensor& mul_tensor_hpu_(Tensor& self, const Tensor& other) {
  PT_KERNEL_BEGIN;
  if (self.is_same(other)) {
    auto &tensor = self.pow_(2.0);
    PT_KERNEL_END;
    return tensor;
  }

  do_generic_tensor_binary_op_inplace(
      self, other, "mult", SynapsePassType::FORWARD_PASS);
  PT_KERNEL_END;
  return self;
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
  if (self.is_same(other)) {
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
Tensor mul_scalar_hpu(const Tensor& self, Scalar other) {
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
 * @brief Kernel implementation for inplace output = self.mul_(Scalar other)
 * @param self - first input
 * @param other - second input
 * self = self * other
 ************************************************************************/
Tensor& mul_scalar_hpu_(Tensor& self, Scalar other) {
  PT_KERNEL_BEGIN;
  auto multiplier_tensor = convert_scalar_to_tensor_using_self(self, other);
  self.mul_(multiplier_tensor);
  PT_KERNEL_END;
  return self;
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

/****************************************************************************
 * @brief Kernel implementation for result = torch.div(input, denom, out=out)
 * @param result - output
 * @param self - first input
 * @param other - second input
 ***************************************************************************/
Tensor& div_tensor_hpu_out(
    Tensor& result,
    const Tensor& self,
    const Tensor& other) {
  PT_KERNEL_BEGIN;
  do_generic_tensor_binary_op_out(
      result, self, other, "div", SynapsePassType::FORWARD_PASS);
  PT_KERNEL_END;
  return result;
}
/*************************************************************************
 * @brief Kernel implementation for inplace torch.div_(self,other)
 * @param self - first input
 * @param other - second input
 ************************************************************************/
Tensor& div_tensor_hpu_(Tensor& self, const Tensor& other) {
  PT_KERNEL_BEGIN;
  do_generic_tensor_binary_op_inplace(
      self, other, "div", SynapsePassType::FORWARD_PASS);
  PT_KERNEL_END;
  return self;
}

/*************************************************************************
 * @brief Kernel implementation for inplace div.Scalar(self,other)
 * @param self - first input
 * @param other - second input of scalar type
 ************************************************************************/
Tensor div_scalar_hpu(
    const Tensor& self,
    Scalar other) { // TODO: Add test by using an extension module for new op
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
 * @brief Kernel implementation for inplace div_.Scalar(self,other)
 * @param self - first input
 * @param other - second input of scalar type
 ************************************************************************/
Tensor& div_scalar_hpu_(
    Tensor& self,
    Scalar other) { // TODO: Add test by using an extension module for new op
                    // at python level
  PT_KERNEL_BEGIN;
  auto divisor_tensor = convert_scalar_to_tensor_using_self(self, other);
  self = self.div_(divisor_tensor);
  PT_KERNEL_END;
  return self;
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
 * @brief Kernel implementation for out = self.pow(other)
 * @param self - first input
 * @param other - second input
 ************************************************************************/
Tensor& pow_tensor_tensor_hpu_(Tensor& self, const Tensor& other) {
  PT_KERNEL_BEGIN;
  do_generic_tensor_binary_op_inplace(
      self, other, "pow", SynapsePassType::FORWARD_PASS);
  PT_KERNEL_END;
  return self;
}

/*************************************************************************
 * @brief Kernel implementation for out = self.pow(,other)
 * @param self [in,out]- Tensor 1D bf16/FP32
 * @param other [in] - Scalar
 ************************************************************************/
Tensor pow_tensor_scalar_hpu(const Tensor& self, Scalar other) {
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

Tensor& pow_tensor_scalar_hpu_(Tensor& self, Scalar other) {
  PT_KERNEL_BEGIN;
  auto exponent_tensor = convert_scalar_to_tensor_using_self(self, other);
  self.pow_(exponent_tensor);
  PT_KERNEL_END;
  return self;
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

static auto& KernelRegistry =
    habana::KernelRegistry()
        .add(
            "aten::add",
            [](const int device_id, c10::ScalarType node_type) {
              return std::make_shared<habana::AddOperator>(
                  device_id, node_type);
            })
        .add(
            "aten::sub",
            [](const int device_id, c10::ScalarType node_type) {
              return std::make_shared<habana::SubOperator>(
                  device_id, node_type);
            })
        .add(
            "aten::rsub",
            [](const int device_id, c10::ScalarType node_type) {
              return std::make_shared<habana::RsubOperator>(
                  device_id, node_type);
            })
        .add(
            "aten::mul",
            [](const int device_id, c10::ScalarType node_type) {
              return std::make_shared<habana::MulOperator>(
                  device_id, node_type);
            })
        .add(
            "aten::div",
            [](const int device_id, c10::ScalarType node_type) {
              return std::make_shared<habana::DivOperator>(
                  device_id, node_type);
            })
        .add("aten::pow", [](const int device_id, c10::ScalarType node_type) {
          return std::make_shared<habana::PowOperator>(device_id, node_type);
        });


