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

static inline Tensor convert_scalar_to_tensor_using_self(
    const Tensor& self,
    Scalar other) {
  if (self.dim() == 0) {
    self.unsafeGetTensorImpl()->set_sizes_and_strides({1}, {1});
  }
  auto self_hpu = get_hpu_tensor(self);
  auto result =
      habana_helpers::scalar_to_device_tensor(other, self_hpu, self_hpu.ndimension());

  return result;
}

// helper that finally interfaces with synapse generic kernel
static inline Tensor& do_binary_op(
    Tensor& out,
    const Tensor& operand1,
    const Tensor& operand2,
    const std::string& op,
    SynapsePassType pass_type) {
  std::vector<const at::Tensor*> pt_inputs;
  pt_inputs.push_back(&operand1);
  pt_inputs.push_back(&operand2);
  std::vector<const at::Tensor*> pt_outputs;
  pt_outputs.push_back(&out);
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
  std::vector<const at::Tensor*> pt_inputs;
  pt_inputs.push_back(&self);
  pt_inputs.push_back(&other);
  // Build Params for the graph
  std::vector<c10::IValue> stack = {IValue(self), IValue(other)};
  std::string node_type = (SynapsePassType::NO_PASS == pass_type) ? op
                                                                  : op +
          std::string((SynapsePassType::FORWARD_PASS == pass_type) ? "_fwd_"
                                                                   : "_bwd_") +
          habana_helpers::name_suffix_from_type(pt_inputs[0]->scalar_type());
  const auto device_id = pt_inputs[0]->device().index();
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
static inline void do_generic_tensor_binary_op_out(
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

static inline Tensor do_generic_tensor_binary_op(
    const Tensor& operand1,
    const Tensor& operand2,
    const std::string& op,
    SynapsePassType pass_type) {
  auto out_sizes = at::infer_size(operand1.sizes(), operand2.sizes());
  auto output = at::empty(out_sizes, operand1.options());
  do_generic_tensor_binary_op_out(output, operand1, operand2, op, pass_type);
  return output;
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

// self += alpha * other
Tensor& add_tensor_hpu_(Tensor& self, const Tensor& other, Scalar alpha) {
  PT_KERNEL_BEGIN;
  auto alpha_tensor =
      habana_helpers::scalar_to_device_tensor(alpha, other, other.ndimension());
  auto out_mul = at::mul(other, alpha_tensor);

  do_generic_tensor_binary_op_inplace(
      self, out_mul, "add", SynapsePassType::FORWARD_PASS);

  PT_KERNEL_END;
  return self;
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
  auto output = at::empty(operand.sizes(), operand.options());
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

void habana::BinaryOperator::SetPTOutputs(torch::jit::Stack& inputs) {
  Tensor operand1 = inputs[0].toTensor();
  Tensor operand2 = inputs[1].toTensor();

  auto operand = get_correct_input_tensor(operand1, operand2);
  auto output = at::empty(operand.sizes(), operand.options());
  HabanaOperator::SetPTOutputs({output});
}

Tensor binary_op_hpu(
    const std::vector<const at::Tensor*>& pt_inputs,
    const std::string& node_type,
    size_t device_id,
    habana::BinaryOperator* Op) {
  PT_KERNEL_BEGIN;
  auto& device = synapse_helpers::HPURegistrar::get_device(device_id);
  std::vector<c10::IValue> stack;
  for (auto pt_input : pt_inputs) {
    stack.emplace_back(IValue(*pt_input));
  }
  size_t key = Op->GetRecipeKey(node_type, stack);

  if (device.get_recipe_handle_cache().isCached(key)) {
    PT_KERNEL_DEBUG("Cache hit key:", key);
    Op->SetPTInputs(pt_inputs);
    Op->SetPTOutputs(stack);
    Op->Execute(key);
  } else {
    PT_KERNEL_DEBUG("key:", key);
    // Create Graph
    auto graph = habana_helpers::create_graph(device_id, node_type);

    // Assign Inputs to the Operator
    Op->AllocateSynapseInputs(graph, pt_inputs, true);

    // both inputs are not required, just to match graph mode stack
    Op->AllocateAndAddSynapseNode(graph, stack, true);

    // compile and execute the graph
    Op->Compile(graph);
  }
  std::vector<at::Tensor> out = Op->GetOutputs();
  TORCH_CHECK(out.size() == 1, "Incorrect size of outputs");
  PT_KERNEL_END;
  return out.at(0);
}

template <class BinaryOp>
Tensor process_generic_tensor_binary_op(
    const Tensor& operand1,
    const Tensor& operand2,
    const std::string& node_guid) {
  PT_KERNEL_BEGIN;
  if (operand1.dim() == 0) {
    operand1.unsafeGetTensorImpl()->set_sizes_and_strides({1}, {1});
  }
  if (operand2.dim() == 0) {
    operand2.unsafeGetTensorImpl()->set_sizes_and_strides({1}, {1});
  }

  auto operand1_hpu = get_hpu_tensor(operand1);
  auto operand2_hpu = get_hpu_tensor(operand2);
  std::vector<const at::Tensor*> pt_inputs;
  pt_inputs.push_back(&operand1_hpu);
  pt_inputs.push_back(&operand2_hpu);

  size_t device_id = pt_inputs[0]->device().index();
  at::ScalarType scalar_type = pt_inputs[0]->scalar_type();

  std::string node_type =
      node_guid + "_fwd_" + habana_helpers::name_suffix_from_type(scalar_type);

  BinaryOp op(device_id, scalar_type);
  auto out = binary_op_hpu(pt_inputs, node_type, device_id, &op);
  PT_KERNEL_END;
  return out;
}

void habana::BinaryOperatorWithAlpha::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    bool is_output_persistent) {
  // this check is for stack during graph execution
  TORCH_CHECK(
      inputs.size() == 3, "Incorrect size of input expected for add operator");
  TORCH_CHECK(inputs[0].isTensor(), "Input 0 type expected to be tensor");
  TORCH_CHECK(inputs[1].isTensor(), "Input 1 type expected to be tensor");
  Tensor arg1 = inputs[0].toTensor();
  Tensor arg2 = inputs[1].toTensor();

  if (inputs[2].isTensor()) {
    Tensor alpha = inputs[2].toTensor();
    TORCH_CHECK(alpha.numel() == 1, "Alpha should be tensor of size 1");

    habana::MulOperator mulOp(this->p_context_->device_id_, this->scalarType_);
    auto& arg2_syn =
        mulOp.SetSynapseInput(std::move(p_context_->syn_inputs_[1]));
    auto& alpha_syn =
        mulOp.SetSynapseInput(std::move(p_context_->syn_inputs_[2]));

    torch::jit::Stack mulOp_stack = {IValue(arg2), IValue(alpha)};
    mulOp.AllocateAndAddSynapseNode(graph, mulOp_stack, false);

    p_context_->syn_inputs_[2] = std::move(alpha_syn);
    synapse_helpers::tensor_or_ref& mulOp_out_syn_tensor = mulOp.GetSynOutputs()[0];

    bool isArg1modified = false, isArg2modified = false;
    std::vector<synapse_helpers::tensor_or_ref> reshape_syn_output;
    auto out_dims = arg1.ndimension() > arg2.ndimension()
        ? arg1.ndimension()
        : arg2.ndimension();
    ReshapeOperator reshapeOp(this->p_context_->device_id_, this->scalarType_);
    // Make sure that we give tensors that match dims to Synapse
    if (arg1.ndimension() > arg2.ndimension()) {
      isArg2modified = true;
      // Moving the output of mulOp to syn_inputs_[1] prior to reshape
      p_context_->syn_inputs_[1] = std::move(mulOp_out_syn_tensor);
      insert_reshape_op(graph, reshapeOp, arg2, 1, out_dims);
      reshape_syn_output.push_back(std::move(reshapeOp.GetSynOutputs()[0]));
    } else if (arg1.ndimension() < arg2.ndimension()) {
      isArg1modified = true;
      insert_reshape_op(graph, reshapeOp, arg1, 0, out_dims);
      reshape_syn_output.push_back(std::move(reshapeOp.GetSynOutputs()[0]));
    }
    // Restoring arg2_syn to syn_inputs[1] in case it was earlier assigned to mulOp output
    p_context_->syn_inputs_[1] = std::move(arg2_syn);

    auto operand = get_correct_input_tensor(arg1, arg2);
    auto output = at::empty(operand.sizes(), operand.options());
    AllocateSynapseOutput(graph, output, is_output_persistent);

    synapse_helpers::tensor& arg1_syn_tensor = isArg1modified
        ? reshape_syn_output[0]
        : p_context_->syn_inputs_[0];
    synapse_helpers::tensor& arg2_syn_tensor = isArg2modified
        ? reshape_syn_output[0]
        : mulOp_out_syn_tensor;

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
    // TODO: [SW-15658] Handle alpha != 1 for add operator in graph mode
    TORCH_CHECK(
        inputs[2].toScalar().toFloat() == 1.0,
        "Alpha is scalar and not equal to 1 - this configuration is not currently supported");
    bool isArg1modified = false, isArg2modified = false;
    std::vector<synapse_helpers::tensor_or_ref> reshape_syn_output;
    auto out_dims = arg1.ndimension() > arg2.ndimension()
        ? arg1.ndimension()
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
    auto output = at::empty(operand.sizes(), operand.options());
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

template <class BinaryOp>
Tensor process_generic_tensor_binary_op(
    const Tensor& operand1,
    const Tensor& operand2,
    const Tensor& alpha,
    const std::string& node_guid) {
  PT_KERNEL_BEGIN;
  if (operand1.dim() == 0) {
    operand1.unsafeGetTensorImpl()->set_sizes_and_strides({1}, {1});
  }
  if (operand2.dim() == 0) {
    operand2.unsafeGetTensorImpl()->set_sizes_and_strides({1}, {1});
  }

  auto operand1_hpu = get_hpu_tensor(operand1);
  auto operand2_hpu = get_hpu_tensor(operand2);
  auto alpha_hpu = get_hpu_tensor(alpha);
  std::vector<const at::Tensor*> pt_inputs;
  pt_inputs.push_back(&operand1_hpu);
  pt_inputs.push_back(&operand2_hpu);
  pt_inputs.push_back(&alpha_hpu);

  size_t device_id = pt_inputs[0]->device().index();
  at::ScalarType scalar_type = pt_inputs[0]->scalar_type();

  std::string node_type =
      node_guid + "_fwd_" + habana_helpers::name_suffix_from_type(scalar_type);

  BinaryOp op(device_id, scalar_type);
  auto out = binary_op_hpu(pt_inputs, node_type, device_id, &op);
  PT_KERNEL_END;
  return out;
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
    // TODO: Optimize for alpha == 1
    auto alpha_tensor = convert_scalar_to_tensor_using_self(other, alpha);
    output = process_generic_tensor_binary_op<habana::AddOperator>(
        self, other, alpha_tensor, "add");
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
  auto out_mul = do_scalar_scalar_mul(self, other, alpha);
  auto output = do_generic_tensor_binary_op(
      self, out_mul, "add", SynapsePassType::FORWARD_PASS);

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

/***************************************************************************
 * @brief Kernel implementation for out = self.addcmul(tensor1, tensor2,alpha)
 * out = self + value*tensor1*tensor2
 * @param other [in] - Scalar
 * @param self [in,out]- Tensor 1D bf16/FP32
 ****************************************************************************/
Tensor& addcmul_hpu_(
    Tensor& self,
    const Tensor& tensor1,
    const Tensor& tensor2,
    Scalar alpha) {
  PT_KERNEL_BEGIN;
  auto prod = at::mul(tensor1, tensor2);
  self.add_(prod, alpha);
  PT_KERNEL_END;
  return self;
}

/*************************************************************************
 * @brief Kernel implementation for torch.addcdiv_(self,tensor1,tensor2,alpha)
 * @param [in] self - input tensor, 1-4D, FP32/BF16
 * @param [in] tensor1 - input tensor, 1-4D, FP32/BF16
 * @param [in] tensor2 - input tensor, 1-4D, FP32/BF16
 * @param [in] alpha - optional input, default = 1
 ************************************************************************/
Tensor addcdiv_hpu(
    Tensor& self,
    const Tensor& tensor1,
    const Tensor& tensor2,
    Scalar alpha) {
  PT_KERNEL_BEGIN;
  auto output_div = at::div(tensor1, tensor2);
  auto output = at::add(self, output_div, alpha);
  PT_KERNEL_END;
  return output;
}

/*************************************************************************
 * @brief Kernel implementation for inplace
 *torch.addcdiv_(self,tensor1,tensor2,alpha)
 * @param [in] self - input tensor, 1-4D, FP32/BF16
 * @param [in] tensor1 - input tensor, 1-4D, FP32/BF16
 * @param [in] tensor2 - input tensor, 1-4D, FP32/BF16
 * @param [in] alpha - optional input, default = 1
 ************************************************************************/
Tensor& addcdiv_hpu_(
    Tensor& self,
    const Tensor& tensor1,
    const Tensor& tensor2,
    Scalar alpha) {
  PT_KERNEL_BEGIN;
  tensor1.div_(tensor2);
  self.add_(tensor1, alpha);
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

  // TODO: Optimize for alpha == 1
  auto alpha_tensor = convert_scalar_to_tensor_using_self(other, alpha);
  auto output = process_generic_tensor_binary_op<habana::SubOperator>(
      self, other, alpha_tensor, "sub");

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
  auto out_mul = do_scalar_scalar_mul(self, other, alpha);
  auto out = do_generic_tensor_binary_op(
      self, out_mul, "sub", SynapsePassType::FORWARD_PASS);
  PT_KERNEL_END;
  return out;
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

  auto other_tensor = convert_scalar_to_tensor_using_self(self, other);
  auto out = at::sub(other_tensor, self, alpha);

  PT_KERNEL_END;
  return out;
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
    return self.pow_(2.0);
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
    return at::pow(self, 2.0);
  }

  auto output = process_generic_tensor_binary_op<habana::MulOperator>(
      self, other, "mult");

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
  auto out = do_tensor_scalar_mul(self, other);
  PT_KERNEL_END;
  return out;
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
 * @brief Kernel implementation for torch.eq(self,other, out)
 * @param self - first input
 * @param other - second input
 * @param out -  output tensor of bool dtype
 ************************************************************************/
void eq_tensor_out_hpu(
    Tensor& output,
    const Tensor& self,
    const Tensor& other) {
  PT_KERNEL_BEGIN;
  // change dtype bool to int8 to match TPC kernel signature
  // NOTE: This works because both bool and int8 uses 1 byte per element
  // Else we need to overload .to operator with an explicit TPC kernel for
  // typecasting
  output.to(c10::ScalarType::Char);
  do_generic_tensor_binary_op_out(
      output, self, other, "equal", SynapsePassType::FORWARD_PASS);
  // convert back to bool
  output.to(c10::ScalarType::Bool);
  PT_KERNEL_END;
}

/*************************************************************************
 * @brief Kernel implementation for out = torch.eq(self,other)
 * @param self - first input
 * @param other - second input
 ************************************************************************/
Tensor eq_tensor_hpu(Tensor& self, Tensor& other) {
  PT_KERNEL_BEGIN;
  auto tensor_options = self.options();
  auto output =
      at::empty(self.sizes(), tensor_options.dtype(c10::ScalarType::Char));
  at::eq_out(output, self, other);
  PT_KERNEL_END;
  return output;
}

/*************************************************************************
 * @brief Kernel implementation for out = torch.eq(self,other)
 * @param self [in] - input tensor, 1-4D, FP32/BF16
 * @param other [in] - Scalar
 ************************************************************************/
Tensor eq_scalar_tensor_hpu(Tensor& self, Scalar other) {
  PT_KERNEL_BEGIN;

  if (self.dim() == 0) {
    self.unsafeGetTensorImpl()->set_sizes_and_strides({1}, {1});
  }

  auto device_tensor = convert_scalar_to_tensor_using_self(self, other);
  auto out = at::eq(self, device_tensor);

  PT_KERNEL_END;
  return out;
}

/*************************************************************************
 * @brief Kernel implementation for out = torch.div(self,other)
 * @param self - first input
 * @param other - second input
 ************************************************************************/
Tensor div_tensor_hpu(const Tensor& self, const Tensor& other) {
  PT_KERNEL_BEGIN;
  if (other.dim() == 0) {
    other.unsafeGetTensorImpl()->set_sizes_and_strides({1}, {1});
  }

  auto out = process_generic_tensor_binary_op<habana::DivOperator>(
      self, other, "div");
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
  auto out = at::div(self, convert_scalar_to_tensor_using_self(self, other));
  PT_KERNEL_END;
  return out;
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
  auto out = do_generic_tensor_binary_op(
      self, other, "pow", SynapsePassType::FORWARD_PASS);
  PT_KERNEL_END;
  return out;
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
  auto exponent_tensor = convert_scalar_to_tensor_using_self(self, other);
  auto out = at::pow(self, exponent_tensor);
  PT_KERNEL_END;
  return out;
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
  auto base_tensor = convert_scalar_to_tensor_using_self(self, other);
  auto out = at::pow(base_tensor, self);
  PT_KERNEL_END;
  return out;
}

static auto registry =
    torch::RegisterOperators()
        .op(torch::RegisterOperators::options()
                .schema(
                    "aten::add_.Tensor(Tensor(a!) self, Tensor other, *, Scalar alpha=1) -> Tensor(a!)")
                .impl_unboxedOnlyKernel<
                    decltype(add_tensor_hpu_),
                    &add_tensor_hpu_>(DispatchKey::HABANATensorId)
                .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
        .op(torch::RegisterOperators::options()
                .schema(
                    "aten::add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor")
                .impl_unboxedOnlyKernel<
                    decltype(add_tensor_hpu),
                    &add_tensor_hpu>(DispatchKey::HABANATensorId)
                .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
        .op(torch::RegisterOperators::options()
                .schema(
                    "aten::add.Scalar(Tensor self, Scalar other, Scalar alpha = 1) -> Tensor")
                .impl_unboxedOnlyKernel<
                    decltype(add_scalar_hpu),
                    &add_scalar_hpu>(DispatchKey::HABANATensorId)
                .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
        .op(torch::RegisterOperators::options()
                .schema(
                    "aten::add_.Scalar(Tensor(a!) self, Scalar other, Scalar alpha=1) -> Tensor(a!)")
                .impl_unboxedOnlyKernel<
                    decltype(add_scalar_hpu_),
                    &add_scalar_hpu_>(DispatchKey::HABANATensorId)
                .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
        .op(torch::RegisterOperators::options()
                .schema(
                    "aten::addcmul_(Tensor(a !) self, Tensor tensor1, Tensor tensor2, *, Scalar value = 1) -> Tensor(a !)")
                .impl_unboxedOnlyKernel<decltype(addcmul_hpu_), &addcmul_hpu_>(
                    DispatchKey::HABANATensorId)
                .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
        .op(torch::RegisterOperators::options()
                .schema(
                    "aten::addcdiv(Tensor self, Tensor tensor1, Tensor tensor2, *, Scalar value=1) -> Tensor")
                .impl_unboxedOnlyKernel<decltype(addcdiv_hpu), &addcdiv_hpu>(
                    DispatchKey::HABANATensorId)
                .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
        .op(torch::RegisterOperators::options()
                .schema(
                    "aten::addcdiv_(Tensor(a!) self, Tensor tensor1, Tensor tensor2, *, Scalar value=1) -> Tensor(a!)")
                .impl_unboxedOnlyKernel<decltype(addcdiv_hpu_), &addcdiv_hpu_>(
                    DispatchKey::HABANATensorId)
                .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
        .op(torch::RegisterOperators::options()
                .schema(
                    "aten::sub.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor")
                .impl_unboxedOnlyKernel<
                    decltype(sub_tensor_hpu),
                    &sub_tensor_hpu>(DispatchKey::HABANATensorId)
                .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
        .op(torch::RegisterOperators::options()
                .schema(
                    "aten::sub_.Tensor(Tensor(a!) self, Tensor other, *, Scalar alpha=1) -> Tensor(a!)")
                .impl_unboxedOnlyKernel<
                    decltype(sub_tensor_hpu_),
                    &sub_tensor_hpu_>(DispatchKey::HABANATensorId)
                .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
        .op(torch::RegisterOperators::options()
                .schema(
                    "aten::sub.Scalar(Tensor self, Scalar other, Scalar alpha=1) -> Tensor")
                .impl_unboxedOnlyKernel<
                    decltype(sub_scalar_hpu),
                    &sub_scalar_hpu>(DispatchKey::HABANATensorId)
                .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
        .op(torch::RegisterOperators::options()
                .schema(
                    "aten::sub_.Scalar(Tensor(a!) self, Scalar other, Scalar alpha=1) -> Tensor(a!)")
                .impl_unboxedOnlyKernel<
                    decltype(sub_scalar_hpu_),
                    &sub_scalar_hpu_>(DispatchKey::HABANATensorId)
                .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
        .op(torch::RegisterOperators::options()
                .schema(
                    "aten::mul_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)")
                .impl_unboxedOnlyKernel<
                    decltype(mul_tensor_hpu_),
                    &mul_tensor_hpu_>(DispatchKey::HABANATensorId)
                .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
        .op(torch::RegisterOperators::options()
                .schema("aten::mul.Tensor(Tensor self, Tensor other) -> Tensor")
                .impl_unboxedOnlyKernel<
                    decltype(mul_tensor_hpu),
                    &mul_tensor_hpu>(DispatchKey::HABANATensorId)
                .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
        .op(torch::RegisterOperators::options()
                .schema("aten::mul.Scalar(Tensor self, Scalar other) -> Tensor")
                .impl_unboxedOnlyKernel<
                    decltype(mul_scalar_hpu),
                    &mul_scalar_hpu>(DispatchKey::HABANATensorId)
                .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
        .op(torch::RegisterOperators::options()
                .schema(
                    "aten::mul_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)")
                .impl_unboxedOnlyKernel<
                    decltype(mul_scalar_hpu_),
                    &mul_scalar_hpu_>(DispatchKey::HABANATensorId)
                .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
        .op(torch::RegisterOperators::options()
                .schema("aten::eq.Tensor(Tensor self, Tensor other) -> Tensor")
                .impl_unboxedOnlyKernel<
                    decltype(eq_tensor_hpu),
                    &eq_tensor_hpu>(DispatchKey::HABANATensorId)
                .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
        .op(torch::RegisterOperators::options()
                .schema(
                    "aten::eq.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)")
                .impl_unboxedOnlyKernel<
                    decltype(eq_tensor_out_hpu),
                    &eq_tensor_out_hpu>(DispatchKey::HABANATensorId)
                .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
        .op(torch::RegisterOperators::options()
                .schema("aten::eq.Scalar(Tensor self, Scalar other) -> Tensor")
                .impl_unboxedOnlyKernel<
                    decltype(eq_scalar_tensor_hpu),
                    &eq_scalar_tensor_hpu>(DispatchKey::HABANATensorId)
                .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
        .op(torch::RegisterOperators::options()
                .schema("aten::div.Tensor(Tensor self, Tensor other) -> Tensor")
                .impl_unboxedOnlyKernel<
                    decltype(div_tensor_hpu),
                    &div_tensor_hpu>(DispatchKey::HABANATensorId)
                .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
        .op(torch::RegisterOperators::options()
                .schema(
                    "aten::div.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)")
                .impl_unboxedOnlyKernel<
                    decltype(div_tensor_hpu_out),
                    &div_tensor_hpu_out>(DispatchKey::HABANATensorId)
                .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
        .op(torch::RegisterOperators::options()
                .schema(
                    "aten::div_.Tensor(Tensor(a!) self, Tensor other) -> (Tensor(a!))")
                .impl_unboxedOnlyKernel<
                    decltype(div_tensor_hpu_),
                    &div_tensor_hpu_>(DispatchKey::HABANATensorId)
                .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
        .op(torch::RegisterOperators::options()
                .schema("aten::div.Scalar(Tensor self, Scalar other) -> Tensor")
                .impl_unboxedOnlyKernel<
                    decltype(div_scalar_hpu),
                    &div_scalar_hpu>(DispatchKey::HABANATensorId)
                .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
        .op(torch::RegisterOperators::options()
                .schema(
                    "aten::div_.Scalar(Tensor(a!) self, Scalar other) -> (Tensor(a!))")
                .impl_unboxedOnlyKernel<
                    decltype(div_scalar_hpu_),
                    &div_scalar_hpu_>(DispatchKey::HABANATensorId)
                .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
        .op(torch::RegisterOperators::options()
                .schema(
                    "aten::pow.Tensor_Tensor(Tensor self, Tensor exponent) -> Tensor")
                .impl_unboxedOnlyKernel<
                    decltype(pow_tensor_tensor_hpu),
                    &pow_tensor_tensor_hpu>(DispatchKey::HABANATensorId)
                .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
        .op(torch::RegisterOperators::options()
                .schema(
                    "aten::pow_.Tensor(Tensor(a!) self, Tensor exponent) -> Tensor(a!)")
                .impl_unboxedOnlyKernel<
                    decltype(pow_tensor_tensor_hpu_),
                    &pow_tensor_tensor_hpu_>(DispatchKey::HABANATensorId)
                .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
        .op(torch::RegisterOperators::options()
                .schema(
                    "aten::pow.Tensor_Scalar(Tensor self, Scalar exponent) -> Tensor")
                .impl_unboxedOnlyKernel<
                    decltype(pow_tensor_scalar_hpu),
                    &pow_tensor_scalar_hpu>(DispatchKey::HABANATensorId)
                .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
        .op(torch::RegisterOperators::options()
                .schema(
                    "aten::pow_.Scalar(Tensor(a !) self, Scalar exponent) -> Tensor(a !)")
                .impl_unboxedOnlyKernel<
                    decltype(pow_tensor_scalar_hpu_),
                    &pow_tensor_scalar_hpu_>(DispatchKey::HABANATensorId)
                .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
        .op(torch::RegisterOperators::options()
                .schema(
                    "aten::pow.Scalar(Scalar self, Tensor exponent)->Tensor")
                .impl_unboxedOnlyKernel<
                    decltype(pow_scalar_tensor_hpu),
                    &pow_scalar_tensor_hpu>(DispatchKey::HABANATensorId)
                .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
        .op(torch::RegisterOperators::options()
                .schema(
                    "aten::rsub.Scalar(Tensor self, Scalar other, Scalar alpha=1) -> Tensor")
                .impl_unboxedOnlyKernel<
                    decltype(rsub_scalar_hpu),
                    &rsub_scalar_hpu>(DispatchKey::HABANATensorId)
                .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA));
