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
#include "habana_helpers/tensor_utils.h"
#include "habana_kernels/basic_kernels.h"
#include "habana_kernels/binary_inplace_kernels.h"
#include "habana_kernels/binary_kernels.h"
#include "habana_kernels/kernel_utils.h"
#include "habana_kernels/simple_generic_kernel.h"
#include "habana_kernels/tensor_shape_kernels.h"

using namespace torch;

/************************************************************************
 * @brief This function implements synapse node addition for
 * binary operators where 2 inputs are tensors & 3rd input is a scalar.
 * Mismatch in input tensor dims is taken care of using reshape nodes,
 * whereas scalar to tensor conversion is done using constant node.
 ************************************************************************/
void habana::BinaryInplaceOperatorWithAlpha::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    const habana::OutputMetaDataVector& output_metadata) {
  static_cast<void>(output_metadata);
  TORCH_CHECK(
      inputs.size() == 3, "Incorrect size of input expected for add operator");
  TORCH_CHECK(inputs[0].isTensor(), "Input 0 type expected to be tensor");
  TORCH_CHECK(inputs[1].isTensor(), "Input 1 type expected to be tensor");
  Tensor arg1 = inputs[0].toTensor();
  Tensor arg2 = inputs[1].toTensor();

  if (inputs[2].toScalar().toFloat() != 1.0) {
    // Multiplication between arg2 and alpha is required
    auto mulOp = make_operator<habana::MulOperator>(
        this->p_context_->device_id_, this->scalarType_);
    mulOp->SetSynapseInput(p_context_->syn_inputs_[1]);
    torch::jit::Stack mulOp_stack{inputs[1], inputs[2]};

    mulOp->AllocateAndAddSynapseNode(
        graph, mulOp_stack, habana::OutputMetaDataVector(1));
    synapse_helpers::tensor_or_ref& mulOp_out = mulOp->GetSynOutputs()[0];

    // Note here we are using input[0] to store output[0]
    p_context_->syn_outputs_.emplace_back(
        habana_helpers::duplicate_tensor_in_memory_section(
            p_context_->syn_inputs_[0], graph, output_metadata.at(0).external));
    p_context_->pt_outputs_.emplace_back(arg1);

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
    // Note here we are using input[0] to store output[0]
    p_context_->syn_outputs_.emplace_back(
        habana_helpers::duplicate_tensor_in_memory_section(
            p_context_->syn_inputs_[0], graph, output_metadata.at(0).external));
    p_context_->pt_outputs_.emplace_back(arg1);
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
void habana::BinaryInplaceWrapperOperatorWithAlpha::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    const habana::OutputMetaDataVector& output_metadata) {
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

  auto binaryOp = make_operator<BinaryInplaceOperatorWithAlpha>(
      this->p_context_->device_id_, guid_, this->scalarType_);

  if (inputs[0].isTensor() &&
      inputs[1].isTensor()) { // First 2 inputs are both tensors
    binaryOp->SetSynapseInput(p_context_->syn_inputs_[0]);
    binaryOp->SetSynapseInput(p_context_->syn_inputs_[1]);
    binaryOp->AllocateAndAddSynapseNode(graph, inputs, output_metadata);

  } else if (inputs[0].isTensor() && inputs[1].isScalar()) { // 2nd input is a
                                                             // scalar
    auto arg1 = inputs[0].toTensor();
    // add node to convert scalar to tensor
    auto constOp = make_operator<ConstantOperator>(
        this->p_context_->device_id_, this->scalarType_);
    auto const_shape_tensor = habana_helpers::createPTTensor(
        arg1, {1}, arg1.options(), at::MemoryFormat::Contiguous, false);
    torch::jit::Stack constOp_stack = {IValue(const_shape_tensor), inputs[1]};
    constOp->AllocateAndAddSynapseNode(
        graph, constOp_stack, habana::OutputMetaDataVector(1));
    binaryOp->SetSynapseInput(p_context_->syn_inputs_[0]);
    binaryOp->SetSynapseInput(constOp->GetSynOutputs()[0]);
    // replace 2nd scalar input with a tensor in stack
    inputs.erase(inputs.cbegin() + 1);
    inputs.emplace(inputs.cbegin() + 1, constOp->GetOutputs()[0]);
    binaryOp->AllocateAndAddSynapseNode(graph, inputs, output_metadata);
  }

  p_context_->pt_outputs_.emplace_back(binaryOp->GetOutputs()[0]);
  p_context_->syn_outputs_.emplace_back(
      std::move(binaryOp->GetSynOutputs()[0]));
}

/************************************************************************
 * @brief This function implements synapse node addition for
 * inplace binary operators where both inputs are tensors. Mismatch in
 * input tensor dims is also taken care of using reshape nodes.
 ************************************************************************/
void habana::BinaryInplaceOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    const habana::OutputMetaDataVector& output_metadata) {
  // this check is for stack during graph execution
  static_cast<void>(output_metadata);
  TORCH_CHECK(
      inputs.size() == 2,
      "Incorrect size of input expected for Binary operator");
  TORCH_CHECK(inputs[0].isTensor(), "Input type expected to be tensor");
  TORCH_CHECK(inputs[1].isTensor(), "Input type expected to be tensor");
  Tensor arg1 = inputs[0].toTensor();
  Tensor arg2 = inputs[1].toTensor();

  // Note here we are using input[0] to store output[0]
  p_context_->syn_outputs_.emplace_back(
      habana_helpers::duplicate_tensor_in_memory_section(
          p_context_->syn_inputs_[0], graph, output_metadata.at(0).external));
  p_context_->pt_outputs_.emplace_back(arg1);

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

/************************************************************************
 * @brief This function implements synapse node addition for inplace
 * Binary OPs with 2 input arguments (where 1st input is always a
 * tensor whereas 2nd input can be a tensor or a scalar)
 ************************************************************************/
void habana::BinaryInplaceWrapperOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    const habana::OutputMetaDataVector& output_metadata) {
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

  auto binaryInplaceOp = make_operator<BinaryInplaceOperator>(
      this->p_context_->device_id_, guid_, this->scalarType_);

  if (inputs[0].isTensor() && inputs[1].isTensor()) { // Both inputs are tensors
    binaryInplaceOp->SetSynapseInput(p_context_->syn_inputs_[0]);
    binaryInplaceOp->SetSynapseInput(p_context_->syn_inputs_[1]);
    binaryInplaceOp->AllocateAndAddSynapseNode(graph, inputs, output_metadata);
  } else if (inputs[0].isTensor() && inputs[1].isScalar()) { // 2nd input is a
                                                             // scalar
    auto arg1 = inputs[0].toTensor();
    // add constant node to convert 2nd input to tensor
    auto constOp = make_operator<ConstantOperator>(
        this->p_context_->device_id_, this->scalarType_);
    auto const_shape_tensor = habana_helpers::createPTTensor(
        arg1, {1}, arg1.options(), at::MemoryFormat::Contiguous, false);
    torch::jit::Stack constOp_stack = {IValue(const_shape_tensor), inputs[1]};
    constOp->AllocateAndAddSynapseNode(
        graph, constOp_stack, habana::OutputMetaDataVector(1));
    binaryInplaceOp->SetSynapseInput(p_context_->syn_inputs_[0]);
    binaryInplaceOp->SetSynapseInput(constOp->GetSynOutputs()[0]);
    // replace input scalar with input tensor in the stack
    inputs.pop_back();
    inputs.emplace_back(constOp->GetOutputs()[0]);
    binaryInplaceOp->AllocateAndAddSynapseNode(graph, inputs, output_metadata);
  }

  p_context_->pt_outputs_.emplace_back(binaryInplaceOp->GetOutputs()[0]);
  p_context_->syn_outputs_.emplace_back(
      std::move(binaryInplaceOp->GetSynOutputs()[0]));
}

/************************************************************************
 * @brief Generic wrapper for all Eager mode Binary OP invocations that
 * are inplace
 ************************************************************************/
template <class BinaryInplaceOp>
void process_generic_tensor_inplace_binary_op(
    const std::vector<at::Tensor>& pt_inputs,
    torch::jit::Stack& stack,
    const std::string& node_guid) {
  size_t device_id = pt_inputs[0].device().index();
  at::ScalarType scalar_type = pt_inputs[0].scalar_type();
  std::string node_type =
      node_guid + "_fwd_" + habana_helpers::name_suffix_from_type(scalar_type);

  auto& device = synapse_helpers::HPURegistrar::get_device(device_id);
  BinaryInplaceOp Op(device_id, scalar_type);

  size_t key = Op.GetRecipeKey(node_type, stack, true);

  if (device.get_recipe_handle_cache().isCached(key)) {
    auto patch_output = pt_inputs[0];
    Op.Execute(key, pt_inputs, patch_output);
  } else {
    // both inputs are not required, just to match graph mode stack
    habana::OutputMetaDataVector output_metadata(1);
    output_metadata.at(0).persistent = true;
    Op.CreateGraphAndCompile(key, pt_inputs, stack, output_metadata, true);
  }
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
  // TODO: Add pow operator for graph mode
  if (self.is_same(other)) {
    auto& tensor = self.pow_(2.0);
    PT_KERNEL_END
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
  process_generic_tensor_inplace_binary_op<habana::MulInplaceOperator>(
      pt_inputs, stack, "mult");
  PT_KERNEL_END;
  return self;
}

/*************************************************************************
 * @brief Kernel implementation for inplace output = self.mul_(Scalar other)
 * @param self - first input
 * @param other - second input
 * self = self * other
 ************************************************************************/
Tensor& mul_scalar_hpu_(Tensor& self, const Scalar& other) {
  PT_KERNEL_BEGIN;
  if (self.dim() == 0) {
    self.unsafeGetTensorImpl()->set_sizes_and_strides({1}, {1});
  }
  auto self_hpu = get_hpu_tensor(self);
  std::vector<at::Tensor> pt_inputs{self_hpu};
  torch::jit::Stack stack{IValue(self_hpu), IValue(other)};
  process_generic_tensor_inplace_binary_op<habana::MulInplaceOperator>(
      pt_inputs, stack, "mult");

  PT_KERNEL_END;
  return self;
}

/*************************************************************************
 * @brief Kernel implementation for inplace torch.div_(self,other)
 * @param self - first input
 * @param other - second input
 ************************************************************************/
Tensor& div_tensor_hpu_(Tensor& self, const Tensor& other) {
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
  process_generic_tensor_inplace_binary_op<habana::DivInplaceOperator>(
      pt_inputs, stack, "div");
  PT_KERNEL_END;
  return self;
}

/*************************************************************************
 * @brief Kernel implementation for inplace div_.Scalar(self,other)
 * @param self - first input
 * @param other - second input of scalar type
 ************************************************************************/
Tensor& div_scalar_hpu_(
    Tensor& self,
    const Scalar& other) { // TODO: Add test by using an extension module for new op
                    // at python level
  PT_KERNEL_BEGIN;
  if (self.dim() == 0) {
    self.unsafeGetTensorImpl()->set_sizes_and_strides({1}, {1});
  }
  auto self_hpu = get_hpu_tensor(self);
  std::vector<at::Tensor> pt_inputs{self_hpu};
  torch::jit::Stack stack{IValue(self_hpu), IValue(other)};
  process_generic_tensor_inplace_binary_op<habana::DivInplaceOperator>(
      pt_inputs, stack, "div");

  PT_KERNEL_END;
  return self;
}

/*************************************************************************
 * @brief Kernel implementation for out = self.pow(other)
 * @param self - first input
 * @param other - second input
 ************************************************************************/
Tensor& pow_tensor_tensor_hpu_(Tensor& self, const Tensor& other) {
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
  process_generic_tensor_inplace_binary_op<habana::PowInplaceOperator>(
      pt_inputs, stack, "pow");
  PT_KERNEL_END;
  return self;
}

Tensor& pow_tensor_scalar_hpu_(Tensor& self, Scalar other) {
  PT_KERNEL_BEGIN;
  if (self.dim() == 0) {
    self.unsafeGetTensorImpl()->set_sizes_and_strides({1}, {1});
  }
  auto self_hpu = get_hpu_tensor(self);
  std::vector<at::Tensor> pt_inputs{self_hpu};
  torch::jit::Stack stack{IValue(self_hpu), IValue(other)};
  process_generic_tensor_inplace_binary_op<habana::PowInplaceOperator>(
      pt_inputs, stack, "pow");

  PT_KERNEL_END;
  return self;
}

// self += alpha * other
Tensor& add_tensor_hpu_(Tensor& self, const Tensor& other, Scalar alpha) {
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
  process_generic_tensor_inplace_binary_op<habana::AddInplaceOperator>(
      pt_inputs, stack, "add");
  PT_KERNEL_END;
  return self;
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
    Scalar other,
    Scalar alpha) { // TODO: Add test by using an extension module for new op
                    // at python level
  PT_KERNEL_BEGIN;
  if (self.dim() == 0) {
    self.unsafeGetTensorImpl()->set_sizes_and_strides({1}, {1});
  }
  auto self_hpu = get_hpu_tensor(self);
  std::vector<at::Tensor> pt_inputs{self_hpu};
  torch::jit::Stack stack{IValue(self_hpu), IValue(other), IValue(alpha)};
  process_generic_tensor_inplace_binary_op<habana::AddInplaceOperator>(
      pt_inputs, stack, "add");
  PT_KERNEL_END;
  return self;
}

/*************************************************************************
 * @brief Kernel implementation for inplace torch.sub_(self, alpha, other)
 * @param self - first input
 * @param other - second input
 * @param alpha - optional input
 * self -= alpha * other
 ************************************************************************/
Tensor& sub_tensor_hpu_(Tensor& self, const Tensor& other, const Scalar& alpha) {
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
  process_generic_tensor_inplace_binary_op<habana::SubInplaceOperator>(
      pt_inputs, stack, "sub");
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
Tensor& sub_scalar_hpu_(
    Tensor& self,
    const Scalar& other,
    const Scalar& alpha) { // TODO: No way to test this yet from python
  PT_KERNEL_BEGIN;
  if (self.dim() == 0) {
    self.unsafeGetTensorImpl()->set_sizes_and_strides({1}, {1});
  }
  auto self_hpu = get_hpu_tensor(self);
  std::vector<at::Tensor> pt_inputs{self_hpu};
  torch::jit::Stack stack{IValue(self_hpu), IValue(other), IValue(alpha)};
  process_generic_tensor_inplace_binary_op<habana::SubInplaceOperator>(
      pt_inputs, stack, "sub");
  PT_KERNEL_END;
  return self;
}

void habana::AddcmulInplaceOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    const habana::OutputMetaDataVector& output_metadata) {
  TORCH_CHECK(
      inputs.size() == 4,
      "Incorrect size of inputs expected for Addcmul operator");
  TORCH_CHECK(
      inputs[0].isTensor(),
      "Input arg1 expected to be tensor for Addcmul operator");
  TORCH_CHECK(
      inputs[1].isTensor(),
      "Input arg2 expected to be tensor for Addcmul operator");
  TORCH_CHECK(
      inputs[2].isTensor(),
      "Input arg3 expected to be tensor for Addcmul operator");
  TORCH_CHECK(
      inputs[3].isScalar(),
      "Input arg4 expected to be Scalar for Addcmul operator");

  auto self = inputs[0].toTensor();
  auto tensor1 = inputs[1].toTensor();
  auto tensor2 = inputs[2].toTensor();
  auto alphaValue = inputs[3].toScalar();

  std::vector<c10::IValue> stack;
  at::ScalarType scalar_type = self.scalar_type();

  // special handling required. synapse cannot handle same tensor given as both
  // inputs to a binary op
  if (tensor1.is_same(tensor2)) {
    // x^2 implemented as x*x. Identity node used to create aliased tensor
    // since GC/TPC does not like giving same tensor as both inputs to a
    // binary op
    auto identityOp = make_operator<IdentityOperator>(
        this->p_context_->device_id_, scalar_type);
    identityOp->SetSynapseInput(p_context_->syn_inputs_[1]);
    torch::jit::Stack stack = {IValue(tensor1)};
    identityOp->AllocateAndAddSynapseNode(
        graph, stack, habana::OutputMetaDataVector(1));
    stack.clear();

    auto mulOp = make_operator<habana::MulOperator>(
        this->p_context_->device_id_, scalar_type);
    mulOp->SetSynapseInput(p_context_->syn_inputs_[1]);
    mulOp->SetSynapseInput(identityOp->GetSynOutputs()[0]);
    stack.emplace_back(IValue(tensor1));
    stack.emplace_back(IValue(identityOp->GetOutputs()[0]));
    mulOp->AllocateAndAddSynapseNode(
        graph, stack, habana::OutputMetaDataVector(1));
    stack.clear();

    // Create Add operator
    auto addOp = make_operator<habana::AddInplaceOperator>(
        this->p_context_->device_id_, scalar_type);
    addOp->SetSynapseInput(p_context_->syn_inputs_[0]);
    addOp->SetSynapseInput(mulOp->GetSynOutputs()[0]);
    stack.emplace_back(IValue(self));
    stack.emplace_back(IValue(mulOp->GetOutputs()[0]));
    stack.emplace_back(IValue(alphaValue));
    addOp->AllocateAndAddSynapseNode(graph, stack, output_metadata);
    stack.clear();

    p_context_->syn_outputs_.emplace_back(std::move(addOp->GetSynOutputs()[0]));
    p_context_->pt_outputs_.emplace_back(std::move(addOp->GetOutputs()[0]));
  } else {
    // Create Mul operator
    auto mulOp = make_operator<habana::MulOperator>(
        this->p_context_->device_id_, scalar_type);
    mulOp->SetSynapseInput(p_context_->syn_inputs_[1]);
    mulOp->SetSynapseInput(p_context_->syn_inputs_[2]);
    stack.emplace_back(IValue(tensor1));
    stack.emplace_back(IValue(tensor2));
    mulOp->AllocateAndAddSynapseNode(
        graph, stack, habana::OutputMetaDataVector(1));
    stack.clear();

    // Create Add operator
    auto addOp = make_operator<habana::AddInplaceOperator>(
        this->p_context_->device_id_, scalar_type);
    addOp->SetSynapseInput(p_context_->syn_inputs_[0]);
    addOp->SetSynapseInput(mulOp->GetSynOutputs()[0]);
    stack.emplace_back(IValue(self));
    stack.emplace_back(IValue(mulOp->GetOutputs()[0]));
    stack.emplace_back(IValue(alphaValue));
    addOp->AllocateAndAddSynapseNode(graph, stack, output_metadata);
    stack.clear();

    p_context_->syn_outputs_.emplace_back(std::move(addOp->GetSynOutputs()[0]));
    p_context_->pt_outputs_.emplace_back(std::move(addOp->GetOutputs()[0]));
  }
}

Tensor& addcmul_hpu_(
    Tensor& self,
    const Tensor& tensor1,
    const Tensor& tensor2,
    Scalar alpha) {
  PT_KERNEL_BEGIN;

  at::ScalarType scalar_type = self.scalar_type();
  std::string node_type =
      "addcmul_fwd_" + habana_helpers::name_suffix_from_type(scalar_type);

  size_t device_id = self.device().index();
  auto& device = synapse_helpers::HPURegistrar::get_device(device_id);

  // create the operator
  habana::AddcmulInplaceOperator Op(device_id, scalar_type);

  // Build Params for the graph
  std::vector<c10::IValue> stack = {
      IValue(self), IValue(tensor1), IValue(tensor2), IValue(alpha)};
  size_t key = Op.GetRecipeKey(node_type, stack, true);

  // Assign Inputs to the Operator
  std::vector<at::Tensor> pt_inputs{self, tensor1, tensor2};

  if (device.get_recipe_handle_cache().isCached(key)) {
    std::vector<at::Tensor> v{self};
    Op.SetPTOutputs(v);
    Op.Execute(key, pt_inputs, v);
  } else {
    habana::OutputMetaDataVector output_metadata(1);
    output_metadata.at(0).persistent = true;
    // Create, compile and execute the graph
    Op.CreateGraphAndCompile(key, pt_inputs, stack, output_metadata, true);
  }

  PT_KERNEL_END;
  return self;
}

/*************************************************************************
 * @brief Kernel implementation for inplace
 *torch.addcmul_(self,tensor1,tensor2,alpha)
 * @param [in] self - input tensor, 1-4D, FP32/BF16
 * @param [in] tensor1 - input tensor, 1-4D, FP32/BF16
 * @param [in] tensor2 - input tensor, 1-4D, FP32/BF16
 * @param [in] alpha - optional input, default = 1
 ************************************************************************/
void habana::AddcdivInplaceOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    const habana::OutputMetaDataVector& output_metadata) {
  TORCH_CHECK(
      inputs.size() == 4,
      "Incorrect size of inputs expected for Addcmul operator");
  TORCH_CHECK(
      inputs[0].isTensor(),
      "Input arg1 expected to be tensor for Addcmul operator");
  TORCH_CHECK(
      inputs[1].isTensor(),
      "Input arg2 expected to be tensor for Addcmul operator");
  TORCH_CHECK(
      inputs[2].isTensor(),
      "Input arg3 expected to be tensor for Addcmul operator");
  TORCH_CHECK(
      inputs[3].isScalar(),
      "Input arg4 expected to be Scalar for Addcmul operator");

  auto self = inputs[0].toTensor();
  auto tensor1 = inputs[1].toTensor();
  auto tensor2 = inputs[2].toTensor();
  auto alphaValue = inputs[3].toScalar();

  std::vector<c10::IValue> stack;
  at::ScalarType scalar_type = self.scalar_type();

  // Create Div operator
  auto divOp = make_operator<habana::DivOperator>(
      this->p_context_->device_id_, scalar_type);
  divOp->SetSynapseInput(p_context_->syn_inputs_[1]);
  divOp->SetSynapseInput(p_context_->syn_inputs_[2]);
  stack.emplace_back(IValue(tensor1));
  stack.emplace_back(IValue(tensor2));
  divOp->AllocateAndAddSynapseNode(
      graph, stack, habana::OutputMetaDataVector(1));
  stack.clear();

  // Create Add operator
  auto addOp = make_operator<habana::AddInplaceOperator>(
      this->p_context_->device_id_, scalar_type);
  addOp->SetSynapseInput(p_context_->syn_inputs_[0]);
  addOp->SetSynapseInput(divOp->GetSynOutputs()[0]);
  stack.emplace_back(IValue(self));
  stack.emplace_back(IValue(divOp->GetOutputs()[0]));
  stack.emplace_back(IValue(alphaValue));
  addOp->AllocateAndAddSynapseNode(graph, stack, output_metadata);
  stack.clear();

  p_context_->syn_outputs_.emplace_back(std::move(addOp->GetSynOutputs()[0]));
  p_context_->pt_outputs_.emplace_back(std::move(addOp->GetOutputs()[0]));
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
    const Scalar& alpha) {
  PT_KERNEL_BEGIN;

  at::ScalarType scalar_type = self.scalar_type();
  std::string node_type =
      "addcdiv_fwd_" + habana_helpers::name_suffix_from_type(scalar_type);

  size_t device_id = self.device().index();
  auto& device = synapse_helpers::HPURegistrar::get_device(device_id);

  // create the operator
  habana::AddcdivInplaceOperator Op(device_id, scalar_type);

  // Build Params for the graph
  std::vector<c10::IValue> stack = {
      IValue(self), IValue(tensor1), IValue(tensor2), IValue(alpha)};
  size_t key;

  // TBD: The following option to not cache the recipe is done
  // till a way to accomodate changing scalar values is found.
  const string cacheScalarEnvValue = "PT_HPU_CACHE_RECIPE_WITH_SCALAR";
  const char* cacheValue = getenv(cacheScalarEnvValue.c_str());
  if (cacheValue && (strncmp(cacheValue, "0", 1) == 0)) {
    PT_DEVICE_DEBUG("Caching disabled for ", node_type);
    key = 0;
  } else {
    key = Op.GetRecipeKey(node_type, stack, true);
  }

  // Assign Inputs to the Operator
  std::vector<at::Tensor> pt_inputs{self, tensor1, tensor2};
  if (device.get_recipe_handle_cache().isCached(key)) {
    std::vector<at::Tensor> v{self};
    Op.Execute(key, pt_inputs, v);
  } else {
    habana::OutputMetaDataVector output_metadata(1);
    output_metadata.at(0).persistent = true;
    // compile and execute the graph
    Op.CreateGraphAndCompile(key, pt_inputs, stack, output_metadata, true);
  }

  PT_KERNEL_END;
  return self;
}

static auto& KernelRegistry =
    habana::KernelRegistry()
        .add("aten::mul_.Tensor", KERNEL_FN(MulInplaceOperator))
        .add("aten::mul_.Scalar", KERNEL_FN(MulInplaceOperator))
        .add("aten::pow_.Tensor", KERNEL_FN(PowInplaceOperator))
        .add("aten::pow_.Scalar", KERNEL_FN(PowInplaceOperator))
        .add("aten::add_.Tensor", KERNEL_FN(AddInplaceOperator))
        .add("aten::add_.Scalar", KERNEL_FN(AddInplaceOperator))
        .add("aten::addcmul_", KERNEL_FN(AddcmulInplaceOperator))
        .add("aten::addcdiv_", KERNEL_FN(AddcdivInplaceOperator))
        .add("aten::div_.Tensor", KERNEL_FN(DivInplaceOperator))
        .add("aten::div_.Scalar", KERNEL_FN(DivInplaceOperator))
        .add("aten::sub_.Tensor", KERNEL_FN(SubInplaceOperator))
        .add("aten::sub_.Scalar", KERNEL_FN(SubInplaceOperator));
