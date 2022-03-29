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
#include <ATen/InferSize.h>
#include <synapse_api.h>
#include <torch/script.h>

#include "habana_device/HPUCheck.h"
#include "habana_device/hpu_cached_devices.h"
#include "habana_helpers/logging.h"
#include "habana_kernels/binary_composite_kernels.h"
#include "habana_kernels/binary_kernels.h"

using namespace torch;
using namespace habana;

std::vector<int64_t> AddcmulOperator::compute_output_shape(
    const Tensor& arg1,
    const Tensor& arg2,
    const Tensor& arg3) {
  auto tmp = at::infer_size(arg1.sizes(), arg2.sizes());
  return {at::infer_size(tmp, arg3.sizes())};
}

/***************************************************************************
 * @brief Kernel implementation for out = self.addcmul(tensor1, tensor2,alpha)
 * out = self + value*tensor1*tensor2
 * @param other [in] - Scalar
 * @param self [in,out]- Tensor 1D bf16/FP32
 ****************************************************************************/
void AddcmulOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    const OutputMetaDataVector& output_metadata) {
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

  // Create Mul operator
  auto mulOp =
      make_operator<MulOperator>(this->p_context_->device_id_, scalar_type);
  mulOp->SetSynapseInput(p_context_->syn_inputs_[1]);
  mulOp->SetSynapseInput(p_context_->syn_inputs_[2]);
  stack.emplace_back(IValue(tensor1));
  stack.emplace_back(IValue(tensor2));
  mulOp->AllocateAndAddSynapseNode(graph, stack, OutputMetaDataVector(1));
  stack.clear();
  // Create Add operator
  auto addOp =
      make_operator<AddOperator>(this->p_context_->device_id_, scalar_type);
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

Tensor addcmul_hpu(
    const Tensor& self,
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
  AddcmulOperator Op(device_id, scalar_type);

  // Build Params for the graph
  std::vector<c10::IValue> stack = {
      IValue(self), IValue(tensor1), IValue(tensor2), IValue(alpha)};
  size_t key = Op.GetRecipeKey(node_type, stack);

  // Assign Inputs to the Operator
  std::vector<at::Tensor> pt_inputs{self, tensor1, tensor2};

  if (device.get_recipe_handle_cache().isCached(key)) {
    auto result =
        at::empty(self.sizes(), self.options(), self.suggest_memory_format());
    Op.Execute(key, pt_inputs, result);
  } else {
    OutputMetaDataVector output_metadata(1);
    output_metadata.at(0).persistent = true;
    Op.CreateGraphAndCompile(key, pt_inputs, stack, output_metadata, true);
  }

  std::vector<at::Tensor> out = Op.GetOutputs();
  TORCH_CHECK(out.size() == 1, "Incorrect size of outputs");

  PT_KERNEL_END;
  return out.at(0);
}

/*************************************************************************
 * @brief Kernel implementation for torch.addcdiv_(self,tensor1,tensor2,alpha)
 * @param [in] self - input tensor, 1-4D, FP32/BF16
 * @param [in] tensor1 - input tensor, 1-4D, FP32/BF16
 * @param [in] tensor2 - input tensor, 1-4D, FP32/BF16
 * @param [in] alpha - optional input, default = 1
 ************************************************************************/
void AddcdivOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    const OutputMetaDataVector& output_metadata) {
  TORCH_CHECK(
      inputs.size() == 4,
      "Incorrect size of inputs expected for Addcdiv operator");
  TORCH_CHECK(
      inputs[0].isTensor(),
      "Input arg1 expected to be tensor for Addcdiv operator");
  TORCH_CHECK(
      inputs[1].isTensor(),
      "Input arg2 expected to be tensor for Addcdiv operator");
  TORCH_CHECK(
      inputs[2].isTensor(),
      "Input arg3 expected to be tensor for Addcdiv operator");
  TORCH_CHECK(
      inputs[3].isScalar(),
      "Input arg4 expected to be Scalar for Addcdiv operator");

  auto self = inputs[0].toTensor();
  auto tensor1 = inputs[1].toTensor();
  auto tensor2 = inputs[2].toTensor();
  auto alphaValue = inputs[3].toScalar();

  std::vector<c10::IValue> stack;
  at::ScalarType scalar_type = self.scalar_type();

  // Create Div operator
  auto divOp =
      make_operator<DivOperator>(this->p_context_->device_id_, scalar_type);
  divOp->SetSynapseInput(p_context_->syn_inputs_[1]);
  divOp->SetSynapseInput(p_context_->syn_inputs_[2]);
  stack.emplace_back(IValue(tensor1));
  stack.emplace_back(IValue(tensor2));
  divOp->AllocateAndAddSynapseNode(graph, stack, OutputMetaDataVector(1));
  stack.clear();

  // Create Add operator
  auto addOp =
      make_operator<AddOperator>(this->p_context_->device_id_, scalar_type);
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

Tensor addcdiv_hpu(
    const Tensor& self,
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
  AddcdivOperator Op(device_id, scalar_type);

  // Build Params for the graph
  std::vector<c10::IValue> stack = {
      IValue(self), IValue(tensor1), IValue(tensor2), IValue(alpha)};
  size_t key = Op.GetRecipeKey(node_type, stack);

  // Assign Inputs to the Operator
  std::vector<at::Tensor> pt_inputs{self, tensor1, tensor2};

  if (device.get_recipe_handle_cache().isCached(key)) {
    auto result =
        at::empty(self.sizes(), self.options(), self.suggest_memory_format());
    Op.Execute(key, pt_inputs, result);
  } else {
    OutputMetaDataVector output_metadata(1);
    output_metadata.at(0).persistent = true;
    Op.CreateGraphAndCompile(key, pt_inputs, stack, output_metadata, true);
  }

  std::vector<at::Tensor> out = Op.GetOutputs();
  TORCH_CHECK(out.size() == 1, "Incorrect size of outputs");

  PT_KERNEL_END;
  return out.at(0);
}

static auto& KernelRegistry =
    habana::KernelRegistry()
        .add("aten::addcmul", KERNEL_FN(AddcmulOperator))
        .add("aten::addcdiv", KERNEL_FN(AddcdivOperator));
