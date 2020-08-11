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
#include "habana_helpers/graph.h"
#include "habana_helpers/logging.h"
#include "habana_kernels/binary_kernels.h"
#include "habana_kernels/binary_composite_kernels.h"

using namespace torch;

/***************************************************************************
 * @brief Kernel implementation for out = self.addcmul(tensor1, tensor2,alpha)
 * out = self + value*tensor1*tensor2
 * @param other [in] - Scalar
 * @param self [in,out]- Tensor 1D bf16/FP32
 ****************************************************************************/
void AddcmulOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    bool is_output_persistent) {
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
  MulOperator mulOp(this->p_context_->device_id_, scalar_type);
  auto& mul_syn_1 = mulOp.SetSynapseInput(std::move(p_context_->syn_inputs_[1]));
  auto& mul_syn_2 = mulOp.SetSynapseInput(std::move(p_context_->syn_inputs_[2]));
  stack.emplace_back(IValue(tensor1));
  stack.emplace_back(IValue(tensor2));
  mulOp.AllocateAndAddSynapseNode(graph, stack, false);
  p_context_->syn_inputs_[1] = std::move(mul_syn_1);
  p_context_->syn_inputs_[2] = std::move(mul_syn_2);
  stack.clear();



  // Create Add operator
  AddOperator addOp(this->p_context_->device_id_, scalar_type);
  auto& add_syn = addOp.SetSynapseInput(std::move(p_context_->syn_inputs_[0]));
  addOp.SetSynapseInput(std::move(mulOp.GetSynOutputs()[0]));
  stack.emplace_back(IValue(self));
  stack.emplace_back(IValue(mulOp.GetOutputs()[0]));
  stack.emplace_back(IValue(alphaValue));
  addOp.AllocateAndAddSynapseNode(graph, stack, is_output_persistent);
  p_context_->syn_inputs_[0] = std::move(add_syn);
  stack.clear();


  p_context_->syn_outputs_.emplace_back(std::move(addOp.GetSynOutputs()[0]));
  p_context_->pt_outputs_.emplace_back(std::move(addOp.GetOutputs()[0]));
}

 Tensor addcmul_hpu(
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
  AddcmulOperator Op(device_id, scalar_type);

  // Build Params for the graph
  std::vector<c10::IValue> stack = {IValue(self),
                                    IValue(tensor1),
                                    IValue(tensor2),
                                    IValue(alpha)};
  size_t key = Op.GetRecipeKey(node_type, stack);

  // Assign Inputs to the Operator
  std::vector<at::Tensor> pt_inputs{self, tensor1, tensor2};

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
void AddcdivOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    bool is_output_persistent) {
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
  DivOperator divOp(this->p_context_->device_id_, scalar_type);
  auto& div_syn_1 = divOp.SetSynapseInput(std::move(p_context_->syn_inputs_[1]));
  auto& div_syn_2 = divOp.SetSynapseInput(std::move(p_context_->syn_inputs_[2]));
  stack.emplace_back(IValue(tensor1));
  stack.emplace_back(IValue(tensor2));
  divOp.AllocateAndAddSynapseNode(graph, stack, false);
  p_context_->syn_inputs_[1] = std::move(div_syn_1);
  p_context_->syn_inputs_[2] = std::move(div_syn_2);
  stack.clear();



  // Create Add operator
  AddOperator addOp(this->p_context_->device_id_, scalar_type);
  auto& add_syn = addOp.SetSynapseInput(std::move(p_context_->syn_inputs_[0]));
  addOp.SetSynapseInput(std::move(divOp.GetSynOutputs()[0]));
  stack.emplace_back(IValue(self));
  stack.emplace_back(IValue(divOp.GetOutputs()[0]));
  stack.emplace_back(IValue(alphaValue));
  addOp.AllocateAndAddSynapseNode(graph, stack, is_output_persistent);
  p_context_->syn_inputs_[0] = std::move(add_syn);
  stack.clear();


  p_context_->syn_outputs_.emplace_back(std::move(addOp.GetSynOutputs()[0]));
  p_context_->pt_outputs_.emplace_back(std::move(addOp.GetOutputs()[0]));
}

Tensor addcdiv_hpu(
    Tensor& self,
    const Tensor& tensor1,
    const Tensor& tensor2,
    Scalar alpha) {
  PT_KERNEL_BEGIN;

  at::ScalarType scalar_type = self.scalar_type();
  std::string node_type =
      "addcdiv_fwd_" + habana_helpers::name_suffix_from_type(scalar_type);

  size_t device_id = self.device().index();
  auto& device = synapse_helpers::HPURegistrar::get_device(device_id);

  // create the operator
  AddcdivOperator Op(device_id, scalar_type);

  // Build Params for the graph
  std::vector<c10::IValue> stack = {IValue(self),
                                    IValue(tensor1),
                                    IValue(tensor2),
                                    IValue(alpha)};
  size_t key = Op.GetRecipeKey(node_type, stack);

  // Assign Inputs to the Operator
  std::vector<at::Tensor> pt_inputs{self, tensor1, tensor2};

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


static auto& KernelRegistry =
    habana::KernelRegistry()
        .add("aten::addcmul",
             [](const int device_id, c10::ScalarType node_type) {
               return std::make_shared<AddcmulOperator>(device_id, node_type);
             })
        .add("aten::addcdiv",
             [](const int device_id, c10::ScalarType node_type) {
               return std::make_shared<AddcdivOperator>(device_id, node_type);
        });

static auto registry =
    torch::RegisterOperators()
        .op(torch::RegisterOperators::options()
                .schema(
                    "aten::addcmul(Tensor self, Tensor tensor1, Tensor tensor2, *, Scalar value=1) -> Tensor")
                .impl_unboxedOnlyKernel<decltype(addcmul_hpu), &addcmul_hpu>(
                    DispatchKey::HABANATensorId)
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
                .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA));
