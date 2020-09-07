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
#include "habana_kernels/compare_kernels.h"
#include "habana_kernels/kernel_utils.h"
#include "habana_kernels/simple_generic_kernel.h"
#include "habana_kernels/tensor_shape_kernels.h"

using namespace torch;

void CompareOutOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    bool is_output_persistent) {
  TORCH_CHECK(
      inputs.size() == 3,
      "Incorrect size of input arguments for Compare Out Operator");
  TORCH_CHECK(
      inputs[0].isTensor(),
      "Input arg 1 for compare op needs to be tensor type");
  TORCH_CHECK(
      inputs[1].isTensor(),
      "Input arg 2 for compare op needs to be of tensor type");
  TORCH_CHECK(
      inputs[2].isTensor(),
      "Input arg 3 for compare op needs to be of tensor type");
  Tensor self = inputs[0].toTensor();
  Tensor other = inputs[1].toTensor();
  Tensor output = inputs[2].toTensor();

  if (self.ndimension() != other.ndimension()) {
    //
    // we need to reshape tensor which has lesser dimesnions, reshape_tensor_idx
    // points to the tensor for which we need to reshape & input_tensor_idx
    // points to tensor which goes directly to compare kernel without reshape
    int32_t reshape_tensor_idx =
        (self.ndimension() > other.ndimension()) ? 1 : 0;
    int32_t input_tensor_idx = (self.ndimension() > other.ndimension()) ? 0 : 1;
    Tensor& reshape_tensor =
        (self.ndimension() > other.ndimension()) ? other : self;
    Tensor& input_tensor =
        (self.ndimension() > other.ndimension()) ? self : other;
    std::vector<int64_t> reshaped_sizes = std::vector<int64_t>(
        input_tensor.ndimension() - reshape_tensor.ndimension(), 1);
    auto reshape_tensor_sizes = reshape_tensor.sizes().vec();

    reshaped_sizes.insert(
        reshaped_sizes.end(),
        reshape_tensor_sizes.begin(),
        reshape_tensor_sizes.end());

    ReshapeOperator reshape(this->p_context_->device_id_, this->scalarType_);
    auto& reshape_in_syn_tensor = reshape.SetSynapseInput(
        std::move(p_context_->syn_inputs_[reshape_tensor_idx]));
    torch::jit::Stack stack = {IValue(reshape_tensor), IValue(reshaped_sizes)};
    reshape.AllocateAndAddSynapseNode(graph, stack, false);
    p_context_->syn_inputs_[reshape_tensor_idx] =
        std::move(reshape_in_syn_tensor);

    AllocateSynapseOutput(graph, output, is_output_persistent);
    synapse_helpers::tensor& reshape_out_syn_tensor =
        reshape.GetSynOutputs()[0];
    std::vector<synTensor> syn_inputs(2, nullptr);
    synapse_helpers::tensor& output_syn_tensor = p_context_->syn_outputs_[0];
    std::vector<synTensor> syn_outputs{output_syn_tensor.get()};
    synapse_helpers::tensor& input_syn_tensor =
        p_context_->syn_inputs_[input_tensor_idx];
    syn_inputs[input_tensor_idx] = input_syn_tensor.get();
    syn_inputs[reshape_tensor_idx] = reshape_out_syn_tensor.get();
    graph.add_node(
        std::move(syn_inputs),
        std::move(syn_outputs),
        nullptr,
        0,
        std::move(guid_));
  } else {
    AllocateSynapseOutput(graph, output, is_output_persistent);
    AddNodeToSynapseGraph(graph, nullptr, 0);
  }
}

void CompareOutWrapperOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    bool is_output_persistent) {
  // this check is for stack during graph execution
  TORCH_CHECK(
      inputs.size() == 3,
      "Incorrect size of input expected for Compare operator");
  // Note that there is no (Scalar, Tensor) version for comparison ops
  // in native_functions.yaml
  TORCH_CHECK(inputs[0].isTensor(), "Input arg1 type expected to be tensor");
  TORCH_CHECK(
      inputs[1].isTensor() || inputs[1].isScalar(),
      "Input arg2 type expected to be a tensor or scalar");
  TORCH_CHECK(inputs[2].isTensor(), "Input arg3 type expected to be tensor");

  CompareOutOperator compareOp(
      this->p_context_->device_id_, this->scalarType_, guid_);

  if (inputs[1].isTensor()) { // Both inputs are tensors
    auto& syn_arg1 =
        compareOp.SetSynapseInput(std::move(p_context_->syn_inputs_[0]));
    auto& syn_arg2 =
        compareOp.SetSynapseInput(std::move(p_context_->syn_inputs_[1]));
    compareOp.AllocateAndAddSynapseNode(graph, inputs, is_output_persistent);
    p_context_->syn_inputs_[0] = std::move(syn_arg1);
    p_context_->syn_inputs_[1] = std::move(syn_arg2);

  } else { // 2nd input is a scalar
    // add constant node to convert 2nd input to tensor
    ConstantOperator constOp(this->p_context_->device_id_, this->scalarType_);
    auto& syn_arg1 =
        compareOp.SetSynapseInput(std::move(p_context_->syn_inputs_[0]));
    constOp.AllocateAndAddSynapseNode(graph, inputs, false);
    UNUSED auto& syn_arg2 =
        compareOp.SetSynapseInput(std::move(constOp.GetSynOutputs()[0]));
    // replace 2nd scalar input with a tensor in stack
    inputs.erase(inputs.cbegin() + 1);
    inputs.emplace(inputs.cbegin() + 1, constOp.GetOutputs()[0]);
    compareOp.AllocateAndAddSynapseNode(graph, inputs, is_output_persistent);
    p_context_->syn_inputs_[0] = std::move(syn_arg1);
  }

  p_context_->pt_outputs_.emplace_back(compareOp.GetOutputs()[0]);
  p_context_->syn_outputs_.emplace_back(
      std::move(compareOp.GetSynOutputs()[0]));
}

void CompareWrapperOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    bool is_output_persistent) {
  TORCH_CHECK(
      inputs.size() == 2,
      "Incorrect size of input arguments for Compare Operator");
  TORCH_CHECK(inputs[0].isTensor(), "Input arg1 type expected to be a tensor");
  TORCH_CHECK(
      inputs[1].isTensor() || inputs[1].isScalar(),
      "Input arg2 type expected to be a tensor or scalar");
  Tensor self = inputs[0].toTensor();
  auto output = habana_helpers::createPTTensor(
      self,
      self.sizes(),
      self.options(),
      self.suggest_memory_format(),
      c10::ScalarType::Bool,
      is_output_persistent);
  inputs.push_back(output);
  CompareOutWrapperOperator::AllocateAndAddSynapseNode(
      graph, inputs, is_output_persistent);
}

void CompareOutWrapperOperator::SetPTOutputs(torch::jit::Stack& inputs) {
  Tensor operand;
  if (inputs[0].isTensor() && inputs[1].isTensor()) {
    operand =
        get_correct_input_tensor(inputs[0].toTensor(), inputs[1].toTensor());
  } else if (inputs[0].isTensor()) {
    operand = inputs[0].toTensor();
  } else {
    operand = inputs[1].toTensor();
  }
  auto output = habana_helpers::createPTTensor(
      operand,
      operand.sizes(),
      operand.options(),
      operand.suggest_memory_format(),
      c10::ScalarType::Bool,
      true);
  HabanaOperator::SetPTOutputs({output});
}

template <class CompareOp>
Tensor compare_op_hpu(
    const std::vector<at::Tensor>& pt_inputs,
    torch::jit::Stack& stack,
    const std::string& node_guid) {
  PT_KERNEL_BEGIN;
  size_t device_id = pt_inputs[0].device().index();
  auto& device = synapse_helpers::HPURegistrar::get_device(device_id);
  at::ScalarType scalar_type = pt_inputs[0].scalar_type();
  std::string node_type =
      node_guid + "_fwd_" + habana_helpers::name_suffix_from_type(scalar_type);

  CompareOp Op(device_id, scalar_type);
  size_t key = Op.GetRecipeKey(node_type, stack);
  if (device.get_recipe_handle_cache().isCached(key)) {
    PT_KERNEL_DEBUG("Cache hit key:", key);
    Op.SetPTInputs(pt_inputs);
    Op.SetPTOutputs(stack);
    Op.Execute(key);
  } else {
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
  PT_KERNEL_END;
  return out[0];
}

/*************************************************************************
 * @brief Kernel implementation for aten.gt(self, other)
 * @param self - tensor_0
 * @param other - tensor_1
 ************************************************************************/
Tensor gt_hpu(Tensor& self, Tensor& other) {
  PT_KERNEL_BEGIN;
  std::vector<at::Tensor> pt_inputs{self, other};
  torch::jit::Stack stack{IValue(self), IValue(other)};
  auto output = compare_op_hpu<GtOperator>(pt_inputs, stack, "gt");
  PT_KERNEL_END;
  return output;
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
  std::vector<at::Tensor> pt_inputs{self, other, output};
  torch::jit::Stack stack{IValue(self), IValue(other), IValue(output)};
  compare_op_hpu<EqOutOperator>(pt_inputs, stack, "equal");
  PT_KERNEL_END;
}

/*************************************************************************
 * @brief Kernel implementation for out = torch.eq(self,other)
 * @param self - first input
 * @param other - second input
 ************************************************************************/
Tensor eq_tensor_hpu(Tensor& self, Tensor& other) {
  PT_KERNEL_BEGIN;
  std::vector<at::Tensor> pt_inputs{self, other};
  torch::jit::Stack stack{IValue(self), IValue(other)};
  auto output = compare_op_hpu<EqOperator>(pt_inputs, stack, "equal");
  PT_KERNEL_END;
  return output;
}

/*************************************************************************
 * @brief Kernel implementation for out = torch.eq(self,other)
 * @param self [in] - input tensor, 1-4D, FP32/BF16
 * @param other [in] - Scalar
 ************************************************************************/
Tensor eq_tensor_scalar_hpu(Tensor& self, Scalar other) {
  PT_KERNEL_BEGIN;
  if (self.dim() == 0) {
    self.unsafeGetTensorImpl()->set_sizes_and_strides({1}, {1});
  }
  std::vector<at::Tensor> pt_inputs{self};
  torch::jit::Stack stack{IValue(self), IValue(other)};
  auto output = compare_op_hpu<EqOperator>(pt_inputs, stack, "equal");
  PT_KERNEL_END;
  return output;
}

/*************************************************************************
 * @brief Kernel implementation for out = torch.lt(self,other)
 * @param self [in] - input tensor, 1-4D, FP32/BF16
 * @param other [in] - Scalar
 ************************************************************************/
Tensor lt_scalar_hpu(Tensor& self, Scalar other) {
  PT_KERNEL_BEGIN;
  if (self.dim() == 0) {
    self.unsafeGetTensorImpl()->set_sizes_and_strides({1}, {1});
  }
  std::vector<at::Tensor> pt_inputs{self};
  torch::jit::Stack stack{IValue(self), IValue(other)};
  auto output = compare_op_hpu<LtOperator>(pt_inputs, stack, "lt");
  PT_KERNEL_END;
  return output;
}

/*************************************************************************
 * @brief Kernel implementation for out = torch.lt(self,other)
 * @param self [in] - input tensor, 1-4D, FP32/BF16
 * @param other [in] - input tensor, 1-4D, FP32/BF16
 ************************************************************************/
Tensor lt_tensor_hpu(Tensor& self, Tensor& other) {
  PT_KERNEL_BEGIN;
  std::vector<at::Tensor> pt_inputs{self, other};
  torch::jit::Stack stack{IValue(self), IValue(other)};
  auto output = compare_op_hpu<LtOperator>(pt_inputs, stack, "lt");
  PT_KERNEL_END;
  return output;
}

static auto& KernelRegistry =
    habana::KernelRegistry()
        .add(
            "aten::gt",
            [](const int device_id, c10::ScalarType node_type) {
              return std::make_shared<GtOperator>(device_id, node_type);
            })
        .add(
            "aten::eq",
            [](const int device_id, c10::ScalarType node_type) {
              return std::make_shared<EqOperator>(device_id, node_type);
            })
        .add("aten::lt", [](const int device_id, c10::ScalarType node_type) {
          return std::make_shared<LtOperator>(device_id, node_type);
        });

static auto registry =
    torch::RegisterOperators()
        .op(torch::RegisterOperators::options()
                .schema("aten::gt.Tensor(Tensor self, Tensor other) -> Tensor")
                .impl_unboxedOnlyKernel<decltype(gt_hpu), &gt_hpu>(
                    DispatchKey::HABANATensorId)
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
                    decltype(eq_tensor_scalar_hpu),
                    &eq_tensor_scalar_hpu>(DispatchKey::HABANATensorId)
                .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
        .op(torch::RegisterOperators::options()
                .schema("aten::lt.Scalar(Tensor self, Scalar other) -> Tensor")
                .impl_unboxedOnlyKernel<
                    decltype(lt_scalar_hpu),
                    &lt_scalar_hpu>(DispatchKey::HABANATensorId)
                .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
        .op(torch::RegisterOperators::options()
                .schema("aten::lt.Tensor(Tensor self, Tensor other) -> Tensor")
                .impl_unboxedOnlyKernel<
                    decltype(lt_tensor_hpu),
                    &lt_tensor_hpu>(DispatchKey::HABANATensorId)
                .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA));