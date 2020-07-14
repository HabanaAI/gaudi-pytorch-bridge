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

void CompareOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    bool is_output_persistent) {
  TORCH_CHECK(
      inputs.size() == 2,
      "Incorrect size of input arguments for Compare Operator");
  TORCH_CHECK(
      inputs[0].isTensor(),
      "Input arg 1 for compare op needs to be tensor type");
  TORCH_CHECK(
      inputs[1].isTensor(),
      "Input arg 2 for compare op needs to be of tensor type");
  Tensor self = inputs[0].toTensor();
  Tensor other = inputs[1].toTensor();
  auto operand = get_correct_input_tensor(self, other);
  auto output = at::empty(
      operand.sizes(),
      operand.options().dtype(c10::ScalarType::Bool),
      operand.suggest_memory_format());
  inputs.push_back(output);
  CompareOutOperator::AllocateAndAddSynapseNode(
      graph, inputs, is_output_persistent);
}

Tensor compare_op_hpu(
    const std::vector<at::Tensor>& pt_inputs,
    const std::string& node_type,
    size_t device_id,
    CompareOutOperator* Op) {
  PT_KERNEL_BEGIN;
  // Build Params for the graph
  std::vector<c10::IValue> stack;
  for (auto pt_input : pt_inputs) {
    stack.emplace_back(IValue(pt_input));
  }

  // Create Graph
  auto graph = habana_helpers::create_graph(device_id, node_type);

  // temp_pt_inputs is created because EqOut has 3 Tensors in pt_inputs
  // which makes GC throw an error because it expects 2 inputs for equal
  std::vector<at::Tensor> temp_pt_inputs{pt_inputs[0], pt_inputs[1]};
  // Assign Inputs to the Operator
  Op->AllocateSynapseInputs(graph, temp_pt_inputs, true);

  Op->AllocateAndAddSynapseNode(graph, stack, true);

  // compile and execute the graph
  Op->Compile(graph);

  std::vector<at::Tensor> out = Op->GetOutputs();
  TORCH_CHECK(out.size() == 1, "Incorrect size of outputs");
  PT_KERNEL_END;
  return out.at(0);
}

/*************************************************************************
 * @brief Kernel implementation for aten.gt(self, other)
 * @param self - tensor_0
 * @param other - tensor_1
 ************************************************************************/
Tensor gt_hpu(Tensor& self, Tensor& other) {
  PT_KERNEL_BEGIN;
  size_t device_id = self.device().index();
  at::ScalarType scalar_type = self.scalar_type();
  std::string node_type =
      "gt_fwd_" + habana_helpers::name_suffix_from_type(scalar_type);
  std::vector<at::Tensor> pt_inputs{self, other};

  GtOperator op(device_id, scalar_type);
  auto out = compare_op_hpu(pt_inputs, node_type, device_id, &op);
  PT_KERNEL_END;
  return out;
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
  size_t device_id = self.device().index();
  at::ScalarType scalar_type = self.scalar_type();
  std::string node_type =
      "equal_fwd_" + habana_helpers::name_suffix_from_type(scalar_type);
  std::vector<at::Tensor> pt_inputs{self, other, output};

  EqOutOperator op(device_id, scalar_type);
  compare_op_hpu(pt_inputs, node_type, device_id, &op);
  PT_KERNEL_END;
}

/*************************************************************************
 * @brief Kernel implementation for out = torch.eq(self,other)
 * @param self - first input
 * @param other - second input
 ************************************************************************/
Tensor eq_tensor_hpu(Tensor& self, Tensor& other) {
  PT_KERNEL_BEGIN;
  size_t device_id = self.device().index();
  at::ScalarType scalar_type = self.scalar_type();
  std::string node_type =
      "equal_fwd_" + habana_helpers::name_suffix_from_type(scalar_type);
  std::vector<at::Tensor> pt_inputs{self, other};

  EqOperator op(device_id, scalar_type);
  auto output = compare_op_hpu(pt_inputs, node_type, device_id, &op);
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

static auto& KernelRegistry =
    habana::KernelRegistry()
        .add(
            "aten::gt",
            [](const int device_id, c10::ScalarType node_type) {
              return std::make_shared<GtOperator>(device_id, node_type);
            })
        .add("aten::eq", [](const int device_id, c10::ScalarType node_type) {
          return std::make_shared<EqOperator>(device_id, node_type);
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
                    decltype(eq_scalar_tensor_hpu),
                    &eq_scalar_tensor_hpu>(DispatchKey::HABANATensorId)
                .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA));
