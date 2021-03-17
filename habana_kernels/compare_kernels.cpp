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
#include <torch/csrc/jit/ir/irparser.h>
#include <torch/script.h>

#include "habana_bridge/kernel/hpu_habana_launch_op_pt.h"
#include "habana_device/HPUCheck.h"
#include "habana_device/hpu_cached_devices.h"
#include "habana_helpers/graph.h"
#include "habana_helpers/logging.h"
#include "habana_kernels/binary_kernels.h"
#include "habana_kernels/compare_kernels.h"
#include "habana_kernels/kernel_utils.h"
#include "habana_kernels/simple_generic_kernel.h"
#include "habana_kernels/tensor_shape_kernels.h"
#include "habana_lazy/hlexec.h"
#include "habana_lazy/passes/transform_graph.h"

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

void CompareOutWrapperOperator::SetPTOutputs(torch::jit::Stack& inputs) {
  auto output = inputs[2].toTensor();
  std::vector<at::Tensor> v{output};
  HabanaOperator::SetPTOutputs(v);
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
  std::vector<int64_t> out_shape;
  Tensor operand;
  if (inputs[0].isTensor() && inputs[1].isTensor()) {
    operand = inputs[0].toTensor();
    out_shape =
        compute_output_shape(inputs[0].toTensor(), inputs[1].toTensor());
  } else if (inputs[0].isTensor()) {
    operand = inputs[0].toTensor();
    out_shape = operand.sizes().vec();
  } else {
    operand = inputs[1].toTensor();
    out_shape = operand.sizes().vec();
  }
  auto output = habana_helpers::createPTTensor(
      operand,
      IntArrayRef(out_shape.data(), out_shape.size()),
      operand.options(),
      operand.suggest_memory_format(),
      c10::ScalarType::Bool,
      is_output_persistent);
  inputs.push_back(output);
  CompareOutWrapperOperator::AllocateAndAddSynapseNode(
      graph, inputs, is_output_persistent);
}

void CompareWrapperOperator::SetPTOutputs(torch::jit::Stack& inputs) {
  std::vector<int64_t> out_shape;
  Tensor operand;
  if (inputs[0].isTensor() && inputs[1].isTensor()) {
    operand = inputs[0].toTensor();
    out_shape =
        compute_output_shape(inputs[0].toTensor(), inputs[1].toTensor());
  } else if (inputs[0].isTensor()) {
    operand = inputs[0].toTensor();
    out_shape = operand.sizes().vec();
  } else {
    operand = inputs[1].toTensor();
    out_shape = operand.sizes().vec();
  }
  auto output = habana_helpers::createPTTensor(
      operand,
      IntArrayRef(out_shape.data(), out_shape.size()),
      operand.options(),
      operand.suggest_memory_format(),
      c10::ScalarType::Bool,
      true);
  std::vector<at::Tensor> v{output};
  HabanaOperator::SetPTOutputs(v);
}

std::vector<int64_t> CompareWrapperOperator::compute_output_shape(
    const Tensor& arg1,
    const Tensor& arg2) {
  auto out_size = habana_helpers::compute_broadcast_shape(arg1, arg2);
  return out_size;
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
Tensor gt_tensor_hpu(const Tensor& self, const Tensor& other) {
  PT_KERNEL_BEGIN;
  std::vector<at::Tensor> pt_inputs{self, other};
  torch::jit::Stack stack{IValue(self), IValue(other)};
  auto output = compare_op_hpu<GtOperator>(pt_inputs, stack, "gt");
  PT_KERNEL_END;
  return output;
}

/*************************************************************************
 * @brief Kernel implementation for aten.gt(self, other)
 * @param self - tensor_0
 * @param other - Scalar
 ************************************************************************/
Tensor gt_scalar_hpu(const Tensor& self, Scalar other) {
  PT_KERNEL_BEGIN;
  if (self.dim() == 0) {
    self.unsafeGetTensorImpl()->set_sizes_and_strides({1}, {1});
  }
  std::vector<at::Tensor> pt_inputs{self};
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
Tensor& eq_tensor_out_hpu(
    Tensor& output,
    const Tensor& self,
    const Tensor& other) {
  PT_KERNEL_BEGIN;
  std::vector<at::Tensor> pt_inputs{self, other, output};
  torch::jit::Stack stack{IValue(self), IValue(other), IValue(output)};
  compare_op_hpu<EqOutOperator>(pt_inputs, stack, "equal");
  PT_KERNEL_END;
  return output;
}

/*************************************************************************
 * @brief Kernel implementation for out = torch.eq(self,other)
 * @param self - first input
 * @param other - second input
 ************************************************************************/
Tensor eq_tensor_hpu(const Tensor& self, const Tensor& other) {
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
Tensor eq_tensor_scalar_hpu(const Tensor& self, Scalar other) {
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
Tensor lt_scalar_hpu(const Tensor& self, Scalar other) {
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
Tensor lt_tensor_hpu(const Tensor& self, const Tensor& other) {
  PT_KERNEL_BEGIN;
  std::vector<at::Tensor> pt_inputs{self, other};
  torch::jit::Stack stack{IValue(self), IValue(other)};
  auto output = compare_op_hpu<LtOperator>(pt_inputs, stack, "lt");
  PT_KERNEL_END;
  return output;
}

/*************************************************************************
 * @brief Kernel implementation for out = torch.ge(self,other)
 * @param self [in] - input tensor, 1-4D, FP32/BF16
 * @param other [in] - Scalar
 ************************************************************************/
Tensor ge_scalar_hpu(const Tensor& self, Scalar other) {
  PT_KERNEL_BEGIN;
  if (self.dim() == 0) {
    self.unsafeGetTensorImpl()->set_sizes_and_strides({1}, {1});
  }
  std::vector<at::Tensor> pt_inputs{self};
  torch::jit::Stack stack{IValue(self), IValue(other)};
  auto output = compare_op_hpu<GeOperator>(pt_inputs, stack, "ge");
  PT_KERNEL_END;
  return output;
}

/*************************************************************************
 * @brief Kernel implementation for out = torch.ge(self,other)
 * @param self [in] - input tensor, 1-4D, FP32/BF16
 * @param other [in] - input tensor, 1-4D, FP32/BF16
 ************************************************************************/
Tensor ge_tensor_hpu(const Tensor& self, const Tensor& other) {
  PT_KERNEL_BEGIN;
  std::vector<at::Tensor> pt_inputs{self, other};
  torch::jit::Stack stack{IValue(self), IValue(other)};
  auto output = compare_op_hpu<GeOperator>(pt_inputs, stack, "ge");
  PT_KERNEL_END;
  return output;
}

/*************************************************************************
 * @brief Kernel implementation for aten.ne(self, other)
 * @param self - tensor_0
 * @param other - tensor_1
 ************************************************************************/
Tensor ne_tensor_hpu(const Tensor& self, const Tensor& other) {
  PT_KERNEL_BEGIN;
  // create OP graph and populate the stack with inputs
  auto graph = std::make_shared<torch::jit::Graph>();
  const auto graph_string = R"IR(
  graph(%a, %b):
    %c : Tensor = aten::ne(%a, %b)
    return (%c))IR";
  torch::jit::parseIR(graph_string, graph.get());
  torch::jit::Stack stack = {self, other};

  habana_lazy::exec::OptPassCfg::GetInstance()->enable_subgraph_rewrite = true;
  habana_lazy::transform_graph(graph);
  habana_lazy::exec::OptPassCfg::GetInstance()->enable_subgraph_rewrite = false;
  // reset instance count so that graph_id always remains same
  // this ensures that we get a cache hit if inputs have not changed
  HabanaLaunchOpPT::instance_count_ = 0;

  // Execute OP graph
  HabanaLaunchOpPT launch{graph, false};
  launch.run(stack);

  // Pop output from stack
  PT_KERNEL_END;
  return stack.back().toTensor();
}

/*************************************************************************
 * @brief Kernel implementation for aten.ne(self, other)
 * @param self - tensor_0
 * @param other - Scalar
 ************************************************************************/
Tensor ne_scalar_hpu(const Tensor& self, Scalar other) {
  PT_KERNEL_BEGIN;
  if (self.dim() == 0) {
    self.unsafeGetTensorImpl()->set_sizes_and_strides({1}, {1});
  }
  // create OP graph and populate the stack with inputs
  auto graph = std::make_shared<torch::jit::Graph>();
  const auto graph_string = R"IR(
  graph(%a, %b : int):
    %c : Tensor = aten::ne(%a, %b)
    return (%c))IR";
  torch::jit::parseIR(graph_string, graph.get());
  torch::jit::Stack stack = {self, other};

  habana_lazy::exec::OptPassCfg::GetInstance()->enable_subgraph_rewrite = true;
  habana_lazy::transform_graph(graph);
  habana_lazy::exec::OptPassCfg::GetInstance()->enable_subgraph_rewrite = false;

  // reset instance count so that graph_id always remains same
  // this ensures that we get a cache hit if inputs have not changed
  HabanaLaunchOpPT::instance_count_ = 0;

  // Execute OP graph
  HabanaLaunchOpPT launch{graph, false};
  launch.run(stack);

  // Pop output from stack
  PT_KERNEL_END;
  return stack.back().toTensor();
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
        .add(
            "aten::lt",
            [](const int device_id, c10::ScalarType node_type) {
              return std::make_shared<LtOperator>(device_id, node_type);
            })
        .add("aten::ge", [](const int device_id, c10::ScalarType node_type) {
          return std::make_shared<GeOperator>(device_id, node_type);
        });
