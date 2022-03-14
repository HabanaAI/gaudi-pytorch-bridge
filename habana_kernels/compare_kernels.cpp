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
using namespace habana;

void CompareOutOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    const OutputMetaDataVector& output_metadata) {
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
  // Tensor self = inputs[0].toTensor();
  // Tensor other = inputs[1].toTensor();
  Tensor output = inputs[2].toTensor();

  AllocateSynapseOutput(graph, output, output_metadata.at(0));
  AddNodeToSynapseGraph(graph, nullptr, 0);
}

void CompareOutWrapperOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    const OutputMetaDataVector& output_metadata) {
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

  auto compareOp = make_operator<CompareOutOperator>(
      this->p_context_->device_id_, this->scalarType_, guid_);

  if (inputs[1].isTensor()) { // Both inputs are tensors
    compareOp->SetSynapseInput(p_context_->syn_inputs_[0]);
    compareOp->SetSynapseInput(p_context_->syn_inputs_[1]);
    compareOp->AllocateAndAddSynapseNode(graph, inputs, output_metadata);
  } else { // 2nd input is a scalar
    // add constant node to convert 2nd input to tensor
    auto arg1 = inputs[0].toTensor();
    auto constOp = make_operator<ConstantOperator>(
        this->p_context_->device_id_, this->scalarType_);
    auto const_shape_tensor = habana_helpers::createPTTensor(
        arg1, {1}, arg1.options(), at::MemoryFormat::Contiguous, false);
    torch::jit::Stack constOp_stack = {IValue(const_shape_tensor), inputs[1]};
    constOp->AllocateAndAddSynapseNode(
        graph, constOp_stack, OutputMetaDataVector(1));
    compareOp->SetSynapseInput(p_context_->syn_inputs_[0]);
    compareOp->SetSynapseInput(constOp->GetSynOutputs()[0]);
    // replace 2nd scalar input with a tensor in stack
    inputs.erase(inputs.cbegin() + 1);
    inputs.emplace(inputs.cbegin() + 1, constOp->GetOutputs()[0]);
    compareOp->AllocateAndAddSynapseNode(graph, inputs, output_metadata);
  }

  p_context_->pt_outputs_.emplace_back(compareOp->GetOutputs()[0]);
  synapse_helpers::tensor& out_syn_t = compareOp->GetSynOutputs()[0];
  p_context_->syn_outputs_.emplace_back(out_syn_t);
}

void CompareOutWrapperOperator::SetPTOutputs(torch::jit::Stack& inputs) {
  auto output = inputs[2].toTensor();
  std::vector<at::Tensor> v{output};
  HabanaOperator::SetPTOutputs(v);
}

void CompareWrapperOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    const OutputMetaDataVector& output_metadata) {
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
      output_metadata.at(0).persistent);
  inputs.push_back(output);
  CompareOutWrapperOperator::AllocateAndAddSynapseNode(
      graph, inputs, output_metadata);
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

void GeOutOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    const OutputMetaDataVector& output_metadata) {
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

  Tensor output = inputs[2].toTensor();

  p_context_->syn_outputs_.emplace_back(
      habana_helpers::duplicate_tensor_in_memory_section(
          p_context_->syn_inputs_[2], graph, output_metadata.at(0).external));
  p_context_->pt_outputs_.emplace_back(output);

  std::vector<synTensor> syn_in;
  syn_in.emplace_back(
      static_cast<synapse_helpers::tensor&>(p_context_->syn_inputs_[0]).get());

  if (!inputs[1].isTensor()) { // 2nd input is a scalar
    // add constant node to convert 2nd input to tensor
    auto self = inputs[0].toTensor();
    auto constOp = make_operator<ConstantOperator>(
        this->p_context_->device_id_, this->scalarType_);
    auto const_shape_tensor = habana_helpers::createPTTensor(
        self, {1}, self.options(), at::MemoryFormat::Contiguous, false);
    torch::jit::Stack constOp_stack = {IValue(const_shape_tensor), inputs[1]};
    constOp->AllocateAndAddSynapseNode(
        graph, constOp_stack, OutputMetaDataVector(1));
    syn_in.emplace_back(
        static_cast<synapse_helpers::tensor&>(constOp->GetSynOutputs()[0])
            .get());
  } else {
    syn_in.emplace_back(
        static_cast<synapse_helpers::tensor&>(p_context_->syn_inputs_[1])
            .get());
  }

  std::vector<synTensor> syn_out;
  syn_out.emplace_back(
      static_cast<synapse_helpers::tensor&>(p_context_->syn_outputs_[0]).get());

  graph.add_node(std::move(syn_in), std::move(syn_out), nullptr, 0, guid_);
}

template <class CompareOp>
Tensor compare_op_hpu(
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
    OutputMetaDataVector output_metadata(1);
    output_metadata.at(0).persistent = true;
    Op.AllocateAndAddSynapseNode(graph, stack, output_metadata);

    // compile and execute the graph
    Op.Compile(graph);
  }

  std::vector<at::Tensor> out = Op.GetOutputs();
  TORCH_CHECK(out.size() == 1, "Incorrect size of outputs");
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
Tensor eq_tensor_scalar_hpu(const Tensor& self, const Scalar& other) {
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
Tensor ge_scalar_hpu(const Tensor& self, const Scalar& other) {
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
 * @brief Kernel implementation for out = torch.le(self,other)
 * @param self [in] - input tensor, 1-5D, FP32/BF16/I8/U8/I32
 * @param other [in] - Scalar
 ************************************************************************/
Tensor le_scalar_hpu(const Tensor& self, const Scalar& other) {
  PT_KERNEL_BEGIN;
  bool isSelf_0d = false;
  if (self.dim() == 0) {
    self.unsafeGetTensorImpl()->set_sizes_and_strides({1}, {1});
    isSelf_0d = true;
  }
  std::vector<at::Tensor> pt_inputs{self};
  torch::jit::Stack stack{IValue(self), IValue(other)};
  auto output = compare_op_hpu<LeOperator>(pt_inputs, stack, "less_equal");
  if (isSelf_0d) {
    self.unsafeGetTensorImpl()->set_sizes_and_strides({}, {});
    output.unsafeGetTensorImpl()->set_sizes_and_strides({}, {});
  }
  PT_KERNEL_END;
  return output;
}

/*************************************************************************
 * @brief Kernel implementation for out = torch.le(self,other)
 * @param self [in] - input tensor, 1-5D, FP32/BF16/I8/U8/I32
 * @param other [in] - input tensor, 1-5D, FP32/BF16/I8/U8/I32
 ************************************************************************/
Tensor le_tensor_hpu(const Tensor& self, const Tensor& other) {
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

  std::vector<at::Tensor> pt_inputs{self, other};
  torch::jit::Stack stack{IValue(self), IValue(other)};
  auto output = compare_op_hpu<LeOperator>(pt_inputs, stack, "less_equal");

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

/*************************************************************************
 * @brief Kernel implementation for aten.ne(self, other)
 * @param self - tensor_0
 * @param other - tensor_1
 ************************************************************************/
Tensor ne_tensor_hpu(const Tensor& self_in, const Tensor& other_in) {
  PT_KERNEL_BEGIN;
  auto self = self_in;
  // Char & Bool are both treated as I8 on Habana device. This cosmetic dtype
  // overwrite is done here to avoid problems in adding trivial cast node later.
  if (self.scalar_type() == c10::ScalarType::Char) {
    self = self.to(c10::ScalarType::Bool);
  }
  auto other = other_in;
  if (other.scalar_type() == c10::ScalarType::Char) {
    other = other.to(c10::ScalarType::Bool);
  }
  // create OP graph and populate the stack with inputs
  auto graph = std::make_shared<torch::jit::Graph>();
  const auto graph_string = R"IR(
  graph(%a, %b):
    %c : Tensor = aten::ne(%a, %b)
    return (%c))IR";
  torch::jit::parseIR(graph_string, graph.get());
  torch::jit::Stack stack = {self, other};

  habana_lazy::transform_graph(graph);

  std::shared_ptr<habana_lazy::OptimizedJITGraphAndMetaData>
      jit_ir_graph_and_mdata =
          std::make_shared<habana_lazy::OptimizedJITGraphAndMetaData>();
  jit_ir_graph_and_mdata->set_cached_graph(graph);
  jit_ir_graph_and_mdata->SetOpName("ne_tensor");
  // Execute OP graph
  HabanaLaunchOpPT launch{jit_ir_graph_and_mdata};
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
Tensor ne_scalar_hpu(const Tensor& self_in, Scalar other) {
  PT_KERNEL_BEGIN;
  if (self_in.dim() == 0) {
    self_in.unsafeGetTensorImpl()->set_sizes_and_strides({1}, {1});
  }
  auto self = self_in;
  if (self.scalar_type() == c10::ScalarType::Long) {
    self = habana_helpers::cast_tensor_to_integer(self_in);
  } else if (self.scalar_type() == c10::ScalarType::Char) {
    // Char & Bool are both treated as I8 on Habana device. This cosmetic dtype
    // overwrite is done here to avoid problems in adding trivial cast node
    // later.
    self = self.to(c10::ScalarType::Bool);
  }
  // create OP graph and populate the stack with inputs
  auto graph = std::make_shared<torch::jit::Graph>();
  const auto graph_string = R"IR(
  graph(%a, %b : int):
    %c : Tensor = aten::ne(%a, %b)
    return (%c))IR";
  torch::jit::parseIR(graph_string, graph.get());
  torch::jit::Stack stack = {self, other};

  habana_lazy::transform_graph(graph);

  std::shared_ptr<habana_lazy::OptimizedJITGraphAndMetaData>
      jit_ir_graph_and_mdata =
          std::make_shared<habana_lazy::OptimizedJITGraphAndMetaData>();
  jit_ir_graph_and_mdata->set_cached_graph(graph);
  jit_ir_graph_and_mdata->SetOpName("ne_scalar");
  // Execute OP graph
  HabanaLaunchOpPT launch{jit_ir_graph_and_mdata};
  launch.run(stack);

  // Pop output from stack
  PT_KERNEL_END;
  return stack.back().toTensor();
}
// Lazy implementation of compare ops have been moved to auto code gen.
// Hence all registrations are done as part of auto code gen.
