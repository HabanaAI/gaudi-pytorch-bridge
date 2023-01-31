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

#include "backend/create_pt_tensor.h"
#include "backend/helpers/create_tensor.h"
#include "habana_device/HPUCheck.h"
#include "habana_device/hpu_cached_devices.h"
#include "habana_helpers/dtype_helpers.h"
#include "habana_helpers/tensor_utils.h"
#include "habana_kernels/basic_kernels.h"
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

habana::OutputShapeInfRetType habana::BinaryOperator::ComputeOutputShape(
    torch::jit::Stack& inputs) {
  Tensor arg1 = inputs[0].toTensor();
  Tensor arg2 = inputs[1].toTensor();
  auto shape_out = BinaryOperator::compute_output_shape(arg1, arg2);

  OutputShapeInfRetType out;

  auto memory_format = at::MemoryFormat::Contiguous;
  if ((arg1.suggest_memory_format() == at::MemoryFormat::ChannelsLast) ||
      (arg2.suggest_memory_format() == at::MemoryFormat::ChannelsLast)) {
    memory_format = at::MemoryFormat::ChannelsLast;
  }

  // For cases where 2nd argument to a binary op is a scalar (e.g. a = b + 5),
  // but gets converted to a tensor (by dispatcher or bridge) before reaching
  // kernel we need to cast 2nd argument to same type as 1st argument.
  if (arg1.dtype() == c10::ScalarType::BFloat16 &&
      arg1.dtype() != arg2.dtype()) {
    auto castOp = make_operator<CastOperator>(
        this->p_context_->device_id_, "cast_f32_to_bf16");
    torch::jit::Stack stack = {IValue(arg2), IValue(c10::ScalarType::BFloat16)};
    // Cast Input is pushed in to Syn Input
    (void)out.call_ComputeOutputShape(castOp, stack);
    auto shape_out = BinaryOperator::compute_output_shape(arg1, arg2);
    out.AddOutputTensor(TensorMetaData(
        shape_out,
        HabanaOperator::CalculateStrides(shape_out, memory_format),
        c10::ScalarType::BFloat16,
        memory_format));
  } else {
    auto shape_out = BinaryOperator::compute_output_shape(arg1, arg2);
    out.AddOutputTensor(TensorMetaData(
        shape_out,
        HabanaOperator::CalculateStrides(
            shape_out, arg1.suggest_memory_format()),
        arg1.scalar_type(),
        arg1.suggest_memory_format()));
  }
  return out;
}

bool habana::BinaryOperator::MaybeMultiplyWithBool(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    const OutputMetaData& output_metadata) {
  Tensor arg1 = inputs[0].toTensor();
  Tensor arg2 = inputs[1].toTensor();
  bool is_arg1_integral = isIntegralType(arg1.scalar_type(), false);
  // habana_helpers::is_integral_tensor(arg1);
  bool is_arg2_integral = isIntegralType(arg2.scalar_type(), false);
  // habana_helpers::is_integral_tensor(arg2);

  // This if block is introduced to support bool tensors for mult
  // TPC does not support bool for mult operation
  // cast node is added before and after the tpc call to support bool
  // This change is done as an WA to avoid any script change
  // as a part of <https://jira.habana-labs.com/browse/SW-48605>
  HABANA_ASSERT(
      guid_.substr(0, 4) == "mult",
      "MaybeMultiplyWithBool supports only mult op");
  if ((arg1.scalar_type() == c10::ScalarType::Bool || is_arg1_integral) &&
      (arg2.scalar_type() == c10::ScalarType::Bool || is_arg2_integral) &&
      !(is_arg1_integral && is_arg2_integral)) {
    // NOTE: TO DO: if integral type is U8 we will fail in cast
    c10::ScalarType final_out_dtype = c10::ScalarType::Int;

    if (arg1.scalar_type() == arg2.scalar_type()) {
      final_out_dtype = arg1.scalar_type();
    } else {
      // Check if we have this key to find the dtype to which smaller dtype
      // tensor should be promoted to
      int pos = -1;
      habana_helpers::type_promotion_for_two_tensor_inputs(
          inputs, pos, final_out_dtype);
    }
    // Cast Input tensor to Int tensor
    std::string node1_type = (arg1.scalar_type() == c10::ScalarType::Int)
        ? "cast_identity"
        : (arg1.scalar_type() == c10::ScalarType::Short) ? "cast_i16_to_i32"
                                                         : "cast_i8_to_i32";
    // Create the operator
    // Build Params for the graph
    std::shared_ptr<HabanaOperator> castOp1;
    std::vector<c10::IValue> stack;
    auto md = OutputMetaDataVector(1);
    md[0].dtype = at::kInt;

    if (arg1.scalar_type() != c10::ScalarType::Int) {
      castOp1 =
          make_operator<CastOperator>(this->p_context_->device_id_, node1_type);
      castOp1->SetSynapseInput(p_context_->syn_inputs_[0]);
      stack = {arg1, c10::ScalarType::Int};
      castOp1->AllocateAndAddSynapseNode(graph, stack, md);
    } else {
      castOp1 = make_operator<IdentityOperator>(
          this->p_context_->device_id_, c10::ScalarType::Int);
      castOp1->SetSynapseInput(p_context_->syn_inputs_[0]);
      stack = {arg1};
      castOp1->AllocateAndAddSynapseNode(graph, stack, md);
    }
    std::string node2_type = (arg2.scalar_type() == c10::ScalarType::Int)
        ? "cast_identity"
        : (arg2.scalar_type() == c10::ScalarType::Short) ? "cast_i16_to_i32"
                                                         : "cast_i8_to_i32";
    // Create the operator
    std::shared_ptr<HabanaOperator> castOp2;
    // Build Params for the graph
    if (arg2.scalar_type() != c10::ScalarType::Int) {
      castOp2 =
          make_operator<CastOperator>(this->p_context_->device_id_, node2_type);
      castOp2->SetSynapseInput(p_context_->syn_inputs_[1]);
      stack = {arg2, c10::ScalarType::Int};
      castOp2->AllocateAndAddSynapseNode(graph, stack, md);
    } else {
      castOp2 = make_operator<IdentityOperator>(
          this->p_context_->device_id_, c10::ScalarType::Int);
      castOp2->SetSynapseInput(p_context_->syn_inputs_[1]);
      stack = {arg2};
      castOp2->AllocateAndAddSynapseNode(graph, stack, md);
    }

    // Add the Mult node
    auto out_shape = BinaryOperator::compute_output_shape(arg1, arg2);
    auto output_mult = habana::createPTTensor(
        arg1,
        IntArrayRef(out_shape.data(), out_shape.size()),
        arg1.options(),
        arg1.suggest_memory_format(),
        c10::ScalarType::Int,
        (final_out_dtype == c10::ScalarType::Int) ? output_metadata.persistent
                                                  : false);

    AllocateSynapseOutput(
        graph,
        output_mult,
        (final_out_dtype == c10::ScalarType::Int) ? output_metadata
                                                  : OutputMetaData(),
        false);
    synapse_helpers::tensor& synOutput = p_context_->syn_outputs_[0];
    synapse_helpers::tensor& synInput1 = castOp1->GetSynOutputs()[0];
    synapse_helpers::tensor& synInput2 = castOp2->GetSynOutputs()[0];

    std::vector<synTensor> syn_in{synInput1.get(), synInput2.get()};
    std::vector<synTensor> syn_out{synOutput.get()};
    guid_ =
        MULT_GUID + habana_helpers::name_suffix_from_type(c10::ScalarType::Int);
    graph.add_node(
        std::move(syn_in),
        std::move(syn_out),
        nullptr,
        0,
        guid_,
        nullptr,
        nullptr,
        nullptr,
        deterministic);
    if (final_out_dtype != c10::ScalarType::Int) {
      // NOTE: TO DO: need to handle integral type U8
      // Cast Int tensor to Bool tensor
      auto node_type = (final_out_dtype == c10::ScalarType::Short)
          ? "cast_i32_to_i16"
          : "cast_i32_to_i8";
      // Create the operator
      auto finalCastOp =
          make_operator<CastOperator>(this->p_context_->device_id_, node_type);
      finalCastOp->SetSynapseInput(p_context_->syn_outputs_[0]);
      // Build Params for the graph
      stack = {output_mult, final_out_dtype};
      auto md = output_metadata;
      md.dtype = final_out_dtype;
      finalCastOp->AllocateAndAddSynapseNode(graph, stack, {md});
      p_context_->syn_outputs_.pop_back();
      p_context_->pt_outputs_.pop_back();

      p_context_->syn_outputs_.emplace_back(
          std::move(finalCastOp->GetSynOutputs()[0]));
      p_context_->pt_outputs_.emplace_back(finalCastOp->GetOutputs()[0]);
    }
    return true;
  }
  return false;
}
/************************************************************************
 * @brief This function implements synapse node addition for
 * binary operators where both inputs are tensors. Mismatch in input
 * tensor dims is also taken care of using reshape nodes.
 ************************************************************************/
void habana::BinaryOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    const habana::OutputMetaDataVector& output_metadata) {
  // this check is for stack during graph execution
  TORCH_CHECK(
      inputs.size() == 2,
      "Incorrect size of input expected for Binary operator");
  TORCH_CHECK(inputs[0].isTensor(), "Input type expected to be tensor");
  TORCH_CHECK(inputs[1].isTensor(), "Input type expected to be tensor");
  Tensor arg1 = inputs[0].toTensor();
  Tensor arg2 = inputs[1].toTensor();

  if (guid_.substr(0, 4) == "mult") {
    auto was_bool_mult =
        MaybeMultiplyWithBool(graph, inputs, output_metadata.at(0));
    if (was_bool_mult)
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
  if (arg1.dtype() == c10::ScalarType::BFloat16 &&
      arg1.dtype() != arg2.dtype()) {
    std::string node_type = "cast_f32_to_bf16";
    auto castOp =
        make_operator<CastOperator>(this->p_context_->device_id_, node_type);
    castOp->SetSynapseInput(arg2_syn_tensor);
    torch::jit::Stack stack = {IValue(arg2), IValue(c10::ScalarType::BFloat16)};
    auto md = habana::OutputMetaDataVector(1);
    md[0].dtype = stack[1].toScalarType();
    castOp->AllocateAndAddSynapseNode(graph, stack, md);
    synapse_helpers::tensor& syn_tensor = std::move(castOp->GetSynOutputs()[0]);
    syn_inputs.push_back(syn_tensor.get());
    auto out_shape = BinaryOperator::compute_output_shape(arg1, arg2);
    auto output = habana::createPTTensor(
        arg1,
        IntArrayRef(out_shape.data(), out_shape.size()),
        arg1.options(),
        memory_format,
        c10::ScalarType::BFloat16,
        output_metadata.at(0).persistent);
    AllocateSynapseOutput(graph, output, output_metadata.at(0));
  } else {
    syn_inputs.push_back(arg2_syn_tensor.get());
    auto out_shape = BinaryOperator::compute_output_shape(arg1, arg2);
    auto output = habana::createPTTensor(
        arg1,
        IntArrayRef(out_shape.data(), out_shape.size()),
        arg1.options(),
        memory_format,
        output_metadata.at(0).persistent);
    AllocateSynapseOutput(graph, output, output_metadata.at(0));
  }

  synapse_helpers::tensor& output_syn_tensor = p_context_->syn_outputs_[0];
  std::vector<synTensor> syn_outputs{output_syn_tensor.get()};

  graph.add_node(
      std::move(syn_inputs),
      std::move(syn_outputs),
      nullptr,
      0,
      guid_,
      nullptr,
      nullptr,
      nullptr,
      deterministic);
}

/************************************************************************
 * @brief This function implements synapse node addition for Binary OPs
 * with 2 input arguments (where 1st input is always a tensor whereas
 * 2nd input can be a tensor or a scalar)
 ************************************************************************/
void habana::BinaryWrapperOperator::AllocateAndAddSynapseNode(
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

  std::shared_ptr<HabanaOperator> binaryOp;
  if (inputs[0].isTensor() && inputs[1].isTensor()) { // Both inputs are tensors
    binaryOp = make_operator<BinaryOperator>(
        this->p_context_->device_id_, guid_, this->scalarType_);
    binaryOp->SetSynapseInput(p_context_->syn_inputs_[0]);
    binaryOp->SetSynapseInput(p_context_->syn_inputs_[1]);
    binaryOp->AllocateAndAddSynapseNode(graph, inputs, output_metadata);

  } else if (inputs[0].isTensor() && inputs[1].isScalar()) { // 2nd input is a
                                                             // scalar
    auto arg1 = inputs[0].toTensor();
    // add constant node to convert 2nd input to tensor
    auto constOp = make_operator<ConstantOperator>(
        this->p_context_->device_id_, this->scalarType_);
    auto const_shape_tensor = habana::createPTTensor(
        arg1, {1}, arg1.options(), at::MemoryFormat::Contiguous, false);
    torch::jit::Stack constOp_stack = {IValue(const_shape_tensor), inputs[1]};
    constOp->AllocateAndAddSynapseNode(
        graph, constOp_stack, habana::OutputMetaDataVector(1));

    binaryOp = make_operator<BinaryOperator>(
        this->p_context_->device_id_, guid_, this->scalarType_);
    binaryOp->SetSynapseInput(p_context_->syn_inputs_[0]);
    binaryOp->SetSynapseInput(constOp->GetSynOutputs()[0]);
    // replace input scalar with input tensor in the stack
    auto scalar_input = inputs.back();
    inputs.pop_back();
    inputs.emplace_back(constOp->GetOutputs()[0]);
    binaryOp->AllocateAndAddSynapseNode(graph, inputs, output_metadata);
    // revert the stack changes
    if (GET_ENV_FLAG_NEW(PT_HPU_VALIDATE_COMPUTE_SHAPE) ||
        GET_ENV_FLAG_NEW(PT_HPU_ENABLE_FAST_SHAPE_INFERENCE)) {
      inputs.pop_back();
      inputs.emplace_back(scalar_input);
    }
  } else { // 1st input is a scalar
    auto arg2 = inputs[1].toTensor();
    // add constant node to convert 1st input to tensor
    auto constOp = make_operator<ConstantOperator>(
        this->p_context_->device_id_, this->scalarType_);
    auto const_shape_tensor = habana::createPTTensor(
        arg2, {1}, arg2.options(), at::MemoryFormat::Contiguous, false);
    torch::jit::Stack constOp_stack = {IValue(const_shape_tensor), inputs[0]};
    constOp->AllocateAndAddSynapseNode(
        graph, constOp_stack, habana::OutputMetaDataVector(1));

    binaryOp = make_operator<BinaryOperator>(
        this->p_context_->device_id_, guid_, this->scalarType_);
    binaryOp->SetSynapseInput(constOp->GetSynOutputs()[0]);
    binaryOp->SetSynapseInput(p_context_->syn_inputs_[0]);
    // replace input scalar with input tensor in the stack
    auto scalar_input = inputs.front();
    inputs.erase(inputs.cbegin());
    inputs.emplace(inputs.cbegin(), constOp->GetOutputs()[0]);
    binaryOp->AllocateAndAddSynapseNode(graph, inputs, output_metadata);
    // revert the stack changes
    if (GET_ENV_FLAG_NEW(PT_HPU_VALIDATE_COMPUTE_SHAPE) ||
        GET_ENV_FLAG_NEW(PT_HPU_ENABLE_FAST_SHAPE_INFERENCE)) {
      inputs.erase(inputs.cbegin());
      inputs.emplace(inputs.cbegin(), scalar_input);
    }
  }

  p_context_->pt_outputs_.emplace_back(binaryOp->GetOutputs()[0]);
  p_context_->syn_outputs_.emplace_back(
      std::move(binaryOp->GetSynOutputs()[0]));
}

habana::OutputShapeInfRetType habana::BinaryWrapperOperator::ComputeOutputShape(
    torch::jit::Stack& inputs) {
  OutputShapeInfRetType out;
  std::shared_ptr<HabanaOperator> binaryOp;
  if (inputs[0].isTensor() && inputs[1].isTensor()) { // Both inputs are tensors
    binaryOp = make_operator<BinaryOperator>(
        this->p_context_->device_id_, guid_, this->scalarType_);
    auto binaryOp_out = out.call_ComputeOutputShape(binaryOp, inputs);
    auto out_tensor = binaryOp_out.GetOutputTensor(0);
    out.MoveToOutput(std::move(out_tensor));
  } else if (inputs[0].isTensor() && inputs[1].isScalar()) { // 2nd input is a
                                                             // scalar
    auto arg1 = inputs[0].toTensor();
    // add constant node to convert 2nd input to tensor
    auto constOp = make_operator<ConstantOperator>(
        this->p_context_->device_id_, this->scalarType_);
    auto const_shape_tensor = habana::createPTTensor(
        arg1, {1}, arg1.options(), at::MemoryFormat::Contiguous, false);
    torch::jit::Stack constOp_stack = {IValue(const_shape_tensor), inputs[1]};
    auto constOp_out = out.call_ComputeOutputShape(constOp, constOp_stack);
    auto const_out_tensor = std::get<1>(constOp_out.GetOutputTensor(0));

    // replace input scalar with input tensor in the stack
    auto scalar_input = inputs.back();
    inputs.pop_back();
    inputs.emplace_back(const_out_tensor);
    binaryOp = make_operator<BinaryOperator>(
        this->p_context_->device_id_, guid_, this->scalarType_);
    auto binaryOp_out = out.call_ComputeOutputShape(binaryOp, inputs);
    auto out_tensor = binaryOp_out.GetOutputTensor(0);
    out.MoveToOutput(std::move(out_tensor));
    // revert the stack changes
    if (GET_ENV_FLAG_NEW(PT_HPU_VALIDATE_COMPUTE_SHAPE) ||
        GET_ENV_FLAG_NEW(PT_HPU_ENABLE_FAST_SHAPE_INFERENCE)) {
      inputs.pop_back();
      inputs.emplace_back(scalar_input);
    }
  } else { // 1st input is a scalar
    auto arg2 = inputs[1].toTensor();
    // add constant node to convert 1st input to tensor
    auto constOp = make_operator<ConstantOperator>(
        this->p_context_->device_id_, this->scalarType_);
    auto const_shape_tensor = habana::createPTTensor(
        arg2, {1}, arg2.options(), at::MemoryFormat::Contiguous, false);
    torch::jit::Stack constOp_stack = {IValue(const_shape_tensor), inputs[0]};
    auto constOp_out = out.call_ComputeOutputShape(constOp, constOp_stack);
    auto const_out_tensor = std::get<1>(constOp_out.GetOutputTensor(0));

    // replace input scalar with input tensor in the stack
    auto scalar_input = inputs.front();
    inputs.erase(inputs.cbegin());
    inputs.emplace(inputs.cbegin(), const_out_tensor);
    binaryOp = make_operator<BinaryOperator>(
        this->p_context_->device_id_, guid_, this->scalarType_);
    auto binaryOp_out = out.call_ComputeOutputShape(binaryOp, inputs);
    auto out_tensor = binaryOp_out.GetOutputTensor(0);
    out.MoveToOutput(std::move(out_tensor));
    // revert the stack changes
    if (GET_ENV_FLAG_NEW(PT_HPU_VALIDATE_COMPUTE_SHAPE) ||
        GET_ENV_FLAG_NEW(PT_HPU_ENABLE_FAST_SHAPE_INFERENCE)) {
      inputs.erase(inputs.cbegin());
      inputs.emplace(inputs.cbegin(), scalar_input);
    }
  }
  return out;
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

habana::OutputShapeInfRetType habana::BinaryOperatorWithAlpha::
    ComputeOutputShape(torch::jit::Stack& inputs) {
  Tensor arg1 = inputs[0].toTensor();
  Tensor arg2 = inputs[1].toTensor();

  OutputShapeInfRetType out;
  auto memory_format = at::MemoryFormat::Contiguous;
  if ((arg1.suggest_memory_format() == at::MemoryFormat::ChannelsLast) ||
      (arg2.suggest_memory_format() == at::MemoryFormat::ChannelsLast)) {
    memory_format = at::MemoryFormat::ChannelsLast;
  }

  if (inputs[2].toScalar().toFloat() != 1.0) {
    // Multiplication between arg2 and alpha is required
    auto mulOp = make_operator<habana::MulOperator>(
        this->p_context_->device_id_, this->scalarType_);
    torch::jit::Stack mulOp_stack{inputs[1], inputs[2]};
    // Mul output is pushed in to Syn Input
    (void)out.call_ComputeOutputShape(mulOp, mulOp_stack);
  }
  auto shape_out = BinaryOperator::compute_output_shape(arg1, arg2);
  out.AddOutputTensor(TensorMetaData(
      shape_out,
      HabanaOperator::CalculateStrides(shape_out, memory_format),
      arg1.scalar_type(),
      memory_format));
  return out;
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
    const habana::OutputMetaDataVector& output_metadata) {
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

    mulOp->AllocateAndAddSynapseNode(
        graph, mulOp_stack, habana::OutputMetaDataVector(1));
    synapse_helpers::tensor_or_ref& mulOp_out = mulOp->GetSynOutputs()[0];

    auto out_shape = BinaryOperator::compute_output_shape(arg1, arg2);
    auto output = habana::createPTTensor(
        arg1,
        IntArrayRef(out_shape.data(), out_shape.size()),
        arg1.options(),
        memory_format,
        output_metadata.at(0).persistent);

    AllocateSynapseOutput(graph, output, output_metadata.at(0));

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
        guid_,
        nullptr,
        nullptr,
        nullptr,
        deterministic);
  } else {
    auto out_shape = BinaryOperator::compute_output_shape(arg1, arg2);
    auto output = habana::createPTTensor(
        arg1,
        IntArrayRef(out_shape.data(), out_shape.size()),
        arg1.options(),
        memory_format,
        output_metadata.at(0).persistent);
    AllocateSynapseOutput(graph, output, output_metadata.at(0));

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
        guid_,
        nullptr,
        nullptr,
        nullptr,
        deterministic);
  }
}

habana::OutputShapeInfRetType habana::BinaryWrapperOperatorWithAlpha::
    ComputeOutputShape(torch::jit::Stack& inputs) {
  OutputShapeInfRetType out;
  std::shared_ptr<HabanaOperator> binaryOp;
  if (inputs[0].isTensor() &&
      inputs[1].isTensor()) { // First 2 inputs are both tensors
    binaryOp = make_operator<BinaryOperatorWithAlpha>(
        this->p_context_->device_id_, guid_, this->scalarType_);
    auto binaryOp_out = out.call_ComputeOutputShape(binaryOp, inputs);
    auto out_tensor = binaryOp_out.GetOutputTensor(0);
    out.MoveToOutput(std::move(out_tensor));
  } else if (inputs[0].isTensor() && inputs[1].isScalar()) { // 2nd input is a
                                                             // scalar
    auto arg1 = inputs[0].toTensor();
    // add constant node to convert 2nd input to tensor
    auto constOp = make_operator<ConstantOperator>(
        this->p_context_->device_id_, this->scalarType_);
    auto const_shape_tensor = habana::createPTTensor(
        arg1, {1}, arg1.options(), at::MemoryFormat::Contiguous, false);
    torch::jit::Stack constOp_stack = {IValue(const_shape_tensor), inputs[1]};
    auto constOp_out = out.call_ComputeOutputShape(constOp, constOp_stack);
    auto const_out_tensor = std::get<1>(constOp_out.GetOutputTensor(0));

    // replace 2nd scalar input with a tensor in stack
    auto scalar_input = inputs.at(1);
    inputs.erase(inputs.cbegin() + 1);
    inputs.emplace(inputs.cbegin() + 1, const_out_tensor);
    binaryOp = make_operator<BinaryOperatorWithAlpha>(
        this->p_context_->device_id_, guid_, this->scalarType_);
    auto binaryOp_out = out.call_ComputeOutputShape(binaryOp, inputs);
    auto out_tensor = binaryOp_out.GetOutputTensor(0);
    out.MoveToOutput(std::move(out_tensor));
    // revert the stack changes
    if (GET_ENV_FLAG_NEW(PT_HPU_VALIDATE_COMPUTE_SHAPE) ||
        GET_ENV_FLAG_NEW(PT_HPU_ENABLE_FAST_SHAPE_INFERENCE)) {
      inputs.erase(inputs.cbegin() + 1);
      inputs.emplace(inputs.cbegin() + 1, scalar_input);
    }
  } else { // 1st input is a scalar
    auto arg2 = inputs[1].toTensor();
    // add constant node to convert 1st input to tensor
    auto constOp = make_operator<ConstantOperator>(
        this->p_context_->device_id_, this->scalarType_);
    auto const_shape_tensor = habana::createPTTensor(
        arg2, {1}, arg2.options(), at::MemoryFormat::Contiguous, false);
    torch::jit::Stack constOp_stack = {IValue(const_shape_tensor), inputs[0]};
    auto constOp_out = out.call_ComputeOutputShape(constOp, constOp_stack);
    auto const_out_tensor = std::get<1>(constOp_out.GetOutputTensor(0));

    // replace 1st scalar input with a tensor in stack
    auto scalar_input = inputs.front();
    inputs.erase(inputs.cbegin());
    inputs.emplace(inputs.cbegin(), const_out_tensor);
    binaryOp = make_operator<BinaryOperatorWithAlpha>(
        this->p_context_->device_id_, guid_, this->scalarType_);
    auto binaryOp_out = out.call_ComputeOutputShape(binaryOp, inputs);
    auto out_tensor = binaryOp_out.GetOutputTensor(0);
    out.MoveToOutput(std::move(out_tensor));
    // revert the stack changes
    if (GET_ENV_FLAG_NEW(PT_HPU_VALIDATE_COMPUTE_SHAPE) ||
        GET_ENV_FLAG_NEW(PT_HPU_ENABLE_FAST_SHAPE_INFERENCE)) {
      inputs.erase(inputs.cbegin());
      inputs.emplace(inputs.cbegin(), scalar_input);
    }
  }
  return out;
}

/************************************************************************
 * @brief This function implements synapse node addition for Binary OPs
 * with 3 input arguments (where 1st input is always a tensor whereas
 * 2nd and 3rd inputs can be a tensor or a scalar)
 ************************************************************************/
void habana::BinaryWrapperOperatorWithAlpha::AllocateAndAddSynapseNode(
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

  std::shared_ptr<HabanaOperator> binaryOp;
  if (inputs[0].isTensor() &&
      inputs[1].isTensor()) { // First 2 inputs are both tensors
    binaryOp = make_operator<BinaryOperatorWithAlpha>(
        this->p_context_->device_id_, guid_, this->scalarType_);
    binaryOp->SetSynapseInput(p_context_->syn_inputs_[0]);
    binaryOp->SetSynapseInput(p_context_->syn_inputs_[1]);
    binaryOp->AllocateAndAddSynapseNode(graph, inputs, output_metadata);
  } else if (inputs[0].isTensor() && inputs[1].isScalar()) { // 2nd input is a
                                                             // scalar
    auto arg1 = inputs[0].toTensor();
    // add node to convert scalar to tensor
    auto constOp = make_operator<ConstantOperator>(
        this->p_context_->device_id_, this->scalarType_);
    auto const_shape_tensor = habana::createPTTensor(
        arg1, {1}, arg1.options(), at::MemoryFormat::Contiguous, false);
    torch::jit::Stack constOp_stack = {IValue(const_shape_tensor), inputs[1]};
    constOp->AllocateAndAddSynapseNode(
        graph, constOp_stack, habana::OutputMetaDataVector(1));

    binaryOp = make_operator<BinaryOperatorWithAlpha>(
        this->p_context_->device_id_, guid_, this->scalarType_);
    binaryOp->SetSynapseInput(p_context_->syn_inputs_[0]);
    binaryOp->SetSynapseInput(constOp->GetSynOutputs()[0]);
    // replace 2nd scalar input with a tensor in stack
    auto scalar_input = inputs.at(1);
    inputs.erase(inputs.cbegin() + 1);
    inputs.emplace(inputs.cbegin() + 1, constOp->GetOutputs()[0]);
    binaryOp->AllocateAndAddSynapseNode(graph, inputs, output_metadata);
    // revert the stack changes
    if (GET_ENV_FLAG_NEW(PT_HPU_VALIDATE_COMPUTE_SHAPE) ||
        GET_ENV_FLAG_NEW(PT_HPU_ENABLE_FAST_SHAPE_INFERENCE)) {
      inputs.erase(inputs.cbegin() + 1);
      inputs.emplace(inputs.cbegin() + 1, scalar_input);
    }
  } else { // 1st input is a scalar
    auto arg2 = inputs[1].toTensor();
    // add node to convert scalar to tensor
    auto constOp = make_operator<ConstantOperator>(
        this->p_context_->device_id_, this->scalarType_);
    auto const_shape_tensor = habana::createPTTensor(
        arg2, {1}, arg2.options(), at::MemoryFormat::Contiguous, false);
    torch::jit::Stack constOp_stack = {IValue(const_shape_tensor), inputs[0]};
    constOp->AllocateAndAddSynapseNode(
        graph, constOp_stack, habana::OutputMetaDataVector(1));

    binaryOp = make_operator<BinaryOperatorWithAlpha>(
        this->p_context_->device_id_, guid_, this->scalarType_);
    binaryOp->SetSynapseInput(constOp->GetSynOutputs()[0]);
    binaryOp->SetSynapseInput(p_context_->syn_inputs_[0]);

    // replace 1st scalar input with a tensor in stack
    auto scalar_input = inputs.front();
    inputs.erase(inputs.cbegin());
    inputs.emplace(inputs.cbegin(), constOp->GetOutputs()[0]);
    binaryOp->AllocateAndAddSynapseNode(graph, inputs, output_metadata);
    // revert the stack changes
    if (GET_ENV_FLAG_NEW(PT_HPU_VALIDATE_COMPUTE_SHAPE) ||
        GET_ENV_FLAG_NEW(PT_HPU_ENABLE_FAST_SHAPE_INFERENCE)) {
      inputs.erase(inputs.cbegin());
      inputs.emplace(inputs.cbegin(), scalar_input);
    }
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
      if (habana_helpers::is_downcast_to_int_needed(
              stack[i].toTensor().scalar_type())) {
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
    Op.Execute(key, pt_inputs, stack);
  } else {
    habana::OutputMetaDataVector output_metadata(1);
    output_metadata.at(0).persistent = true;
    // compile and execute the graph
    Op.CreateGraphAndCompile(key, pt_inputs, stack, output_metadata, true);
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
Tensor add_tensor_hpu(
    const Tensor& self,
    const Tensor& other,
    const Scalar& alpha) {
  PT_KERNEL_BEGIN;
  Tensor output;
  if (self.dim() == 0) {
    SET_SIZE_STRIDE_1D(self);
  }
  if (other.dim() == 0) {
    SET_SIZE_STRIDE_1D(other);
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
Tensor add_scalar_hpu(
    const Tensor& self,
    const Scalar& other,
    const Scalar& alpha) {
  PT_KERNEL_BEGIN;

  if (self.dim() == 0) {
    SET_SIZE_STRIDE_1D(self);
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
Tensor sub_tensor_hpu(
    const Tensor& self,
    const Tensor& other,
    const Scalar& alpha) {
  PT_KERNEL_BEGIN;

  if (self.dim() == 0) {
    SET_SIZE_STRIDE_1D(self);
  }
  if (other.dim() == 0) {
    SET_SIZE_STRIDE_1D(other);
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
    SET_SIZE_STRIDE_1D(self);
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
    const habana::OutputMetaDataVector& output_metadata) {
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
  SubOperator::AllocateAndAddSynapseNode(graph, inputs, output_metadata);
}

/*************************************************************************
 * @brief Kernel implementation for rsub Scalar torch.rsub(self, other,
 *  alpha)
 * @param self - first input tensor, 1-4D, FP32/BF16
 * @param other - second input Scalar
 * @param alpha - optional input Scalar, default = 1
 * output = other - self * alpha
 ************************************************************************/
Tensor rsub_scalar_hpu(
    const Tensor& self,
    const Scalar& other,
    const Scalar& alpha) {
  PT_KERNEL_BEGIN;

  if (self.dim() == 0) {
    SET_SIZE_STRIDE_1D(self);
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
    SET_SIZE_STRIDE_1D(self);
  }
  if (other.dim() == 0) {
    SET_SIZE_STRIDE_1D(other);
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
    SET_SIZE_STRIDE_1D(self);
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
    SET_SIZE_STRIDE_1D(self);
  }
  if (other.dim() == 0) {
    SET_SIZE_STRIDE_1D(other);
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
    const Scalar&
        other) { // TODO: Add test by using an extension module for new op
  // at python level
  PT_KERNEL_BEGIN;

  if (self.dim() == 0) {
    SET_SIZE_STRIDE_1D(self);
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
    SET_SIZE_STRIDE_1D(self);
  }
  if (other.dim() == 0) {
    SET_SIZE_STRIDE_1D(other);
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
    SET_SIZE_STRIDE_1D(self);
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
    SET_SIZE_STRIDE_1D(self);
  }
  auto self_hpu = get_hpu_tensor(self);
  std::vector<at::Tensor> pt_inputs{self_hpu};
  torch::jit::Stack stack{IValue(other), IValue(self_hpu)};
  auto output = process_generic_tensor_binary_op<habana::PowOperator>(
      pt_inputs, stack, "pow");

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
    SET_SIZE_STRIDE_1D(self);
  }
  if (other.dim() == 0) {
    SET_SIZE_STRIDE_1D(other);
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
    SET_SIZE_STRIDE_1D(self);
  }
  if (other.dim() == 0) {
    SET_SIZE_STRIDE_1D(other);
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
    remainder = habana::createPTTensor(
        self,
        IntArrayRef(out_shape.data(), out_shape.size()),
        self.options(),
        self.suggest_memory_format(),
        true);
  } else {
    auto self = inputs[0].toTensor();
    remainder = habana::createPTTensor(self, true);
  }
  std::vector<at::Tensor> v{remainder};
  HabanaOperator::SetPTOutputs(v);
}

void habana::RemainderWrapperOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    const habana::OutputMetaDataVector& output_metadata) {
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
    remainderOp->AllocateAndAddSynapseNode(graph, inputs, output_metadata);
  } else { // 2nd input is a scalar
    // add constant node to convert 2nd input to tensor
    auto arg1 = inputs[0].toTensor();
    auto constOp = make_operator<ConstantOperator>(
        this->p_context_->device_id_, this->scalarType_);
    auto const_shape_tensor = habana::createPTTensor(
        arg1, {1}, arg1.options(), at::MemoryFormat::Contiguous, false);
    torch::jit::Stack constOp_stack = {IValue(const_shape_tensor), inputs[1]};
    constOp->AllocateAndAddSynapseNode(
        graph, constOp_stack, habana::OutputMetaDataVector(1));
    remainderOp->SetSynapseInput(p_context_->syn_inputs_[0]);
    remainderOp->SetSynapseInput(constOp->GetSynOutputs()[0]);
    // replace 2nd scalar input with a tensor in stack
    inputs.erase(inputs.cbegin() + 1);
    inputs.emplace(inputs.cbegin() + 1, constOp->GetOutputs()[0]);
    remainderOp->AllocateAndAddSynapseNode(graph, inputs, output_metadata);
  }

  p_context_->pt_outputs_.emplace_back(remainderOp->GetOutputs()[1]);
  p_context_->syn_outputs_.emplace_back(
      std::move(remainderOp->GetSynOutputs()[1]));
}

void habana::RemainderOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    const habana::OutputMetaDataVector& output_metadata) {
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
  auto quotient = habana::createPTTensor(
      self,
      IntArrayRef(out_shape.data(), out_shape.size()),
      self.options(),
      self.suggest_memory_format(),
      false);
  auto remainder = habana::createPTTensor(
      self,
      IntArrayRef(out_shape.data(), out_shape.size()),
      self.options(),
      self.suggest_memory_format(),
      output_metadata.at(0).persistent);
  AllocateSynapseOutputs(
      graph,
      {quotient, remainder},
      {habana::OutputMetaData(), output_metadata.at(0)});
  AddNodeToSynapseGraph(graph, &params, sizeof(params));
}

Tensor remainder_tensor_hpu(const Tensor& self, const Tensor& other) {
  PT_KERNEL_BEGIN;

  bool isSelf_0d = false;
  bool isOther_0d = false;
  if (self.dim() == 0) {
    SET_SIZE_STRIDE_1D(self);
    isSelf_0d = true;
  }
  if (other.dim() == 0) {
    SET_SIZE_STRIDE_1D(other);
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
    Op.Execute(key, pt_inputs, stack);
  } else {
    // Build Params for the graph
    habana::OutputMetaDataVector output_metadata(1);
    output_metadata.at(0).persistent = true;
    // compile and execute the graph
    Op.CreateGraphAndCompile(key, pt_inputs, stack, output_metadata, true);
  }
  std::vector<at::Tensor> out = Op.GetOutputs();
  TORCH_CHECK(out.size() == 1, "Incorrect size of outputs");

  auto output = out.at(0);
  if (isSelf_0d && isOther_0d) {
    SET_SIZE_STRIDE_0D(output);
  }
  if (isSelf_0d) {
    SET_SIZE_STRIDE_0D(self);
  }
  if (isOther_0d) {
    SET_SIZE_STRIDE_0D(other);
  }

  PT_KERNEL_END;
  return output;
}

void habana::RemainderInplaceOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    const habana::OutputMetaDataVector& output_metadata) {
  // this check is for stack during graph execution
  TORCH_CHECK(
      inputs.size() == 2,
      "Incorrect size of input expected for Binary operator");
  TORCH_CHECK(inputs[0].isTensor(), "Input type expected to be tensor");
  TORCH_CHECK(inputs[1].isTensor(), "Input type expected to be tensor");
  Tensor self = inputs[0].toTensor();
  Tensor other = inputs[1].toTensor();

  auto out_shape = BinaryOperator::compute_output_shape(self, other);
  auto quotient = habana::createPTTensor(
      self,
      IntArrayRef(out_shape.data(), out_shape.size()),
      self.options(),
      self.suggest_memory_format(),
      output_metadata.at(0).persistent);
  AllocateSynapseOutput(graph, quotient, output_metadata.at(0), false);

  // Note here we are using input[0] to store output[1]
  p_context_->syn_outputs_.emplace_back(
      habana_helpers::duplicate_tensor_in_memory_section(
          p_context_->syn_inputs_[0], graph, output_metadata.at(0).external));
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
      std::move(this->guid_),
      nullptr,
      nullptr,
      nullptr,
      deterministic);
}

void habana::RemainderInplaceWrapperOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    const habana::OutputMetaDataVector& output_metadata) {
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
        graph, inputs, {OutputMetaData(), output_metadata.at(0)});
  } else { // 2nd input is a scalar
    // add constant node to convert 2nd input to tensor
    auto arg1 = inputs[0].toTensor();
    auto constOp = make_operator<ConstantOperator>(
        this->p_context_->device_id_, this->scalarType_);
    auto const_shape_tensor = habana::createPTTensor(
        arg1, {1}, arg1.options(), at::MemoryFormat::Contiguous, false);
    torch::jit::Stack constOp_stack = {IValue(const_shape_tensor), inputs[1]};
    constOp->AllocateAndAddSynapseNode(
        graph, constOp_stack, habana::OutputMetaDataVector(1));
    remainderOp->SetSynapseInput(p_context_->syn_inputs_[0]);
    remainderOp->SetSynapseInput(constOp->GetSynOutputs()[0]);
    // replace 2nd scalar input with a tensor in stack
    inputs.erase(inputs.cbegin() + 1);
    inputs.emplace(inputs.cbegin() + 1, constOp->GetOutputs()[0]);
    remainderOp->AllocateAndAddSynapseNode(
        graph, inputs, {OutputMetaData(), output_metadata.at(0)});
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
    SET_SIZE_STRIDE_1D(self);
    isSelf_0d = true;
  }
  if (other.dim() == 0) {
    SET_SIZE_STRIDE_1D(other);
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
    Op.Execute(key, pt_inputs, self);
  } else {
    // Build Params for the graph
    habana::OutputMetaDataVector output_metadata(1);
    output_metadata.at(0).persistent = true;
    // compile and execute the graph
    Op.CreateGraphAndCompile(key, pt_inputs, stack, output_metadata, true);
  }
  std::vector<at::Tensor> out = Op.GetOutputs();
  TORCH_CHECK(out.size() == 1, "Incorrect size of outputs");

  if (isSelf_0d && isOther_0d) {
    SET_SIZE_STRIDE_0D(self);
  }
  if (isOther_0d) {
    SET_SIZE_STRIDE_0D(other);
  }
  PT_KERNEL_END;

  return self;
}

Tensor remainder_scalar_hpu(const Tensor& self, const at::Scalar& other) {
  PT_KERNEL_BEGIN;
  bool isSelf_0d = false;
  if (self.dim() == 0) {
    SET_SIZE_STRIDE_1D(self);
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
    Op.Execute(key, pt_inputs, stack);
  } else {
    // Build Params for the graph
    habana::OutputMetaDataVector output_metadata(1);
    output_metadata.at(0).persistent = true;
    // compile and execute the graph
    Op.CreateGraphAndCompile(key, pt_inputs, stack, output_metadata, true);
  }
  std::vector<at::Tensor> out = Op.GetOutputs();
  TORCH_CHECK(out.size() == 1, "Incorrect size of outputs");

  auto output = out.at(0);
  if (isSelf_0d) {
    SET_SIZE_STRIDE_0D(self);
    SET_SIZE_STRIDE_0D(output);
  }
  PT_KERNEL_END;
  return output;
}

Tensor& remainder_scalar_hpu_(Tensor& self, const at::Scalar& other) {
  PT_KERNEL_BEGIN;

  bool isSelf_0d = false;
  if (self.dim() == 0) {
    SET_SIZE_STRIDE_1D(self);
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
    Op.Execute(key, pt_inputs, self);
  } else {
    // Build Params for the graph
    habana::OutputMetaDataVector output_metadata(1);
    output_metadata.at(0).persistent = true;
    // compile and execute the graph
    Op.CreateGraphAndCompile(key, pt_inputs, stack, output_metadata, true);
  }
  std::vector<at::Tensor> out = Op.GetOutputs();
  TORCH_CHECK(out.size() == 1, "Incorrect size of outputs");

  if (isSelf_0d) {
    SET_SIZE_STRIDE_0D(self);
  }
  PT_KERNEL_END;

  return self;
}

void habana::RemainderOutOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    const habana::OutputMetaDataVector& output_metadata) {
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
  auto quotient = habana::createPTTensor(
      self,
      IntArrayRef(out_shape.data(), out_shape.size()),
      self.options(),
      self.suggest_memory_format(),
      false);

  AllocateSynapseOutput(graph, quotient, OutputMetaData());
  synapse_helpers::tensor& arg1_syn_tensor = p_context_->syn_inputs_[0];
  synapse_helpers::tensor& arg2_syn_tensor = p_context_->syn_inputs_[1];

  p_context_->syn_outputs_.emplace_back(
      habana_helpers::duplicate_tensor_in_memory_section(
          p_context_->syn_inputs_[2], graph, output_metadata.at(0).external));
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
      guid_,
      nullptr,
      nullptr,
      nullptr,
      deterministic);
}

void habana::RemainderOutWrapperOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    const habana::OutputMetaDataVector& output_metadata) {
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
    remainderOp->AllocateAndAddSynapseNode(graph, inputs, output_metadata);
  } else { // 2nd input is a scalar
    // add constant node to convert 2nd input to tensor
    auto arg1 = inputs[0].toTensor();
    auto constOp = make_operator<ConstantOperator>(
        this->p_context_->device_id_, this->scalarType_);
    auto const_shape_tensor = habana::createPTTensor(
        arg1, {1}, arg1.options(), at::MemoryFormat::Contiguous, false);
    torch::jit::Stack constOp_stack = {IValue(const_shape_tensor), inputs[1]};
    constOp->AllocateAndAddSynapseNode(
        graph, constOp_stack, habana::OutputMetaDataVector(1));
    remainderOp->SetSynapseInput(p_context_->syn_inputs_[0]);
    remainderOp->SetSynapseInput(constOp->GetSynOutputs()[0]);
    remainderOp->SetSynapseInput(p_context_->syn_inputs_[1]);
    // replace 2nd scalar input with a tensor in stack
    inputs.erase(inputs.cbegin() + 1);
    inputs.emplace(inputs.cbegin() + 1, constOp->GetOutputs()[0]);
    remainderOp->AllocateAndAddSynapseNode(graph, inputs, output_metadata);
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
    SET_SIZE_STRIDE_1D(self);
    isSelf_0d = true;
  }
  if (other.dim() == 0) {
    SET_SIZE_STRIDE_1D(other);
    isOther_0d = true;
  }
  if (result.dim() == 0) {
    SET_SIZE_STRIDE_1D(result);
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
    Op.Execute(key, pt_inputs, result);
  } else {
    // Build Params for the graph
    habana::OutputMetaDataVector output_metadata(1);
    output_metadata.at(0).persistent = true;
    // compile and execute the graph
    Op.CreateGraphAndCompile(key, pt_inputs, stack, output_metadata, true);
  }

  std::vector<at::Tensor> out = Op.GetOutputs();
  TORCH_CHECK(out.size() == 1, "Incorrect size of outputs");

  if (isSelf_0d && isOther_0d) {
    SET_SIZE_STRIDE_0D(result);
  }
  if (isSelf_0d) {
    SET_SIZE_STRIDE_0D(self);
  }
  if (isOther_0d) {
    SET_SIZE_STRIDE_0D(other);
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
    SET_SIZE_STRIDE_1D(self);
    isSelf_0d = true;
  }
  if (result.dim() == 0) {
    SET_SIZE_STRIDE_1D(result);
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
    Op.Execute(key, pt_inputs, result);
  } else {
    // Build Params for the graph
    habana::OutputMetaDataVector output_metadata(1);
    output_metadata.at(0).persistent = true;
    // compile and execute the graph
    Op.CreateGraphAndCompile(key, pt_inputs, stack, output_metadata, true);
  }

  std::vector<at::Tensor> out = Op.GetOutputs();
  TORCH_CHECK(out.size() == 1, "Incorrect size of outputs");

  if (isSelf_0d) {
    SET_SIZE_STRIDE_0D(self);
    SET_SIZE_STRIDE_0D(result);
  }
  PT_KERNEL_END;
  return result;
}

static auto& BinaryKernelsKernelRegistry =
    habana::KernelRegistry()
        .add("aten::add.Tensor", KERNEL_FN(AddOperator))
        .add("aten::add.Scalar", KERNEL_FN(AddOperator))
        .add("hpu::rsub.Tensor", KERNEL_FN(RsubOperator));
