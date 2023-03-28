/*******************************************************************************
 * Copyright (C) 2020-2023 Habana Labs, Ltd. an Intel Company
 * All Rights Reserved.
 *
 * Unauthorized copying of this file or any element(s) within it, via any medium
 * is strictly prohibited.
 * This file contains Habana Labs, Ltd. proprietary and confidential information
 * and is subject to the confidentiality and license agreements under which it
 * was provided.
 *
 *******************************************************************************
 */
#include <ATen/InferSize.h>
#include <perf_lib_layer_params.h>
#include <synapse_api.h>
#include <torch/script.h>

#include "backend/create_pt_tensor.h"
#include "backend/helpers/create_tensor.h"
#include "backend/helpers/graph.h"
#include "backend/helpers/tensor_utils.h"
#include "backend/kernel/hpu_shape_inference.h"
#include "habana_device/HPUCheck.h"
#include "habana_device/hpu_cached_devices.h"
#include "habana_helpers/frontend_utils.h"
#include "habana_helpers/logging_pt.h"
#include "habana_kernels/basic_kernels.h"
#include "habana_kernels/embedding_kernels.h"
#include "habana_kernels/index_kernels.h"
#include "habana_kernels/tensor_shape_kernels.h"
#include "habana_kernels/topk_kernels.h"
#include "habana_lazy/aten_lazy_bridge.h"
#include "habana_lazy/tensor_impl.h"
#include "kernel_utils.h"

using namespace torch;
using namespace habana;

std::vector<int64_t> PadOperator::compute_output_shape(
    const at::Tensor& self,
    c10::IntArrayRef pad) {
  auto ndim = self.dim();
  auto lpad = pad.size() / 2;

  TORCH_CHECK(
      pad.size() % 2 == 0,
      "Length of pad must be even but instead it equals ",
      pad.size());

  TORCH_CHECK(
      ndim >= (int64_t)lpad,
      "Length of pad should be no more than twice the number of "
      "dimensions of the input. Pad length is ",
      pad.size(),
      "while the input has ",
      ndim,
      "dimensions.");

  auto shape = self.sizes().vec();

  for (unsigned int i = 0; i < lpad; i++) {
    auto pad_start = pad[2 * i];
    auto pad_end = pad[2 * i + 1];
    shape[ndim - i - 1] += (pad_start + pad_end);
    TORCH_CHECK(
        shape[ndim - i - 1] > 0,
        "The input size ",
        self.sizes()[i],
        ", plus negative padding ",
        pad_start,
        " and ",
        pad_end,
        " resulted in a invalid output size, "
        "Check dimension ",
        i,
        " of your input.");
  }
  return shape;
}

std::vector<int64_t> PadOperator::compute_output_shape_ds(
    const at::Tensor& self,
    c10::IntArrayRef pad_before,
    c10::IntArrayRef pad_after) {
  auto ndim = self.dim();

  auto shape = self.sizes().vec();

  for (unsigned int i = 0; i < ndim; i++) {
    auto pad_start = pad_before[MAX_DIMENSIONS_NUM - i - 1];
    auto pad_end = pad_after[MAX_DIMENSIONS_NUM - i - 1];
    shape[ndim - i - 1] += (pad_start + pad_end);
    TORCH_CHECK(
        shape[ndim - i - 1] > 0,
        "The input size ",
        self.sizes()[i],
        ", plus negative padding ",
        pad_start,
        " and ",
        pad_end,
        " resulted in a invalid output size, "
        "Check dimension ",
        i,
        " of your input.");
  }
  return shape;
}

void PadOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    const OutputMetaDataVector& output_metadata) {
  TORCH_CHECK(
      inputs.size() == 4 || inputs.size() == 3,
      "Incorrect size of inputs expected for PadOperator Operator");
  TORCH_CHECK(
      inputs[0].isTensor(),
      "Input arg1 expected to be Tensor for PadOperator Operator");
  TORCH_CHECK(
      inputs[1].isTensor() || inputs[1].isIntList(),
      "Input arg1 expected to be of type Int or Tensor for PadOperator operator");
  auto pad = inputs[1].isIntList() ? inputs[1].toIntVector()
                                   : inputs[1].toTensor().sizes().vec();
  bool have_shape_tensor = inputs[1].isTensor();
  std::vector<int64_t> shape;
  auto self = inputs[0].toTensor();
  if (have_shape_tensor) {
    TORCH_CHECK(
        p_context_->syn_inputs_[1].ref().is_input_shape_tensor(),
        "Synapse input1 type expected to be shape tensor");
    TORCH_CHECK(
        p_context_->syn_inputs_[2].ref().is_input_shape_tensor(),
        "Synapse input2 type expected to be shape tensor");

    shape = compute_output_shape_ds(
        self,
        inputs[1].toTensor().sizes().vec(),
        inputs[2].toTensor().sizes().vec());
  } else {
    shape = compute_output_shape(self, pad);
  }
  ns_PadKernelEx::Params param;
  auto output = at::empty(shape, self.options());

  if (have_shape_tensor) {
    param.mode = PadMode_t::PAD_MODE_CONSTANT;
    if (self.scalar_type() == c10::ScalarType::Int) {
      param.value.i = inputs[3].toScalar().to<int>();
    } else {
      param.value.f = inputs[3].toScalar().to<float>();
    }
    TORCH_CHECK(p_context_->syn_inputs_.back().ref().is_input_shape_tensor());
  } else {
    auto ndim = self.dim();
    auto lpad = pad.size() / 2;

    param.mode = PadMode_t::PAD_MODE_CONSTANT;
    if (self.scalar_type() == c10::ScalarType::Int) {
      param.value.i = inputs[2].toScalar().to<int>();
    } else {
      param.value.f = inputs[2].toScalar().to<float>();
    }
    memset(param.pads, 0, sizeof(param.pads));
    for (unsigned int i = 0; i < lpad; i++) {
      param.pads[i] = pad[2 * i];
      param.pads[i + ndim] = pad[2 * i + 1];
    }
    // Allocate Shape tensor
    if (graph.is_dynamic_graph()) {
      std::vector<int64_t> pad_before(MAX_DIMENSIONS_NUM);
      std::vector<int64_t> pad_after(MAX_DIMENSIONS_NUM);

      for (unsigned int i = 0; i < pad.size() / 2; i++) {
        pad_before[MAX_DIMENSIONS_NUM - i - 1] = pad[2 * i];
        pad_after[MAX_DIMENSIONS_NUM - i - 1] = pad[2 * i + 1];
      }
      auto pad_before_tensor = at::empty(
          IntArrayRef(pad_before.data(), pad_before.size()),
          self.options().dtype(c10::ScalarType::Int));
      auto pad_after_tensor = at::empty(
          IntArrayRef(pad_after.data(), pad_after.size()),
          self.options().dtype(c10::ScalarType::Int));
      AllocateSynapseShapeTensor(
          graph, pad_before_tensor, INPUT_DESCRIBING_SHAPE_TENSOR);
      AllocateSynapseShapeTensor(
          graph, pad_after_tensor, INPUT_DESCRIBING_SHAPE_TENSOR);
    }
  }

  AllocateSynapseOutput(graph, output, output_metadata.at(0));
  AddNodeToSynapseGraph(graph, &param, sizeof(param));
}

OutputShapeInfRetType PadOperatorHT::ComputeOutputShape(
    torch::jit::Stack& inputs) {
  std::vector<int64_t> shape;
  auto input = inputs[0].toTensor();
  shape = inputs[2].toTensor().sizes().vec();

  OutputShapeInfRetType out;
  out.AddOutputTensor(TensorMetaData(
      shape,
      HabanaOperator::CalculateStrides(shape, input.suggest_memory_format()),
      input.scalar_type(),
      input.suggest_memory_format()));
  return out;
}
void PadOperatorHT::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    const OutputMetaDataVector& output_metadata) {
  TORCH_CHECK(
      inputs.size() == 4,
      "Incorrect size of inputs expected for PadOperatorHT Operator");
  TORCH_CHECK(
      inputs[0].isTensor(),
      "Input arg1 expected to be Tensor for PadOperatorHT Operator");
  TORCH_CHECK(
      inputs[1].isTensor(),
      "Input arg2 expected to be of type Tensor for PadOperatorHT operator");
  TORCH_CHECK(
      inputs[2].isTensor(),
      "Input arg3 expected to be of type Tensor for PadOperatorHT operator");
  TORCH_CHECK(
      inputs[3].isScalar(),
      "Input arg4 expected to be of type Scalar for PadOperatorHT operator");

  std::vector<int64_t> shape;
  auto self = inputs[0].toTensor();

  TORCH_CHECK(p_context_->syn_inputs_[1].ref().is_host_to_device_tensor());
  shape = inputs[2].toTensor().sizes().vec();
  at::Tensor host_tensor = inputs[1].toTensor();
  auto impl = habana_lazy::GetHbInternalTensorImpl(host_tensor);
  HABANA_ASSERT(impl);
  auto output_shape = inputs[2].toTensor().sizes().vec();
  auto input_shape = self.sizes().vec();
  TORCH_CHECK(
      impl->get_host_dt_type() == habana_lazy::HostDataType::UINT32_T,
      "Incorrect datatype of HOST");
  if (habana::ShapeInference::GetCurrentPass() ==
      habana::ShapeInfo::InferencePass::MIN_SHAPE) {
    auto ndim = self.dim();
    auto in_data = self.sizes().vec();
    std::vector<uint32_t> data(MAX_DIMENSIONS_NUM * 2, 0);
    for (unsigned int i = 0; i < ndim; i++) {
      // order of dims is reversed in H2D tensor
      data[ndim - i - 1] = output_shape[i] - input_shape[i];
    }
    impl->set_min<uint32_t>(data);
  } else if (
      habana::ShapeInference::GetCurrentPass() ==
      habana::ShapeInfo::InferencePass::MAX_SHAPE) {
    auto ndim = self.dim();
    auto in_data = self.sizes().vec();
    std::vector<uint32_t> data(MAX_DIMENSIONS_NUM * 2, 0);
    for (unsigned int i = 0; i < ndim; i++) {
      // order of dims is reversed in H2D tensor
      data[ndim - i - 1] = output_shape[i] - input_shape[i];
    }
    impl->set_max<uint32_t>(data);
  }

  ns_PadKernelEx::Params param;
  param.mode = PadMode_t::PAD_MODE_CONSTANT;
  if (self.scalar_type() == c10::ScalarType::Int) {
    param.value.i = inputs[3].toScalar().to<int>();
  } else {
    param.value.f = inputs[3].toScalar().to<float>();
  }
  // pads value shall be picked from H2D tensor, set this to 0's to be safe
  memset(param.pads, 0, sizeof(param.pads));
  auto output = at::empty(shape, self.options());
  // throw away shape tensor before adding synapse node
  p_context_->syn_inputs_.pop_back();
  AllocateSynapseOutput(graph, output, output_metadata.at(0));
  AddNodeToSynapseGraph(graph, &param, sizeof(param));
}

/** @brief Function implementing torch.nn.functional.pad(input, pad,
 * mode='constant', value=0)
 *  @param self N-dimensional input tensor
 *  @param pad m-elements tuple, where m/2 ≤ input dimensions and m is even
 *  @param value fill value for "constant" padding
 */
Tensor constant_pad_hpu(const Tensor& self, IntArrayRef pad, Scalar value) {
  PT_KERNEL_BEGIN;

  at::ScalarType scalar_type = self.scalar_type();
  std::string node_type =
      "pad_fwd_" + habana_helpers::name_suffix_from_type(scalar_type);

  size_t device_id = self.device().index();

  PadOperator Op(device_id, scalar_type);
  // Create Graph
  auto graph = habana_helpers::create_graph(device_id, node_type);

  // Assign Inputs to the Operator
  std::vector<at::Tensor> pt_inputs{self};
  Op.AllocateSynapseInputs(graph, pt_inputs, true);

  // Build Params for the graph
  std::vector<c10::IValue> stack = {IValue(self), IValue(pad), IValue(value)};
  OutputMetaDataVector output_metadata(1);
  output_metadata.at(0).persistent = true;
  Op.AllocateAndAddSynapseNode(graph, stack, output_metadata);

  // compile and execute the graph
  Op.Compile(graph);

  std::vector<at::Tensor> out = Op.GetOutputs();
  TORCH_CHECK(out.size() == 1, "Incorrect size of outputs");

  PT_KERNEL_END;
  return out.at(0);
}

void EmbeddingDenseBackwardOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    const OutputMetaDataVector& output_metadata) {
  TORCH_CHECK(
      inputs.size() == 5,
      "Incorrect size of inputs expected for embedding operator");
  TORCH_CHECK(
      inputs[0].isTensor(),
      "Input arg1 type expected to be tensor for embedding operator");
  TORCH_CHECK(
      inputs[1].isTensor(),
      "Input arg2 type expected to be tensor for embedding operator");
  TORCH_CHECK(
      inputs[2].isInt(),
      "Input arg3 type expected to be Int for embedding operator");
  TORCH_CHECK(
      inputs[3].isInt(),
      "Input arg4 type expected to be Int for embedding operator");
  TORCH_CHECK(
      inputs[4].isBool(),
      "Input arg5 type expected to be Bool for embedding operator");

  auto grad = inputs[0].toTensor();
  auto indices = inputs[1].toTensor();
  auto num_weights = inputs[2].toInt();
  [[maybe_unused]] auto padding_idx = inputs[3].toInt();
  auto scale_grad_by_freq = inputs[4].toBool();
  int64_t numel = indices.numel();
  TORCH_CHECK(
      scale_grad_by_freq == false, "scale_grad_by_freq = true not supported")

  // create a wrapper PT tensor for the non-persistent tensor for zero filling
  Tensor grad_temp = habana::createPTTensor(
      grad,
      {num_weights, grad.size(-1)},
      grad.options(),
      grad.suggest_memory_format(),
      grad.scalar_type(),
      false);
  auto zeroOp = make_operator<ConstantOutOperator>(
      this->p_context_->device_id_, grad.scalar_type());
  zeroOp->SetPTInputs({grad_temp});
  zeroOp->AllocateSynapseInput(graph, grad_temp, false);
  c10::Scalar zero_val = 0;
  std::vector<c10::IValue> zero_op_stack{IValue(grad_temp), IValue(zero_val)};
  zeroOp->AllocateAndAddSynapseNode(
      graph, zero_op_stack, OutputMetaDataVector(1));

  // Node: indices_flattened = indices.view(-1); // assumes non-FCDs are size 1s
  auto reshape_op_indices = make_operator<ReshapeOperator>(
      indices.device().index(), indices.scalar_type());
  reshape_op_indices->SetSynapseInput(p_context_->syn_inputs_[1]);
  int64_t size1[] = {indices.numel()};
  c10::IntArrayRef modified_indices_shape(size1, 1);
  torch::jit::Stack indices_stack = {
      c10::IValue(indices), c10::IValue(modified_indices_shape)};
  reshape_op_indices->AllocateAndAddSynapseNode(
      graph, indices_stack, OutputMetaDataVector(1));
  auto indices_flattened = reshape_op_indices->GetOutputs()[0];

  // Node: updates = grad.view(size);
  // std::vector<int64_t> size{-1, grad.size(-1)};
  int64_t size2[] = {grad.numel() / grad.size(-1), grad.size(-1)};
  auto reshape_op_grad =
      make_operator<ReshapeOperator>(grad.device().index(), grad.scalar_type());
  reshape_op_grad->SetSynapseInput(p_context_->syn_inputs_[0]);
  c10::IntArrayRef modified_input_shape(size2, 2);
  torch::jit::Stack updates_stack = {
      c10::IValue(grad), c10::IValue(modified_input_shape)};
  reshape_op_grad->AllocateAndAddSynapseNode(
      graph, updates_stack, OutputMetaDataVector(1));
  auto updates = reshape_op_grad->GetOutputs()[0];
  // Create cast operator for indices->Float = node_type = "cast_i32_to_f32" for
  // topk
  auto castIndicesToFloatOp = make_operator<CastOperator>(
      this->p_context_->device_id_, "cast_i32_to_f32");
  castIndicesToFloatOp->SetSynapseInput(reshape_op_indices->GetSynOutputs()[0]);
  c10::ScalarType cast_scalar_type = c10::ScalarType::Float;
  std::vector<c10::IValue> cast_stack{
      IValue(indices_flattened), IValue(cast_scalar_type)};
  OutputMetaData md;
  md.dtype = cast_scalar_type;
  castIndicesToFloatOp->AllocateAndAddSynapseNode(graph, cast_stack, {md});

  // Node: topk_idx = at::topk(cast_indices_pt_tensor, numel);
  auto topkOp =
      make_operator<TopkOperator>(this->p_context_->device_id_, "topk");
  int64_t dim = 0;
  bool largest = true;
  bool sorted = true;
  topkOp->SetSynapseInput(castIndicesToFloatOp->GetSynOutputs()[0]);
  std::vector<c10::IValue> topk_stack{
      IValue(castIndicesToFloatOp->GetOutputs()[0]),
      IValue(numel),
      IValue(dim),
      IValue(largest),
      IValue(sorted)};
  topkOp->AllocateAndAddSynapseNode(graph, topk_stack, OutputMetaDataVector(2));
  // output[0] -> topk_values
  // output[1] -> topk_indices
  // Create cast operator for topk_values = node_type = "cast_f32_to_i32"
  auto castTopkValsOp = make_operator<CastOperator>(
      this->p_context_->device_id_, "cast_f32_to_i32");
  castTopkValsOp->SetSynapseInput(topkOp->GetSynOutputs()[0]);
  cast_scalar_type = c10::ScalarType::Int;
  std::vector<c10::IValue> cast_stack1{
      IValue(topkOp->GetOutputs()[0]), IValue(cast_scalar_type)};
  md.dtype = cast_scalar_type;
  castTopkValsOp->AllocateAndAddSynapseNode(graph, cast_stack1, {md});

  synapse_helpers::tensor& syn_updates = reshape_op_grad->GetSynOutputs()[0];
  // Node: reordered_updates = at::gather(updates, 0, topk_indices);
  auto gatherOp = make_operator<GatherOperator>(
      this->p_context_->device_id_, updates.scalar_type());
  gatherOp->SetSynapseInput(syn_updates);
  gatherOp->SetSynapseInput(topkOp->GetSynOutputs()[1]);
  bool sparse_grad = false;
  std::vector<c10::IValue> gather_stack{
      IValue(updates),
      IValue(dim),
      IValue(topkOp->GetOutputs()[1]),
      IValue(sparse_grad)};
  gatherOp->AllocateAndAddSynapseNode(
      graph, gather_stack, OutputMetaDataVector(1));

  /*
    grad_weight.scatter_add_(0, topk_values, reordered_updates);
  */
  auto scatterAddOp = make_operator<ScatterAddOperator>(
      this->p_context_->device_id_, grad.scalar_type());
  scatterAddOp->SetSynapseInput(zeroOp->GetSynOutputs()[0]);
  scatterAddOp->SetSynapseInput(castTopkValsOp->GetSynOutputs()[0]);
  scatterAddOp->SetSynapseInput(gatherOp->GetSynOutputs()[0]);
  std::vector<c10::IValue> sa_stack{
      IValue(zeroOp->GetOutputs()[0]),
      IValue(dim),
      IValue(castTopkValsOp->GetOutputs()[0]),
      IValue(gatherOp->GetOutputs()[0])};
  scatterAddOp->AllocateAndAddSynapseNode(
      graph,
      sa_stack,
      (padding_idx != -1) ? OutputMetaDataVector(1) : output_metadata);
  auto grad_weight = scatterAddOp->GetOutputs()[0];
  synapse_helpers::tensor& syn_grad_weight = scatterAddOp->GetSynOutputs()[0];
  /*
  if (padding_idx != -1) {
    //zero out the entries of grad_weight/return tensor for entry indexed by
  padding_idx
  }
  */
  if (padding_idx != -1) {
    // create a wrapper PT tensor for the non-persistent tensor for zero filling
    Tensor temp_zeros = habana::createPTTensor(
        grad_weight,
        {grad_weight.size(-1)},
        grad_weight.options(),
        grad_weight.suggest_memory_format(),
        grad_weight.scalar_type(),
        false);
    auto zeroOp1 = make_operator<ConstantOutOperator>(
        this->p_context_->device_id_, grad_weight.scalar_type());
    c10::Scalar zero_val = 0;
    zeroOp1->SetPTInputs({temp_zeros});
    zeroOp1->AllocateSynapseInput(graph, temp_zeros, false);
    zero_op_stack.clear();
    zero_op_stack.emplace_back(IValue(temp_zeros));
    zero_op_stack.emplace_back(IValue(zero_val));
    zeroOp1->AllocateAndAddSynapseNode(
        graph, zero_op_stack, OutputMetaDataVector(1));

    auto topk_indices = topkOp->GetOutputs()[1];
    // create a wrapper PT tensor for the non-persistent tensor holding
    // padding_idx
    Tensor padding_idx_tensor = habana::createPTTensor(
        topk_indices,
        {1},
        topk_indices.options(),
        topk_indices.suggest_memory_format(),
        topk_indices.scalar_type(),
        false);
    std::vector<Tensor> padding_vec = {padding_idx_tensor};
    TensorList pad_indices(padding_vec);
    // Create Constant Operator to convert scalar padding_idx
    // to tensor
    Scalar p_converted = static_cast<int>(padding_idx);
    auto constOp = make_operator<ConstantOperator>(
        this->p_context_->device_id_, topk_indices.scalar_type());
    std::vector<c10::IValue> constOp_stack = {
        IValue(pad_indices[0]), IValue(p_converted)};
    constOp->AllocateAndAddSynapseNode(
        graph, constOp_stack, OutputMetaDataVector(1));

    auto indexputOp = make_operator<IndexPutOperator>(
        this->p_context_->device_id_, grad_weight.scalar_type());
    indexputOp->SetSynapseInput(syn_grad_weight);
    indexputOp->SetSynapseInput(constOp->GetSynOutputs()[0]);
    indexputOp->SetSynapseInput(zeroOp1->GetSynOutputs()[0]);

    std::vector<c10::IValue> indexputOp_stack = {
        IValue(grad_weight),
        IValue(pad_indices),
        IValue(temp_zeros),
        IValue(false)};
    indexputOp->AllocateAndAddSynapseNode(
        graph, indexputOp_stack, output_metadata);
    auto result = indexputOp->GetOutputs()[0];
    synapse_helpers::tensor& syn_result = indexputOp->GetSynOutputs()[0];
    SetPTOutput(result);
    SetSynapseOutput(std::move(syn_result));
  } else {
    SetPTOutput(grad_weight);
    SetSynapseOutput(std::move(syn_grad_weight));
  }
}

void EmbeddingBagSumOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    const OutputMetaDataVector& output_metadata) {
  TORCH_CHECK(
      inputs.size() == 5,
      "Incorrect size of inputs expected for EmbeddingBagSumOperator operator");
  TORCH_CHECK(inputs[0].isTensor(), "Input arg1 type expected to be tensor");
  TORCH_CHECK(inputs[1].isTensor(), "Input arg2 type expected to be tensor");
  TORCH_CHECK(inputs[2].isTensor(), "Input arg3 type expected to be tensor");
  TORCH_CHECK(inputs[3].isTensor(), "Input arg4 type expected to be tensor");
  TORCH_CHECK(inputs[4].isInt(), "Input arg5 type expected to be tensor");

  auto input = inputs[0].toTensor();
  auto indices = inputs[1].toTensor();
  auto offsets = inputs[2].toTensor();
  auto valid_count = inputs[3].toTensor();

  TORCH_CHECK(indices.dim() <= 1, "index tensor cannot be more than 1D")
  TORCH_CHECK(offsets.dim() <= 1, "offsets tensor cannot be more than 1D")
  TORCH_CHECK(input.dim() == 2, "Input tensor should be 2D")
  TORCH_CHECK(valid_count.dim() == 1, "valid count tensor should be 1D")

  auto kernel_mode = inputs[4].toInt();
  auto output_size_dim0 =
      kernel_mode ? (offsets.sizes()[0] - 1) : indices.sizes()[0];
  auto output = habana::createPTTensor(
      input,
      {output_size_dim0, input.size(1)},
      input.options(),
      input.suggest_memory_format(), // TBD: not reqd?
      output_metadata.at(0).persistent);
  if (kernel_mode == 0) {
    auto guid = "gather_with_valid_count_2d_" +
        habana_helpers::name_suffix_from_type(input.scalar_type());
    SetGuid(guid);
    p_context_->syn_inputs_.erase(p_context_->syn_inputs_.begin() + 2);
    p_context_->pt_inputs_.erase(p_context_->pt_inputs_.begin() + 2);
  } else if (kernel_mode == 2) {
    auto guid = "embedding_bag_sum_small_lengths_2d_fwd_" +
        habana_helpers::name_suffix_from_type(input.scalar_type());
    SetGuid(guid);
  }
  AllocateSynapseOutput(graph, output, output_metadata.at(0));
  AddNodeToSynapseGraph(graph, nullptr, 0);
}

void EmbeddingBagSumForwardOperator::AllocateSynapseInputs(
    synapse_helpers::graph& graph,
    const std::vector<at::Tensor>& inputs,
    bool is_persistent) {
  HABANA_ASSERT(inputs.size() == 8);

  // Allocate only the tensors needed for fwd operation
  // index 0 is output tensor
  for (int cnt = 0; cnt < 4; cnt++) {
    HabanaOperator::AllocateSynapseInput(graph, inputs[cnt], is_persistent);
  }
}

/*AllocateSynapseInput needs to be overloaded as it is used in PT bridge code*/
synapse_helpers::tensor& EmbeddingBagSumForwardOperator::AllocateSynapseInput(
    synapse_helpers::graph& graph,
    const at::Tensor& input,
    bool is_persistent,
    synTensorType shape_tensor_type,
    void* host_ptr,
    [[maybe_unused]] const std::string& idx) {
  static_cast<void>(shape_tensor_type);
  static_cast<void>(host_ptr);
  // static_cast<void>(idx);
  if (valid_input_idx.count(input_idx)) {
    auto syn_tensor_input = habana_helpers::create_tensor(
        input, graph, is_persistent, false, c10::nullopt);

    p_context_->syn_inputs_.emplace_back(syn_tensor_input);

    p_context_->pt_inputs_.emplace_back(input);
  }
  input_idx++;
  return p_context_->syn_inputs_.back();
}

/*SetSynapseInput needs to be overloaded as it is used in PT bridge code for
 * intermediate nodes*/
synapse_helpers::tensor_or_ref& EmbeddingBagSumForwardOperator::SetSynapseInput(
    synapse_helpers::tensor& tensor) {
  if (valid_input_idx.count(input_idx)) {
    p_context_->syn_inputs_.emplace_back(tensor);
  }

  input_idx++;
  return p_context_->syn_inputs_.back();
}

void EmbeddingBagSumForwardOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    const OutputMetaDataVector& output_metadata) {
  HABANA_ASSERT(inputs.size() == 8);
  HABANA_ASSERT(inputs[0].isTensor());
  HABANA_ASSERT(inputs[1].isTensor());
  HABANA_ASSERT(inputs[2].isTensor());
  HABANA_ASSERT(inputs[3].isTensor());

  auto input = inputs[0].toTensor();
  auto indices = inputs[1].toTensor();
  auto offsets = inputs[2].toTensor();
  auto valid_count = inputs[3].toTensor();

  HABANA_ASSERT(input.dim() == 2);
  HABANA_ASSERT(indices.dim() == 1);
  HABANA_ASSERT(offsets.dim() == 1);
  HABANA_ASSERT(valid_count.numel() == 2);

  auto out = habana::createPTTensor(
      input,
      {offsets.numel() - 1, input.size(1)},
      input.options(),
      input.suggest_memory_format(),
      output_metadata.at(0).persistent);

  AllocateSynapseOutput(graph, out, output_metadata.at(0));
  AddNodeToSynapseGraph(graph, nullptr, 0);
}

void EmbeddingBagSumBackwardOperator::AllocateSynapseInputs(
    synapse_helpers::graph& graph,
    const std::vector<at::Tensor>& inputs,
    bool is_persistent) {
  HABANA_ASSERT(inputs.size() == 5);

  // Allocate only the tensors needed for bwd operation in the right order
  for (int cnt = 1; cnt < 5; cnt++) {
    HabanaOperator::AllocateSynapseInput(graph, inputs[cnt], is_persistent);
  }
}

/*AllocateSynapseInput needs to be overloaded as it is used in PT bridge code*/
synapse_helpers::tensor& EmbeddingBagSumBackwardOperator::AllocateSynapseInput(
    synapse_helpers::graph& graph,
    const at::Tensor& input,
    bool is_persistent,
    synTensorType shape_tensor_type,
    void* host_ptr,
    [[maybe_unused]] const std::string& idx) {
  static_cast<void>(shape_tensor_type);
  static_cast<void>(host_ptr);
  // static_cast<void>(idx);
  if (valid_input_idx.count(input_idx)) {
    auto syn_tensor_input = habana_helpers::create_tensor(
        input, graph, is_persistent, false, c10::nullopt);

    p_context_->syn_inputs_.emplace_back(syn_tensor_input);

    p_context_->pt_inputs_.emplace_back(input);
  }
  input_idx++;
  return p_context_->syn_inputs_.back();
}

/*SetSynapseInput needs to be overloaded as it is used in PT bridge code for
 * intermediate nodes*/
synapse_helpers::tensor_or_ref& EmbeddingBagSumBackwardOperator::
    SetSynapseInput(synapse_helpers::tensor& tensor) {
  if (valid_input_idx.count(input_idx)) {
    p_context_->syn_inputs_.emplace_back(tensor);
  }

  input_idx++;
  return p_context_->syn_inputs_.back();
}

void EmbeddingBagSumBackwardOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    const OutputMetaDataVector& output_metadata) {
  HABANA_ASSERT(inputs.size() == 5);

  HABANA_ASSERT(inputs[0].isTensor());
  HABANA_ASSERT(inputs[1].isTensor());
  HABANA_ASSERT(inputs[2].isTensor());
  HABANA_ASSERT(inputs[3].isTensor());
  HABANA_ASSERT(inputs[4].isTensor());

  auto out = inputs[0].toTensor();
  auto input = inputs[1].toTensor();
  auto indices_bwd = inputs[2].toTensor();
  auto offsets_bwd = inputs[3].toTensor();
  auto valid_count_bwd = inputs[4].toTensor();

  HABANA_ASSERT(out.dim() == 2);
  HABANA_ASSERT(input.dim() == 2);
  HABANA_ASSERT(indices_bwd.dim() == 1);
  HABANA_ASSERT(offsets_bwd.dim() == 1);
  HABANA_ASSERT(valid_count_bwd.numel() == 2);

  AllocateSynapseOutput(graph, out, output_metadata.at(0));
  AddNodeToSynapseGraph(graph, nullptr, 0);
}

void EmbeddingBagSumBwdKernelModeOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    const OutputMetaDataVector& output_metadata) {
  static_cast<void>(output_metadata);
  HABANA_ASSERT(inputs.size() == 6);

  HABANA_ASSERT(inputs[0].isTensor());
  HABANA_ASSERT(inputs[1].isTensor());
  HABANA_ASSERT(inputs[2].isTensor());
  HABANA_ASSERT(inputs[3].isTensor());
  HABANA_ASSERT(inputs[4].isTensor());
  HABANA_ASSERT(inputs[5].isInt());

  auto out = inputs[0].toTensor();
  auto input = inputs[1].toTensor();
  auto indices_bwd = inputs[2].toTensor();
  auto offsets_bwd = inputs[3].toTensor();
  auto valid_count_bwd = inputs[4].toTensor();
  auto kernel_mode = inputs[5].toInt();

  HABANA_ASSERT(out.dim() == 2);
  HABANA_ASSERT(input.dim() == 2);
  HABANA_ASSERT(indices_bwd.dim() == 1);
  HABANA_ASSERT(offsets_bwd.dim() == 1);
  HABANA_ASSERT(valid_count_bwd.numel() == 2);
  HABANA_ASSERT((kernel_mode >= 0) && (kernel_mode < 3));

  p_context_->syn_outputs_.emplace_back(
      habana_helpers::duplicate_tensor_in_memory_section(
          p_context_->syn_inputs_[0], graph, output_metadata.at(0).external));
  p_context_->syn_inputs_.erase(p_context_->syn_inputs_.begin());

  p_context_->pt_outputs_.emplace_back(p_context_->pt_inputs_[0]);
  p_context_->pt_inputs_.erase(p_context_->pt_inputs_.begin());

  AddNodeToSynapseGraph(graph, nullptr, 0);
}

static auto& EmbeddingKernelsKernelRegistry =
    habana::KernelRegistry()
        .add("aten::constant_pad_nd", KERNEL_FN(PadOperator))
        .add("hpu::constant_pad_nd", KERNEL_FN(PadOperator))
        .add("hpu::constant_pad_nd_ht", KERNEL_FN(PadOperatorHT))
        .add(
            "aten::embedding_bag_sum_fwd",
            KERNEL_FN(EmbeddingBagSumForwardOperator))
        .add(
            "aten::embedding_bag_sum_bwd.out",
            KERNEL_FN(EmbeddingBagSumBackwardOperator))
        .add(
            "aten::embedding_dense_backward",
            KERNEL_FN(EmbeddingDenseBackwardOperator))
        .add("hpu::embedding_bag_sum", KERNEL_FN(EmbeddingBagSumOperator))
        .add(
            "hpu::embedding_bag_sum_bwd_out",
            KERNEL_FN(EmbeddingBagSumBwdKernelModeOperator));
