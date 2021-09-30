/******************************************************************************
 * Copyright (C) 2020 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */
#include <ATen/InferSize.h>
#include <perf_lib_layer_params.h>
#include <synapse_api.h>
#include <torch/script.h>

#include "habana_device/HPUCheck.h"
#include "habana_device/hpu_cached_devices.h"
#include "habana_helpers/graph.h"
#include "habana_helpers/logging.h"
#include "habana_helpers/tensor_utils.h"
#include "habana_helpers/unused_macro.h"
#include "habana_kernels/basic_kernels.h"
#include "habana_kernels/embedding_kernels.h"
#include "habana_kernels/index_kernels.h"
#include "habana_kernels/simple_generic_kernel.h"
#include "habana_kernels/tensor_shape_kernels.h"
#include "habana_kernels/topk_kernels.h"
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

void PadOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    bool is_output_persistent) {
  TORCH_CHECK(
      inputs.size() == 3,
      "Incorrect size of inputs expected for PadOperator Operator");
  TORCH_CHECK(
      inputs[0].isTensor(),
      "Input arg1 expected to be Tensor for PadOperator Operator");
  TORCH_CHECK(
      inputs[1].isIntList(),
      "Input arg2 expected to be IntList for PadOperator Operator");
  TORCH_CHECK(
      inputs[2].isScalar(),
      "Input arg3 expected to be Scalar for PadOperator Operator");

  auto self = inputs[0].toTensor();
  auto pad = inputs[1].toIntList().vec();
  auto value = inputs[2].toScalar();

  auto ndim = self.dim();
  auto lpad = pad.size() / 2;

  auto shape = compute_output_shape(self, pad);

  ns_PadKernelEx::Params param;
  param.mode = PadMode_t::PAD_MODE_CONSTANT;
  param.value.f = value.to<float>();
  memset(param.pads, 0, sizeof(param.pads));
  for (unsigned int i = 0; i < lpad; i++) {
    param.pads[i] = pad[2 * i];
    param.pads[i + ndim] = pad[2 * i + 1];
  }

  auto output = at::empty(shape, self.options());
  AllocateSynapseOutput(graph, output, is_output_persistent);
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
  Op.AllocateAndAddSynapseNode(graph, stack, true);

  // compile and execute the graph
  Op.Compile(graph);

  std::vector<at::Tensor> out = Op.GetOutputs();
  TORCH_CHECK(out.size() == 1, "Incorrect size of outputs");

  PT_KERNEL_END;
  return out.at(0);
}

void EmbeddingOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    bool is_output_persistent) {
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
      inputs[3].isBool(),
      "Input arg4 type expected to be Bool for embedding operator");
  TORCH_CHECK(
      inputs[4].isBool(),
      "Input arg5 type expected to be Bool for embedding operator");

  auto weight = inputs[0].toTensor();
  auto indices = inputs[1].toTensor();
  // auto padding_idx = inputs[2].toInt();
  auto scale_grad_by_freq = inputs[3].toBool();
  auto sparse = inputs[4].toBool();

  TORCH_CHECK(
      scale_grad_by_freq == false, "scale_grad_by_value = true not supported")
  TORCH_CHECK(sparse == false, "sparse embedding not supported")
  // TORCH_WARN(
  //    padding_idx == -1,
  //    "padding index is ignored to mimic CPU implementation.");

  if (indices.dim() == 1) {
    // Create IndexSelect operator
    auto indexSelectOp = make_operator<IndexSelectOperator>(
        this->p_context_->device_id_, weight.scalar_type());
    indexSelectOp->SetSynapseInput(p_context_->syn_inputs_[0]);
    indexSelectOp->SetSynapseInput(p_context_->syn_inputs_[1]);
    // Build Params for the graph
    int64_t dim = 0;
    std::vector<c10::IValue> stack{
        IValue(weight), IValue(dim), IValue(indices)};
    indexSelectOp->AllocateAndAddSynapseNode(
        graph, stack, is_output_persistent);

    p_context_->syn_outputs_.emplace_back(
        std::move(indexSelectOp->GetSynOutputs()[0]));
    p_context_->pt_outputs_.emplace_back(
        std::move(indexSelectOp->GetOutputs()[0]));
  } else {
    auto size = indices.sizes().vec();
    // append size of last N-1 dimensions of weight (assuming its a Nd tensor)
    for (auto d : weight.sizes().slice(1)) {
      size.push_back(d);
    }

    auto ReshapeOp = make_operator<ReshapeOperator>(
        this->p_context_->device_id_, indices.scalar_type());
    ReshapeOp->SetSynapseInput(p_context_->syn_inputs_[1]);

    int64_t data[1];
    data[0] = indices.numel();
    c10::IntArrayRef shape(data, 1);
    // Build Params for the graph
    std::vector<c10::IValue> stack;
    stack.emplace_back(IValue(indices));
    stack.emplace_back(IValue(shape));
    ReshapeOp->AllocateAndAddSynapseNode(graph, stack, false);
    stack.clear();

    auto indexSelectOp = make_operator<IndexSelectOperator>(
        this->p_context_->device_id_, weight.scalar_type());
    indexSelectOp->SetSynapseInput(p_context_->syn_inputs_[0]);
    indexSelectOp->SetSynapseInput(ReshapeOp->GetSynOutputs()[0]);
    // Build Params for the graph
    int64_t dim = 0;
    stack.emplace_back(IValue(weight));
    stack.emplace_back(IValue(dim));
    stack.emplace_back(IValue(ReshapeOp->GetOutputs()[0]));
    indexSelectOp->AllocateAndAddSynapseNode(graph, stack, false);
    stack.clear();

    auto ReshapeOp_2 = make_operator<ReshapeOperator>(
        this->p_context_->device_id_,
        indexSelectOp->GetOutputs()[0].scalar_type());
    ReshapeOp_2->SetSynapseInput(indexSelectOp->GetSynOutputs()[0]);
    // Build Params for the graph
    stack.emplace_back(IValue(indexSelectOp->GetOutputs()[0]));
    stack.emplace_back(IValue(size));
    ReshapeOp_2->AllocateAndAddSynapseNode(graph, stack, is_output_persistent);

    p_context_->syn_outputs_.emplace_back(
        std::move(ReshapeOp_2->GetSynOutputs()[0]));
    p_context_->pt_outputs_.emplace_back(
        std::move(ReshapeOp_2->GetOutputs()[0]));
  }
}

/** @brief simple lookup table that looks up embeddings in a fixed dictionary
 * and size.
 * @param weight (Tensor) The embedding matrix with number of rows equal to the
 * maximum possible index + 1, and number of columns equal to the embedding size
 * @param indices (LongTensor)  Tensor containing indices into the embedding
 * matrix
 * @param padding_idx (int, optional) If given, pads the output with the
 * embedding vector at padding_idx (initialized to zeros) whenever it encounters
 * the index
 * @param scale_grad_by_freq (boolean, optional) If given, this will scale
 * gradients by the inverse of frequency of the words in the mini-batch
 * @param sparse (boolean, optional)  If True, gradient w.r.t. weight will be a
 * sparse tensor.
 */
Tensor embedding_hpu(
    const Tensor& weight,
    const Tensor& indices,
    int64_t padding_idx,
    bool scale_grad_by_freq,
    bool sparse) {
  PT_KERNEL_BEGIN;

  auto indices_int = habana_helpers::cast_tensor_to_integer(indices);
  at::ScalarType scalar_type = weight.scalar_type();
  std::string node_type =
      "embedding_fwd_" + habana_helpers::name_suffix_from_type(scalar_type);
  size_t device_id = weight.device().index();
  auto& device = synapse_helpers::HPURegistrar::get_device(device_id);

  // create the operator
  EmbeddingOperator Op(device_id, scalar_type);

  // Build Params for the graph
  std::vector<c10::IValue> stack = {
      IValue(weight),
      IValue(indices_int),
      IValue(padding_idx),
      IValue(scale_grad_by_freq),
      IValue(sparse)};
  size_t key = Op.GetRecipeKey(node_type, stack);

  // Assign Inputs to the Operator
  std::vector<at::Tensor> pt_inputs{weight, indices_int};

  if (device.get_recipe_handle_cache().isCached(key)) {
    PT_KERNEL_DEBUG("Cache hit key:", key);
    auto size = indices.sizes().vec();
    // append size of last N-1 dimensions of weight (assuming its a Nd tensor)
    for (auto d : weight.sizes().slice(1)) {
      size.push_back(d);
    }
    auto result =
        at::empty(size, weight.options(), weight.suggest_memory_format());
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

void EmbeddingDenseBackwardOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    bool is_output_persistent) {
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
  UNUSED auto padding_idx = inputs[3].toInt();
  auto scale_grad_by_freq = inputs[4].toBool();
  int64_t numel = indices.numel();
  TORCH_CHECK(
      scale_grad_by_freq == false, "scale_grad_by_freq = true not supported")

  // create a wrapper PT tensor for the non-persistent tensor for zero filling
  Tensor grad_temp = habana_helpers::createPTTensor(
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
  zeroOp->AllocateAndAddSynapseNode(graph, zero_op_stack, false);

  // Node: indices_flattened = indices.view(-1); // assumes non-FCDs are size 1s
  auto reshape_op_indices = make_operator<ReshapeOperator>(
      indices.device().index(), indices.scalar_type());
  reshape_op_indices->SetSynapseInput(p_context_->syn_inputs_[1]);
  int64_t size1[] = {indices.numel()};
  c10::IntArrayRef modified_indices_shape(size1, 1);
  torch::jit::Stack indices_stack = {
      c10::IValue(indices), c10::IValue(modified_indices_shape)};
  reshape_op_indices->AllocateAndAddSynapseNode(graph, indices_stack, false);
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
  reshape_op_grad->AllocateAndAddSynapseNode(graph, updates_stack, false);
  auto updates = reshape_op_grad->GetOutputs()[0];
  // Create cast operator for indices->Float = node_type = "cast_i32_to_f32" for
  // topk
  auto castIndicesToFloatOp = make_operator<CastOperator>(
      this->p_context_->device_id_, "cast_i32_to_f32");
  castIndicesToFloatOp->SetSynapseInput(reshape_op_indices->GetSynOutputs()[0]);
  c10::ScalarType cast_scalar_type = c10::ScalarType::Float;
  std::vector<c10::IValue> cast_stack{
      IValue(indices_flattened), IValue(cast_scalar_type)};
  castIndicesToFloatOp->AllocateAndAddSynapseNode(graph, cast_stack, false);

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
  topkOp->AllocateAndAddSynapseNode(graph, topk_stack, {false, false});
  // output[0] -> topk_values
  // output[1] -> topk_indices
  // Create cast operator for topk_values = node_type = "cast_f32_to_i32"
  auto castTopkValsOp = make_operator<CastOperator>(
      this->p_context_->device_id_, "cast_f32_to_i32");
  castTopkValsOp->SetSynapseInput(topkOp->GetSynOutputs()[0]);
  cast_scalar_type = c10::ScalarType::Int;
  std::vector<c10::IValue> cast_stack1{
      IValue(topkOp->GetOutputs()[0]), IValue(cast_scalar_type)};
  castTopkValsOp->AllocateAndAddSynapseNode(graph, cast_stack1, false);

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
  gatherOp->AllocateAndAddSynapseNode(graph, gather_stack, false);

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
      graph, sa_stack, (padding_idx != -1) ? false : is_output_persistent);
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
    Tensor temp_zeros = habana_helpers::createPTTensor(
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
    zeroOp1->AllocateAndAddSynapseNode(graph, zero_op_stack, false);

    auto topk_indices = topkOp->GetOutputs()[1];
    // create a wrapper PT tensor for the non-persistent tensor holding
    // padding_idx
    Tensor padding_idx_tensor = habana_helpers::createPTTensor(
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
    constOp->AllocateAndAddSynapseNode(graph, constOp_stack, false);

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
        graph, indexputOp_stack, is_output_persistent);
    auto result = indexputOp->GetOutputs()[0];
    synapse_helpers::tensor& syn_result = indexputOp->GetSynOutputs()[0];
    SetPTOutput(result);
    SetSynapseOutput(std::move(syn_result));
  } else {
    SetPTOutput(grad_weight);
    SetSynapseOutput(std::move(syn_grad_weight));
  }
}

/** @brief Function implements embedding backward (for dense-tensors)
 * @param grad (Tensor) Input gradient for bwd pass
 * @param indices (LongTensor) Tensor containing indices into the embedding
 * matrix
 * @param num_weights (int) Number fo rows in the weight tensor
 * @param padding_idx (int, optional) If given, pads the output with the
 * embedding vector at padding_idx (initialized to zeros) whenever it encounters
 * the index. NOTE: Currently not supported (not used in cpu implementation
 * also)
 * @param scale_grad_by_freq (boolean, optional) UNUSED: If given, this will
 * scale gradients by the inverse of frequency of the words in the mini-batch
 */
Tensor embedding_dense_backward_hpu(
    const Tensor& grad,
    const Tensor& indices,
    int64_t num_weights,
    int64_t padding_idx,
    bool scale_grad_by_freq) {
  PT_KERNEL_BEGIN;

  auto indices_int = habana_helpers::cast_tensor_to_integer(indices);
  at::ScalarType scalar_type = grad.scalar_type();
  std::string node_type = "embedding_dense_bwd_" +
      habana_helpers::name_suffix_from_type(scalar_type);

  size_t device_id = grad.device().index();
  auto& device = synapse_helpers::HPURegistrar::get_device(device_id);

  // create the operator
  EmbeddingDenseBackwardOperator Op(device_id, scalar_type);

  // Build Params for the graph
  std::vector<c10::IValue> stack = {
      IValue(grad),
      IValue(indices_int),
      IValue(num_weights),
      IValue(padding_idx),
      IValue(scale_grad_by_freq)};
  size_t key = Op.GetRecipeKey(node_type, stack);

  // Assign Inputs to the Operator
  std::vector<at::Tensor> pt_inputs{grad, indices_int};

  if (device.get_recipe_handle_cache().isCached(key)) {
    PT_KERNEL_DEBUG("Cache hit key:", key);
    Op.SetPTInputs(pt_inputs);
    auto grad_weight = at::empty(
        {num_weights, grad.size(-1)},
        grad.options(),
        grad.suggest_memory_format());
    Op.SetPTOutput(grad_weight);
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
  PT_KERNEL_END;

  return out.at(0);
}

void EmbeddingBagSumOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    bool is_output_persistent) {
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

  auto output = habana_helpers::createPTTensor(
      input,
      {offsets.sizes()[0] - 1, input.size(1)},
      input.options(),
      input.suggest_memory_format(), // TBD: not reqd?
      is_output_persistent);
  AllocateSynapseOutput(graph, output, is_output_persistent);
  AddNodeToSynapseGraph(graph, nullptr, 0);
}

/**********************************************************
*@brief
@param [in]  input 2D tensor, FP32/FP16
@param [in]  indices 0-1D, FP32/FP16
@param [in]  offsets 0-1D, FP32/FP16
@param [in]  valid_count  - contains 2 elements namely valid_count_offsets and
valid_count_indices
**********************************************************/
Tensor embedding_bag_sum_hpu(
    const Tensor& input,
    const Tensor& indices,
    const Tensor& offsets,
    const Tensor& valid_count,
    int64_t kernel_mode) {
  PT_KERNEL_BEGIN;

  // Convert index tensor from 0D to 1D if required
  if (indices.dim() == 0) {
    indices.unsafeGetTensorImpl()->set_sizes_and_strides({1}, {1});
  }

  // Convert offsets tensor from 0D to 1D if required
  if (offsets.dim() == 0) {
    offsets.unsafeGetTensorImpl()->set_sizes_and_strides({1}, {1});
  }

  auto indices_i32 = habana_helpers::cast_tensor_to_integer(indices);
  auto offsets_i32 = habana_helpers::cast_tensor_to_integer(offsets);
  auto valid_count_i32 = habana_helpers::cast_tensor_to_integer(valid_count);

  at::ScalarType scalar_type = input.scalar_type();
  std::string node_type;

  // TODO kernel selection
  /*
  if (kernel_mode == 0) {
    node_type = "embedding_bag_sum_2d_fwd_" +
        habana_helpers::name_suffix_from_type(scalar_type);
    ;
  } else {
    node_type = "embedding_bag_sum_small_lengths_2d_fwd_" +
        habana_helpers::name_suffix_from_type(scalar_type);
    ;
  }
  */

  size_t device_id = input.device().index();

  EmbeddingBagSumOperator Op(device_id, scalar_type);
  // Create Graph
  auto graph = habana_helpers::create_graph(device_id, node_type);

  // Assign Inputs to the Operator
  std::vector<at::Tensor> pt_inputs{
      input, indices_i32, offsets_i32, valid_count_i32};
  Op.AllocateSynapseInputs(graph, pt_inputs, true);

  // Build Params for the graph
  std::vector<c10::IValue> stack = {
      IValue(input),
      IValue(indices_i32),
      IValue(offsets_i32),
      IValue(valid_count_i32),
      IValue(kernel_mode)};
  Op.AllocateAndAddSynapseNode(graph, stack, true);

  // compile and execute the graph
  Op.Compile(graph);

  std::vector<at::Tensor> out = Op.GetOutputs();
  TORCH_CHECK(out.size() == 1, "Incorrect size of outputs");

  PT_KERNEL_END;
  return out.at(0);
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
    bool is_shape_tensor) {
  static_cast<void>(is_shape_tensor);
  if (valid_input_idx.count(input_idx)) {
    auto syn_tensor_input = habana_helpers::create_tensor(
        input, graph, is_persistent, c10::nullopt);

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
    bool is_output_persistent) {
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

  auto out = habana_helpers::createPTTensor(
      input,
      {offsets.numel() - 1, input.size(1)},
      input.options(),
      input.suggest_memory_format(),
      is_output_persistent);

  AllocateSynapseOutput(graph, out, is_output_persistent);
  AddNodeToSynapseGraph(graph, nullptr, 0);
}

/**********************************************************
*@brief
@param [in]  input 2D tensor, FP32/BF16
@param [in]  indices_fwd 0-1D, i32
@param [in]  offsets_fwd 0-1D, i32
@param [in]  valid_count_offsets. Needed because offsets_fwd is a persistent
tensor across epochs its size can be larger than the valid_count_offset
@param [in]  indices_bwd 0-1D, i32
@param [in]  offsets_bwd 0-1D, i32
@param [in]  valid_count_bwd, i32
@param [in]  grad_weight FP32/BF16
**********************************************************/
Tensor embedding_bag_sum_fwd_hpu(
    const Tensor& input,
    const Tensor& indices_fwd,
    const Tensor& offsets_fwd,
    const Tensor& valid_count,
    const Tensor& indices_bwd,
    const Tensor& offsets_bwd,
    const Tensor& valid_count_bwd,
    const Tensor& grad_weight) {
  PT_KERNEL_BEGIN;

  at::ScalarType scalar_type = input.scalar_type();
  // TODO support other kernel flavours
  std::string node_type = "embedding_bag_sum_2d_fwd_" +
      habana_helpers::name_suffix_from_type(scalar_type);
  size_t device_id = input.device().index();

  EmbeddingBagSumForwardOperator Op(device_id, scalar_type);

  // Create Graph
  auto graph = habana_helpers::create_graph(device_id, node_type);

  // Assign Inputs to the Operator
  std::vector<at::Tensor> pt_inputs{
      input,
      indices_fwd,
      offsets_fwd,
      valid_count,
      indices_bwd,
      offsets_bwd,
      valid_count_bwd,
      grad_weight};
  Op.AllocateSynapseInputs(graph, pt_inputs, true);

  // Build Params for the graph
  std::vector<c10::IValue> stack = {
      IValue(input),
      IValue(indices_fwd),
      IValue(offsets_fwd),
      IValue(valid_count),
      IValue(indices_bwd),
      IValue(offsets_bwd),
      IValue(valid_count_bwd),
      IValue(grad_weight)};
  Op.AllocateAndAddSynapseNode(graph, stack, true);

  // compile and execute the graph
  Op.Compile(graph);

  std::vector<Tensor> out = Op.GetOutputs();
  HABANA_ASSERT(out.size() == 1);

  PT_KERNEL_END;
  return out.at(0);
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
    bool is_shape_tensor) {
  static_cast<void>(is_shape_tensor);
  if (valid_input_idx.count(input_idx)) {
    auto syn_tensor_input = habana_helpers::create_tensor(
        input, graph, is_persistent, c10::nullopt);

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
    bool is_output_persistent) {
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

  AllocateSynapseOutput(graph, out, is_output_persistent);
  AddNodeToSynapseGraph(graph, nullptr, 0);
}

/**********************************************************
*@brief
@param [in/out] out
@param [in]  input 2D tensor, FP32/BF16
@param [in]  indices_fwd 0-1D, i32
@param [in]  offsets_fwd 0-1D, i32
@param [in]  1D, i32
@param [in]  indices_bwd 0-1D, i32
@param [in]  offsets_bwd 0-1D, i32
@param [in]  valid_count_bwd, i32
**********************************************************/
Tensor& embedding_bag_sum_bwd_out_hpu(
    Tensor& out,
    const Tensor& input,
    const Tensor& indices_bwd,
    const Tensor& offsets_bwd,
    const Tensor& valid_count_bwd) {
  PT_KERNEL_BEGIN;

  at::ScalarType scalar_type = input.scalar_type();
  // TODO support other kernel flavours
  std::string node_type = "embedding_bag_sum_small_lengths_2d_fwd_" +
      habana_helpers::name_suffix_from_type(scalar_type);
  size_t device_id = indices_bwd.device().index();
  auto& device = synapse_helpers::HPURegistrar::get_device(device_id);

  EmbeddingBagSumBackwardOperator Op(device_id, scalar_type);

  // Build Params for the graph
  std::vector<c10::IValue> stack = {
      IValue(out),
      IValue(input),
      IValue(indices_bwd),
      IValue(offsets_bwd),
      IValue(valid_count_bwd)};

  // Assign Inputs to the Operator
  std::vector<at::Tensor> pt_inputs{
      out, input, indices_bwd, offsets_bwd, valid_count_bwd};

  size_t key = Op.GetRecipeKey(node_type, stack);

  if (device.get_recipe_handle_cache().isCached(key)) {
    PT_KERNEL_DEBUG("Cache hit key:", key);

    std::vector<at::Tensor> pt_inputs_slice =
        std::vector<at::Tensor>(pt_inputs.begin() + 1, pt_inputs.end());
    Op.SetPTInputs(pt_inputs_slice);
    Op.SetPTOutput(out);
    Op.Execute(key);
  } else {
    PT_KERNEL_DEBUG("Key:", key);
    auto graph = habana_helpers::create_graph(device_id, node_type);

    Op.AllocateSynapseInputs(graph, pt_inputs, true);
    Op.AllocateAndAddSynapseNode(graph, stack, true);
    // compile and execute the graph
    Op.Compile(graph);
  }

  PT_KERNEL_END;
  return out;
}

void EmbeddingBagSumBwdKernelModeOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    bool is_output_persistent) {
  static_cast<void>(is_output_persistent);
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

  p_context_->syn_outputs_.emplace_back(std::move(p_context_->syn_inputs_[0]));
  p_context_->syn_inputs_.erase(p_context_->syn_inputs_.begin());

  p_context_->pt_outputs_.emplace_back(p_context_->pt_inputs_[0]);
  p_context_->pt_inputs_.erase(p_context_->pt_inputs_.begin());

  AddNodeToSynapseGraph(graph, nullptr, 0);
}

/**********************************************************
*@brief
@param [in/out] out
@param [in]  input 2D tensor, FP32/BF16
@param [in]  indices_fwd 0-1D, i32
@param [in]  offsets_fwd 0-1D, i32
@param [in]  1D, i32
@param [in]  indices_bwd 0-1D, i32
@param [in]  offsets_bwd 0-1D, i32
@param [in]  valid_count_bwd, i32
@param [in]  kernel_mode,  scalar, i64
**********************************************************/
Tensor& embedding_bag_sum_bwd_out_kernel_mode_hpu(
    Tensor& out,
    const Tensor& input,
    const Tensor& indices_bwd,
    const Tensor& offsets_bwd,
    const Tensor& valid_count_bwd,
    int64_t kernel_mode) {
  PT_KERNEL_BEGIN;

  at::ScalarType scalar_type = input.scalar_type();
  // TODO support other kernel flavours
  std::string node_type = "embedding_bag_sum_small_lengths_2d_fwd_" +
      habana_helpers::name_suffix_from_type(scalar_type);
  size_t device_id = indices_bwd.device().index();
  auto& device = synapse_helpers::HPURegistrar::get_device(device_id);

  EmbeddingBagSumBwdKernelModeOperator Op(device_id, scalar_type);

  // Build Params for the graph
  std::vector<c10::IValue> stack = {
      IValue(out),
      IValue(input),
      IValue(indices_bwd),
      IValue(offsets_bwd),
      IValue(valid_count_bwd),
      IValue(kernel_mode)};

  // Assign Inputs to the Operator
  std::vector<at::Tensor> pt_inputs{
      out, input, indices_bwd, offsets_bwd, valid_count_bwd};

  size_t key = Op.GetRecipeKey(node_type, stack);

  if (device.get_recipe_handle_cache().isCached(key)) {
    PT_KERNEL_DEBUG("Cache hit key:", key);

    std::vector<at::Tensor> pt_inputs_slice =
        std::vector<at::Tensor>(pt_inputs.begin() + 1, pt_inputs.end());
    Op.SetPTInputs(pt_inputs_slice);
    Op.SetPTOutput(out);
    Op.Execute(key);
  } else {
    PT_KERNEL_DEBUG("Key:", key);
    auto graph = habana_helpers::create_graph(device_id, node_type);

    Op.AllocateSynapseInputs(graph, pt_inputs, true);
    Op.AllocateAndAddSynapseNode(graph, stack, true);
    // compile and execute the graph
    Op.Compile(graph);
  }

  PT_KERNEL_END;
  return out;
}

static auto& KernelRegistry =
    habana::KernelRegistry()
        .add(
            "aten::constant_pad_nd",
            [](const int device_id, c10::ScalarType node_type) {
              return std::make_shared<PadOperator>(device_id, node_type);
            })
        .add(
            "aten::embedding",
            [](const int device_id, c10::ScalarType node_type) {
              return std::make_shared<EmbeddingOperator>(device_id, node_type);
            })
        .add(
            "aten::embedding_bag_sum_fwd",
            [](const int device_id, c10::ScalarType node_type) {
              return std::make_shared<EmbeddingBagSumForwardOperator>(
                  device_id, node_type);
            })
        .add(
            "aten::embedding_bag_sum_bwd.out",
            [](const int device_id, c10::ScalarType node_type) {
              return std::make_shared<EmbeddingBagSumBackwardOperator>(
                  device_id, node_type);
            })
        .add(
            "aten::embedding_dense_backward",
            [](const int device_id, c10::ScalarType node_type) {
              return std::make_shared<EmbeddingDenseBackwardOperator>(
                  device_id, node_type);
            })
        .add(
            "hpu::embedding_bag_sum",
            [](const int device_id, c10::ScalarType node_type) {
              return std::make_shared<EmbeddingBagSumOperator>(
                  device_id, node_type);
            })
        .add(
            "hpu::embedding_bag_sum_bwd_out",
            [](const int device_id, c10::ScalarType node_type) {
              return std::make_shared<EmbeddingBagSumBwdKernelModeOperator>(
                  device_id, node_type);
            });
