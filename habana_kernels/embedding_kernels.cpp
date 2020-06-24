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
#include "habana_kernels/embedding_kernels.h"
#include "habana_kernels/simple_generic_kernel.h"
#include "kernel_utils.h"

using namespace torch;

/************************************************************************
 * @brief Implments forward pass of Embedding bag op.
 * @param[in] weight - embedding table, 2D, FP32/Fp16
 * @param[in] indices - 1D, Long int
 * @param[in] offsets - 1D, Long int
 * @param[in]  scale_grad_by_freq - bool flag to enable additional scaling of
 *gradient. not supported
 * @param[in] mode - mean/sum. max is not supported
 * @param[in] sparse - bool flag to enable sparse mode. not supported
 * @param[in] per_sample_weights - not supported
 * @param[out] output - 2D, Fp32/FP16
 * @param[out] offset2bag, bag_size - dummy tensors used by CPU Op
 * @param[in] include_last_offset - bool flag to get the size of indices
 * as last element. not supported
 ************************************************************************/
std::tuple<Tensor, Tensor, Tensor, Tensor> embedding_bag_hpu(
    const Tensor& weight,
    const Tensor& indices,
    const Tensor& offsets,
    bool scale_grad_by_freq,
    int64_t mode,
    UNUSED bool sparse,
    Tensor& per_sample_weights,
    UNUSED bool include_last_offset) {
  PT_KERNEL_BEGIN;

  TORCH_CHECK(
      scale_grad_by_freq == false,
      "scaling gradient by frequency is not supported");
  TORCH_CHECK(
      (mode == EmbeddingBagMode_t::EMBEDDING_BAG_MODE_MEAN) ||
          (mode == EmbeddingBagMode_t::EMBEDDING_BAG_MODE_SUM),
      "only sum and mean modes supported");
  TORCH_CHECK(
      per_sample_weights.defined() == false,
      "per sample weight is not supported");

  // TODO to convert long into i32. To be removed once casting  kernel
  // is available for long->i32
  auto indices_i32 = habana_helpers::cast_tensor_to_integer(indices);
  auto offsets_i32 = habana_helpers::cast_tensor_to_integer(offsets);

  auto output = at::empty({offsets.size(0), weight.size(1)}, weight.options());

  std::vector<const at::Tensor*> pt_outputs{&output};
  std::vector<const at::Tensor*> pt_inputs{&weight, &indices_i32, &offsets_i32};

  ns_EmbeddingWithSgdKernel::Params param;
  param.mode = static_cast<EmbeddingBagMode_t>(mode);
  // wd, mom, damp, nesterov
  param.sgd = {0, 0, 0, false};

  synapse_simple_generic_kernel(
      pt_outputs,
      pt_inputs,
      "embedding_sgd",
      &param,
      sizeof(param),
      SynapsePassType::FORWARD_PASS);

  // The below tensors are not returned by TPC nevertheless create them to match
  // function signature
  Tensor offset2bag = at::empty({}, offsets.options());
  auto bag_size = at::empty({}, indices.options());

  PT_KERNEL_END;

  return std::tuple<Tensor, Tensor, Tensor, Tensor>(
      output, offset2bag, bag_size, bag_size);
}

/************************************************************************
 * @brief Implments backward pass of Embedding bag op
 * @param[in] grad - gradient of output, 2D, FP32/Fp16
 * @param[in] indices - 1D, Long int
 * @param[in] offsets - 1D, Long int
 * @param[in]  scale_grad_by_freq - bool flag to enable additional scaling of
 *gradient. not supported
 * @param[in] mode - mean/sum. max is not supported
 * @param[in] per_sample_weights - not supported
 * @param[out] momentum_out - gradient of weights corresponding to the indices
 * @param[out] offset2bag, bag_size - dummy tensors used by CPU Op
 ************************************************************************/
Tensor embedding_bag_bwd_hpu(
    Tensor& grad,
    Tensor& indices,
    Tensor& offsets,
    UNUSED Tensor& offset2bag,
    UNUSED Tensor& bag_size,
    UNUSED Tensor& maximum_indices,
    int num_weights,
    bool scale_grad_by_freq,
    int mode,
    Tensor per_sample_weights) {
  PT_KERNEL_BEGIN;

  TORCH_CHECK(
      scale_grad_by_freq == false,
      "scaling gradient by frequency is not supported");
  TORCH_CHECK(
      mode <= EmbeddingBagMode_t::EMBEDDING_BAG_MODE_MEAN,
      "only sum and mean modes supported");
  TORCH_CHECK(
      per_sample_weights.defined() == false,
      "per sample weight is not supported");

  auto indices_i32 =
      indices.to("cpu").to(c10::ScalarType::Int).to(indices.device());
  auto offsets_i32 =
      offsets.to("cpu").to(c10::ScalarType::Int).to(offsets.device());

  // since SGD output is not used, feed in all zeros
  auto weights_in = at::empty({num_weights, grad.size(1)}, grad.options());
  auto weights_out = at::empty({num_weights, grad.size(1)}, grad.options());

  // This kernel computes gradient + SGD. To get back only gradients, set
  // epoch_number = 0, momentum factor = 1 and fetch output momentum vector
  // Note: Duplicate indices are not supported by this kernel due to RMW issue.
  // In such cases custom op for embedding bag should be used
  auto momentum_in = at::zeros(weights_out.sizes(), weights_out.options());
  auto momentum_out = at::zeros(weights_out.sizes(), weights_out.options());
  // at::zeros works only for float
  auto learning_rate = at::zeros({1}, grad.options());
  auto epoch_num_i32 = learning_rate.toType(c10::ScalarType::Int);

  std::vector<const at::Tensor*> pt_outputs{&weights_out, &momentum_out};
  std::vector<const at::Tensor*> pt_inputs{
      &grad,
      &weights_in,
      &momentum_in,
      &indices_i32,
      &offsets_i32,
      &epoch_num_i32,
      &learning_rate,
  };

  ns_EmbeddingWithSgdKernel::Params param;
  param.mode = static_cast<EmbeddingBagMode_t>(mode);
  // wd, mom, damp, nesterov
  param.sgd = {0, 1, 0, false};

  synapse_simple_generic_kernel(
      pt_outputs,
      pt_inputs,
      "embedding_sgd",
      &param,
      sizeof(param),
      SynapsePassType::BACKWARD_PASS);

  PT_KERNEL_END;

  return momentum_out;
}

/** @brief Function implementing torch.nn.functional.pad(input, pad,
 * mode='constant', value=0)
 *  @param self N-dimensional input tensor
 *  @param pad m-elements tuple, where m/2 ≤ input dimensions and m is even
 *  @param value fill value for "constant" padding
 */
Tensor constant_pad_hpu(const Tensor& self, IntArrayRef pad, Scalar value) {
  PT_KERNEL_BEGIN;

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

  ns_PadKernel::Params param;
  param.value.f = value.to<float>();
  memset(param.pads, 0, sizeof(param.pads));
  for (unsigned int i = 0; i < lpad; i++) {
    auto pad_start = pad[2 * i];
    auto pad_end = pad[2 * i + 1];
    param.pads[i] = pad_start;
    param.pads[i + ndim] = pad_end;
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

  auto output = at::empty(shape, self.options());

  std::vector<const at::Tensor*> pt_outputs{&output};
  std::vector<const at::Tensor*> pt_inputs{&self};

  synapse_simple_generic_kernel(
      pt_outputs,
      pt_inputs,
      "pad",
      &param,
      sizeof(param),
      SynapsePassType::FORWARD_PASS);

  PT_KERNEL_END;
  return output;
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

  TORCH_CHECK(
      scale_grad_by_freq == false, "scale_grad_by_value = true not supported")
  TORCH_CHECK(sparse == false, "sparse embedding not supported")
  TORCH_CHECK(padding_idx == -1, "padding index not supported")

  Tensor output;
  if (indices.dim() == 1) {
    output = weight.index_select(0, indices);
  } else {
    auto size = indices.sizes().vec();
    // append size of last N-1 dimensions of weight (assuming its a Nd tensor)
    for (auto d : weight.sizes().slice(1)) {
      size.push_back(d);
    }
    output = weight.index_select(0, indices.view(-1)).view(size);
  }

  PT_KERNEL_END;
  return output;
}

/** @brief Function implements embedding backward (for dense-tensors)
 * @param grad (Tensor) Input gradient for bwd pass
 * @param indices (LongTensor) Tensor containing indices into the embedding
 * matrix
 * @param num_weights (int) Number fo rows in the weight tensor
 * @param padding_idx (int, optional) If given, pads the output with the
 * embedding vector at padding_idx (initialized to zeros) whenever it encounters
 * the index
 * @param scale_grad_by_freq (boolean, optional) If given, this will scale
 * gradients by the inverse of frequency of the words in the mini-batch
 */
Tensor embedding_dense_backward_hpu(
    const Tensor& grad,
    const Tensor& indices,
    int64_t num_weights,
    int64_t padding_idx,
    bool scale_grad_by_freq) {
  PT_KERNEL_BEGIN;

  TORCH_CHECK(
      scale_grad_by_freq == false, "scale_grad_by_value = true not supported")
  TORCH_CHECK(padding_idx == -1, "padding index not supported")

  auto grad_weight = at::zeros({num_weights, grad.size(-1)}, grad.options());

  Tensor output;
  if (indices.dim() == 1) {
    output = grad_weight.index_put_(indices, grad);
  } else {
    std::vector<int64_t> size{-1, grad.size(-1)};
    output = grad_weight.index_put_(indices.view(-1), grad.view(size));
  }

  PT_KERNEL_END;
  return output;
}

void EmbeddingBagSumOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    bool is_output_persistent) {
  TORCH_CHECK(
      inputs.size() == 4,
      "Incorrect size of inputs expected for gather2d operator");
  TORCH_CHECK(inputs[0].isTensor(), "Input arg1 type expected to be tensor");
  TORCH_CHECK(inputs[1].isTensor(), "Input arg2 type expected to be tensor");
  TORCH_CHECK(inputs[2].isTensor(), "Input arg3 type expected to be tensor");
  TORCH_CHECK(inputs[3].isInt(), "Input arg4 type expected to be tensor");

  auto input = inputs[0].toTensor();
  auto indices = inputs[1].toTensor();
  auto offsets = inputs[2].toTensor();
  auto valid_count_offset = inputs[3].toInt();

  TORCH_CHECK(indices.dim() <= 1, "index tensor cannot be more than 1D")
  TORCH_CHECK(offsets.dim() <= 1, "offsets tensor cannot be more than 1D")
  TORCH_CHECK(input.dim() == 2, "Input tensor should be 2D")

  TORCH_CHECK(
      valid_count_offset > 0, "valid_count_offset should be greater than 0");
  auto output =
      at::empty({valid_count_offset - 1, input.size(1)}, input.options());

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

  // This extraction of scalar value from tensor not done within
  // AllocateAndAddSynapseNode because this cannot be done on device
  // and doing it on CPU will not work in graph mode.
  TORCH_CHECK(valid_count.numel() == 2, "valid_count should have two elements")
  auto data_ptr = static_cast<int64_t*>(valid_count.to("cpu").data_ptr());
  auto valid_count_offset = data_ptr[1]; // valid offset

  at::ScalarType scalar_type = input.scalar_type();
  std::string node_type;
  if (kernel_mode == 0) {
    node_type = "embedding_bag_sum_2d_fwd_" +
        habana_helpers::name_suffix_from_type(scalar_type);
    ;
  } else {
    node_type = "embedding_bag_sum_small_lengths_2d_fwd_" +
        habana_helpers::name_suffix_from_type(scalar_type);
    ;
  }

  size_t device_id = input.device().index();

  EmbeddingBagSumOperator Op(device_id, node_type);
  // Create Graph
  auto graph = habana_helpers::create_graph(device_id, node_type);

  // Assign Inputs to the Operator
  std::vector<const at::Tensor*> pt_inputs{
      &input, &indices_i32, &offsets_i32, &valid_count_i32};
  Op.AllocateSynapseInputs(graph, pt_inputs, true);

  // Build Params for the graph
  std::vector<c10::IValue> stack = {IValue(input),
                                    IValue(indices_i32),
                                    IValue(offsets_i32),
                                    IValue(valid_count_offset)};
  Op.AllocateAndAddSynapseNode(graph, stack, true);

  // compile and execute the graph
  Op.Compile(graph);

  std::vector<at::Tensor> out = Op.GetOutputs();
  TORCH_CHECK(out.size() == 1, "Incorrect size of outputs");

  PT_KERNEL_END;
  return out.at(0);
}

static auto registry =
    torch::RegisterOperators()
        .op(torch::RegisterOperators::options()
                .schema(
                    "aten::_embedding_bag(Tensor weight, Tensor indices, Tensor offsets, bool scale_grad_by_freq=False, int mode=0, bool sparse=False, Tensor? per_sample_weights=None, bool include_last_offset=False) -> (Tensor, Tensor, Tensor, Tensor)")
                .impl_unboxedOnlyKernel<
                    decltype(embedding_bag_hpu),
                    &embedding_bag_hpu>(DispatchKey::HABANATensorId)
                .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
        .op(torch::RegisterOperators::options()
                .schema(
                    "aten::_embedding_bag_dense_backward(Tensor grad, Tensor indices, Tensor offsets, Tensor offset2bag, Tensor bag_size, Tensor maximum_indices, int num_weights, bool scale_grad_by_freq, int mode, Tensor? per_sample_weights) -> Tensor")
                .impl_unboxedOnlyKernel<
                    decltype(embedding_bag_bwd_hpu),
                    &embedding_bag_bwd_hpu>(DispatchKey::HABANATensorId)
                .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
        .op(torch::RegisterOperators::options()
                .schema(
                    "aten::constant_pad_nd(Tensor self, int[] pad, Scalar value=0) -> Tensor")
                .impl_unboxedOnlyKernel<
                    decltype(constant_pad_hpu),
                    &constant_pad_hpu>(DispatchKey::HABANATensorId)
                .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
        .op(torch::RegisterOperators::options()
                .schema(
                    "aten::embedding(Tensor weight, Tensor indices, int padding_idx=-1, bool scale_grad_by_freq=False, bool sparse=False) -> Tensor")
                .impl_unboxedOnlyKernel<
                    decltype(embedding_hpu),
                    &embedding_hpu>(DispatchKey::HABANATensorId)
                .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
        .op(torch::RegisterOperators::options()
                .schema(
                    "aten::embedding_dense_backward(Tensor grad_output, Tensor indices, int num_weights, int padding_idx, bool scale_grad_by_freq) -> Tensor")
                .impl_unboxedOnlyKernel<
                    decltype(embedding_dense_backward_hpu),
                    &embedding_dense_backward_hpu>(DispatchKey::HABANATensorId)
                .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA));
