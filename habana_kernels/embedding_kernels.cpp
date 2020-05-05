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
#include "habana_helpers/logging.h"
#include "habana_helpers/tensor_utils.h"
#include "habana_helpers/unused_macro.h"
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
 ************************************************************************/
std::tuple<Tensor, Tensor, Tensor, Tensor> embedding_bag_hpu(
    const Tensor& weight,
    const Tensor& indices,
    const Tensor& offsets,
    bool scale_grad_by_freq,
    int64_t mode,
    UNUSED bool sparse,
    Tensor& per_sample_weights) {
  LOG_FUNC_BEGIN;

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

  LOG_FUNC_END;

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
  LOG_FUNC_BEGIN;

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

  LOG_FUNC_END;

  return momentum_out;
}

/*static auto registry =
    torch::RegisterOperators()
        .op(torch::RegisterOperators::options()
                .schema(
                    "aten::_embedding_bag(Tensor weight, Tensor indices, Tensor offsets, bool scale_grad_by_freq=False, int mode=0, bool sparse=False, Tensor? per_sample_weights=None) -> (Tensor, Tensor, Tensor, Tensor)")
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
                .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA));
*/