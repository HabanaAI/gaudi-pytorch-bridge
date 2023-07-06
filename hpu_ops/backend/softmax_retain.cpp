/******************************************************************************
 * Copyright (C) 2023 Habana Labs, Ltd. an Intel Company
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

#include "hpu_ops/softmax_retain.h"

namespace habana {

SoftmaxRetainProducer::SoftmaxRetainProducer(
    int device_id,
    c10::ScalarType scalar_type)
    : OpBackend(device_id, "softmax", scalar_type, {0, 0, 0}, {}, {}, false) {}

void SoftmaxRetainProducer::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  StackGetter stackGetter(stack, "SoftmaxRetainProducer::AddNode");
  auto self = getNextInput<TensorsPair>(stackGetter);

  auto out_shape = self.pt_t.sizes().vec();
  auto retain_out_shape = out_shape;
  retain_out_shape.back() = 1;

  ns_Softmax::ParamsV6 params{};
  params.dim = get_dim_in_tpc_order(-1, self.pt_t.dim());
  params.triangularMode = 1;

  auto output = OpBackend::BuildNode(
      this,
      graph,
      {GetGuid(),
       {self.syn_t},
       {{out_shape, ScalarType(), 0},
        {retain_out_shape, ScalarType(), 1},
        {retain_out_shape, at::ScalarType::Float, 2}},
       &params,
       sizeof(params)});

  syn_out(0) = std::move(output[0]);
  syn_out(1) = std::move(output[1]);
  syn_out(2) = std::move(output[2]);
}

SoftmaxRetainConsumer::SoftmaxRetainConsumer(
    int device_id,
    c10::ScalarType scalar_type)
    : OpBackend(device_id, "softmax", scalar_type, {0}, {}, {}, false) {}

void SoftmaxRetainConsumer::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  StackGetter stackGetter(stack, "SoftmaxRetainConsumer::AddNode");
  auto self = getNextInput<TensorsPair>(stackGetter);
  auto max = getNextInput<TensorsPair>(stackGetter);
  auto exp_sum_recpr = getNextInput<TensorsPair>(stackGetter);

  auto out_shape = self.pt_t.sizes().vec();

  ns_Softmax::ParamsV6 params{};
  params.dim = get_dim_in_tpc_order(-1, self.pt_t.dim());
  params.triangularMode = 1;

  auto output = OpBackend::BuildNode(
      this,
      graph,
      {GetGuid(),
       {self.syn_t, max.syn_t, exp_sum_recpr.syn_t},
       {{out_shape, ScalarType(), 0}},
       &params,
       sizeof(params)});

  syn_out(0) = std::move(output[0]);
}

} // namespace habana

static const auto& SoftmaxRetainKernelRegistry =
    habana::KernelRegistry()
        .add(
            "hpu::retain_softmax_producer",
            KERNEL_FN_GLOBAL(habana::SoftmaxRetainProducer))
        .add(
            "hpu::retain_softmax_consumer",
            KERNEL_FN_GLOBAL(habana::SoftmaxRetainConsumer));
