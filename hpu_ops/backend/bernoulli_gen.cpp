/******************************************************************************
 * Copyright (C) 2021-2024 Habana Labs, Ltd. an Intel Company
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

#include "generated/backend/bernoulli.h"
#include "hpu_ops/habana_random_ops.h"

namespace habana {

namespace {
auto bernoulli_impl(
    OpBackend* op,
    synapse_helpers::graph& graph,
    std::vector<synTensor> syn_inputs,
    at::IntArrayRef output_shape,
    at::ScalarType dtype,
    const std::shared_ptr<void>& params,
    size_t params_size) {
  op->CreateShapeTensorInput(graph, dtype, output_shape, syn_inputs);

  auto bernoulli = OpBackend::BuildNode(
      op,
      graph,
      {get_guid_with_precision("pt_bernoulli", dtype),
       std::move(syn_inputs),
       {{output_shape, dtype, 0}},
       params.get(),
       params_size});

  return bernoulli;
}
} // namespace

std::shared_ptr<void> FillHabanaBernoulliParams(
    const at::Stack& stack,
    size_t& size) {
  PARAMS_STUB(ns_RandomBernoulli::ParamsV2);

  params->probability = stack.at(2).toScalar().toFloat();

  return params;
}

std::shared_ptr<void> FillBernoulliWithPParams(
    const at::Stack& stack,
    size_t& size) {
  PARAMS_STUB(ns_RandomBernoulli::ParamsV2);

  if (stack.at(1).isScalar()) {
    params->probability = stack.at(1).toScalar().toFloat();
  }

  return params;
}

OutputMetaDataVector HabanaBernoulliMeta(const at::Stack& stack) {
  const auto& self = stack_tensor(stack, 1);

  OutputMetaData meta;

  meta.shape = self.sizes().vec();
  meta.dtype = self.scalar_type();

  return {meta};
}

OutputMetaDataVector HabanaBernoulliSizeMeta(const at::Stack& stack) {
  OutputMetaData meta;

  meta.shape = stack.at(1).toIntVector();
  meta.dtype =
      stack.at(3).toOptional<at::ScalarType>().value_or(at::ScalarType::Float);
  meta.layout = stack.at(4).toOptional<at::Layout>().value_or(at::kStrided);

  return {meta};
}

void Bernoulli::AddNode(synapse_helpers::graph& graph, const at::Stack& stack) {
  auto seed = stack.at(1).isTensor() ? syn_in(1) : syn_seed();
  std::vector<synTensor> syn_inputs = {syn_in(0), seed};
  auto output_shape = stack_tensor(stack, 0).sizes();
  auto dtype = ScalarType();
  PARAMS_STUB_VARS(ns_RandomBernoulli::ParamsV2, params, params_size);

  syn_out(0) = std::move(bernoulli_impl(
      this,
      graph,
      std::move(syn_inputs),
      output_shape,
      dtype,
      params,
      params_size)[0]);
}

void BernoulliOut::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  std::vector<synTensor> syn_inputs = {syn_in(0), syn_in(1)};
  auto output_shape = stack_tensor(stack, 0).sizes();
  auto dtype = ScalarType();
  PARAMS_STUB_VARS(ns_RandomBernoulli::ParamsV2, params, params_size);

  syn_out(0) = std::move(bernoulli_impl(
      this,
      graph,
      std::move(syn_inputs),
      output_shape,
      dtype,
      params,
      params_size)[0]);
}

void BernoulliWithP::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  auto output_shape = stack_tensor(stack, 0).sizes();
  auto dtype = ScalarType();

  if (stack.at(1).isScalar()) {
    std::vector<synTensor> syn_inputs = {nullptr, syn_in(1)};
    size_t params_size = 0;
    auto params = FillParams(stack, params_size);

    syn_out(0) = std::move(bernoulli_impl(
        this,
        graph,
        std::move(syn_inputs),
        output_shape,
        dtype,
        params,
        params_size)[0]);
  } else {
    std::vector<synTensor> syn_inputs = {syn_in(1), syn_in(2)};
    PARAMS_STUB_VARS(ns_RandomBernoulli::ParamsV2, params, params_size);

    syn_out(0) = std::move(bernoulli_impl(
        this,
        graph,
        std::move(syn_inputs),
        output_shape,
        dtype,
        params,
        params_size)[0]);
  }
}

HabanaBernoulli::HabanaBernoulli(int device_id, c10::ScalarType scalar_type)
    : OpBackend(
          device_id,
          "habana_bernoulli",
          scalar_type,
          {1},
          {},
          {},
          false) {
  SetOutputMetaFn(HabanaBernoulliMeta);
}

void HabanaBernoulli::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  const auto& self = stack_tensor(stack, 1);
  auto syn_inputs = std::vector<synTensor>{syn_in(1), syn_in(0)};
  auto meta = OutputMeta(stack)[0];
  PARAMS_STUB_VARS(ns_RandomBernoulli::ParamsV2, params, params_size);

  syn_out(0) = std::move(bernoulli_impl(
      this,
      graph,
      std::move(syn_inputs),
      meta.shape,
      meta.dtype,
      params,
      params_size)[0]);
}

HabanaBernoulliP::HabanaBernoulliP(int device_id, c10::ScalarType scalar_type)
    : OpBackend(
          device_id,
          "habana_bernoulli_p",
          scalar_type,
          {1},
          {},
          {},
          false) {
  SetOutputMetaFn(HabanaBernoulliMeta);
  SetFillParams(FillHabanaBernoulliParams);
}

void HabanaBernoulliP::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  const auto& self = stack_tensor(stack, 1);
  auto syn_inputs = std::vector<synTensor>{nullptr, syn_in(0)};
  auto meta = OutputMeta(stack)[0];
  size_t params_size = 0;
  auto params = FillParams(stack, params_size);

  syn_out(0) = std::move(bernoulli_impl(
      this,
      graph,
      std::move(syn_inputs),
      meta.shape,
      meta.dtype,
      params,
      params_size)[0]);
}

HabanaBernoulliSize::HabanaBernoulliSize(
    int device_id,
    c10::ScalarType scalar_type)
    : OpBackend(
          device_id,
          "habana_bernoulli_size",
          scalar_type,
          {0},
          {},
          {},
          false) {
  SetOutputMetaFn(HabanaBernoulliSizeMeta);
  SetFillParams(FillHabanaBernoulliParams);
}

void HabanaBernoulliSize::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  auto syn_inputs = std::vector<synTensor>{nullptr, syn_in(0)};
  auto meta = OutputMeta(stack)[0];
  size_t params_size = 0;
  auto params = FillParams(stack, params_size);

  syn_out(0) = std::move(bernoulli_impl(
      this,
      graph,
      std::move(syn_inputs),
      meta.shape,
      meta.dtype,
      params,
      params_size)[0]);
}

HabanaBernoulliTensor::HabanaBernoulliTensor(
    int device_id,
    c10::ScalarType scalar_type)
    : OpBackend(
          device_id,
          "habana_bernoulli_tensor",
          scalar_type,
          {1},
          {},
          {},
          false) {
  SetOutputMetaFn(HabanaBernoulliMeta);
}

void HabanaBernoulliTensor::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  const auto& self = stack_tensor(stack, 1);
  auto syn_inputs = std::vector<synTensor>{syn_in(2), syn_in(0)};
  auto meta = OutputMeta(stack)[0];
  PARAMS_STUB_VARS(ns_RandomBernoulli::ParamsV2, params, params_size);

  syn_out(0) = std::move(bernoulli_impl(
      this,
      graph,
      std::move(syn_inputs),
      meta.shape,
      meta.dtype,
      params,
      params_size)[0]);
}

} // namespace habana

static const auto& HabanaRandomKernelRegistry =
    habana::KernelRegistry()
        .add(
            "hpu::habana_bernoulli_seed",
            KERNEL_FN_GLOBAL(habana::HabanaBernoulli))
        .add(
            "hpu::habana_bernoulli_seed.p",
            KERNEL_FN_GLOBAL(habana::HabanaBernoulliP))
        .add(
            "hpu::habana_bernoulli_seed.Size",
            KERNEL_FN_GLOBAL(habana::HabanaBernoulliSize))
        .add(
            "hpu::habana_bernoulli_seed.Tensor",
            KERNEL_FN_GLOBAL(habana::HabanaBernoulliTensor));
