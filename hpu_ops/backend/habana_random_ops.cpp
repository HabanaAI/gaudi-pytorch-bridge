/******************************************************************************
 * Copyright (C) 2024 Habana Labs, Ltd. an Intel Company
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

#include "hpu_ops/habana_random_ops.h"

namespace habana {

namespace {

OutputMetaDataVector HabanaRandOutputMeta(const at::Stack& stack) {
  OutputMetaData meta;
  meta.shape = stack[1].toIntVector();
  meta.dtype =
      stack[2].toOptional<at::ScalarType>().value_or(at::ScalarType::Float);
  meta.layout = stack[3].toOptional<at::Layout>().value_or(at::kStrided);
  return {meta};
}

std::shared_ptr<void> FillHabanaRandParams(const at::Stack&, size_t& size) {
  PARAMS_STUB(ns_RandomUniform::Params);
  params->low = 0.0;
  params->high = 1.0;
  return params;
}

std::shared_ptr<void> FillHabanaRandnParams(const at::Stack&, size_t& size) {
  PARAMS_STUB(ns_RandomNormal::Params);
  params->mean = 0.0;
  params->stddev = 1.0;
  return params;
}

OutputMetaDataVector HabanaRandintOutputMeta(const at::Stack& stack) {
  OutputMetaData meta;
  meta.shape = stack[3].toIntList().vec();
  meta.dtype =
      stack[4].toOptional<at::ScalarType>().value_or(at::ScalarType::Long);
  meta.layout = stack[5].toOptional<at::Layout>().value_or(at::kStrided);
  return {meta};
}

std::shared_ptr<void> FillHabanaRandintParams(
    const at::Stack& stack,
    size_t& size) {
  PARAMS_STUB(ns_RandomUniform::ParamsV2);
  params->low.i = stack[1].toInt();
  params->high.i = stack[2].toInt();
  return params;
}

OutputMetaDataVector HabanaUniformOutputMeta(const at::Stack& stack) {
  OutputMetaData meta;
  auto& input = stack[1].toTensor();
  meta.shape = input.sizes().vec();
  meta.dtype = input.scalar_type();
  return {meta};
}

std::shared_ptr<void> FillHabanaUniformParams(
    const at::Stack& stack,
    size_t& size) {
  PARAMS_STUB(ns_PhiloxRandomUniform::ParamsV3);
  auto low = stack.at(2).toDouble();
  auto high = stack.at(3).toDouble();
  if (stack_tensor(stack, 1).scalar_type() == at::ScalarType::Int) {
    params->low_i = static_cast<int>(low);
    params->high_i = static_cast<int>(high);
  } else {
    params->low = static_cast<float>(low);
    params->high = static_cast<float>(high);
  }
  return params;
}

OutputMetaDataVector HabanaSeedGeneratorOutputMeta(const at::Stack& stack) {
  OutputMetaData meta;
  meta.shape = {stack[2].toInt()};
  meta.dtype = at::ScalarType::Int;
  return {meta};
}

std::shared_ptr<void> FillHabanaSeedGeneratorParams(
    const at::Stack&,
    size_t& size) {
  PARAMS_STUB(ns_PhiloxRandomUniform::ParamsV3);
  params->low_i = 0;
  params->high_i = std::numeric_limits<int32_t>::max();
  return params;
}

} // namespace

HabanaRandBase::HabanaRandBase(
    int device_id,
    c10::ScalarType scalar_type,
    std::string_view kernel_name)
    : OpBackend(
          device_id,
          kernel_name.data(),
          scalar_type,
          {0},
          {},
          {},
          false) {}

void HabanaRandBase::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  const auto output_meta = GetOutputMetaData()[0];
  const auto& output_shape = output_meta.shape;
  const auto& dtype = output_meta.dtype;

  size_t size = 0;
  auto params = FillParams(stack, size);

  update_guid_dtype(guid_, dtype);

  std::vector<synTensor> inputs;
  if (guid_.find("random_normal") != std::string::npos) {
    inputs.push_back(nullptr);
  }
  inputs.push_back(syn_in(0));
  if (guid_.find("habana_seed_generator") != std::string::npos) {
    SetGuid(get_guid_with_precision("philox_random_uniform", dtype));
    inputs.push_back(syn_in(1));
  }

  CreateShapeTensorInput(graph, dtype, output_shape, inputs);

  auto rand = BuildOp(
      graph,
      guid_,
      std::move(inputs),
      {{output_shape, dtype, 0}},
      params.get(),
      size);
  syn_out(0) = std::move(rand[0]);
}

HabanaRand::HabanaRand(int device_id, c10::ScalarType scalar_type)
    : HabanaRandBase(device_id, scalar_type, "random_uniform") {
  SetFillParams(FillHabanaRandParams);
  SetOutputMetaFn(HabanaRandOutputMeta);
}

HabanaRandn::HabanaRandn(int device_id, c10::ScalarType scalar_type)
    : HabanaRandBase(device_id, scalar_type, "random_normal") {
  SetFillParams(FillHabanaRandnParams);
  SetOutputMetaFn(HabanaRandOutputMeta);
}

HabanaRandint::HabanaRandint(int device_id, c10::ScalarType scalar_type)
    : HabanaRandBase(device_id, scalar_type, "random_uniform") {
  SetFillParams(FillHabanaRandintParams);
  SetOutputMetaFn(HabanaRandintOutputMeta);
}

HabanaUniform::HabanaUniform(int device_id, c10::ScalarType scalar_type)
    : HabanaRandBase(device_id, scalar_type, "philox_random_uniform") {
  SetOutputMetaFn(HabanaUniformOutputMeta);
  SetFillParams(FillHabanaUniformParams);
}

HabanaSeedGenerator::HabanaSeedGenerator(
    int device_id,
    c10::ScalarType scalar_type)
    : HabanaRandBase(device_id, scalar_type, "habana_seed_generator") {
  SetOutputMetaFn(HabanaSeedGeneratorOutputMeta);
  SetFillParams(FillHabanaSeedGeneratorParams);
}

} // namespace habana

static const auto& HabanaRandomKernelRegistry =
    habana::KernelRegistry()
        .add("hpu::habana_rand", KERNEL_FN_GLOBAL(habana::HabanaRand))
        .add("hpu::habana_randn", KERNEL_FN_GLOBAL(habana::HabanaRandn))
        .add("hpu::habana_randint", KERNEL_FN_GLOBAL(habana::HabanaRandint))
        .add("hpu::habana_uniform", KERNEL_FN_GLOBAL(habana::HabanaUniform))
        .add(
            "hpu::habana_seed_generator",
            KERNEL_FN_GLOBAL(habana::HabanaSeedGenerator));
