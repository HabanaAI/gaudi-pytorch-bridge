/**
 * Copyright (c) 2021-2025 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "hpu_ops/habana_random_ops.h"
#include "hpu_ops/hpu_op_helper.h"

namespace habana {

namespace {

OutputMetaDataVector HabanaRandOutputMeta(const at::Stack& stack) {
  OutputMetaData meta;
  if (stack.at(1).isTensor()) {
    meta.shape = stack[1].toTensor().sizes().vec();
  } else {
    meta.shape = stack[1].toIntVector();
  }
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
  static const bool use_philox = GET_ENV_FLAG_NEW(PT_HPU_USE_PHILOX_NORMAL);
  PARAMS_STUB(ns_RandomNormal::ParamsV2);
  params->mean = 0.0;
  params->stddev = 1.0;
  params->usePhilox = use_philox;
  return params;
}

OutputMetaDataVector HabanaRandintOutputMeta(const at::Stack& stack) {
  OutputMetaData meta;
  if (stack.at(3).isTensor()) {
    meta.shape = stack[3].toTensor().sizes().vec();
  } else {
    meta.shape = stack[3].toIntList().vec();
  }
  meta.dtype =
      stack[4].toOptional<at::ScalarType>().value_or(at::ScalarType::Long);
  meta.layout = stack[5].toOptional<at::Layout>().value_or(at::kStrided);
  return {meta};
}

std::shared_ptr<void> FillHabanaRandintParams(
    const at::Stack& stack,
    size_t& size) {
  PARAMS_STUB(ns_RandomUniform::ParamsV2);
  const auto dtype =
      stack[4].toOptional<at::ScalarType>().value_or(at::ScalarType::Long);
  if (c10::isFloatingType(dtype)) {
    params->low.f = static_cast<float>(stack[1].toInt());
    params->high.f = static_cast<float>(stack[2].toInt());
  } else {
    params->low.i = stack[1].toInt();
    params->high.i = stack[2].toInt();
  }
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

HabanaRandomBase::HabanaRandomBase(
    int device_id,
    std::string_view kernel_name,
    c10::ScalarType scalar_type,
    std::vector<int> res_ids)
    : OpBackend(
          device_id,
          kernel_name.data(),
          scalar_type,
          std::move(res_ids),
          {},
          {},
          false) {}

using namespace std::literals;

void HabanaRandomBase::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  const auto output_meta = GetOutputMetaData()[0];
  const auto& output_shape = output_meta.shape;
  const auto& dtype = output_meta.dtype;

  size_t size = 0;
  auto rand_params = FillParams(stack, size);

  update_guid_dtype(guid_, dtype);

  std::vector<synTensor> inputs;
  if (guid_.find("random_normal") != std::string::npos) {
    inputs.push_back(nullptr);
  }
  inputs.push_back(syn_in(0));
  if (guid_.find("habana_seed_generator") != std::string::npos) {
    SetGuid(get_guid_with_precision("philox_random_uniform"sv, dtype));
    inputs.push_back(syn_in(1));
  }
  if ((guid_.find("habana_seed_generator") == std::string::npos) &&
      stack.at(2).isTensor()) {
    inputs.push_back(syn_in(1));
  } else {
    CreateShapeTensorInput(graph, dtype, output_shape, inputs);
  }
  auto rand = BuildOp(
      graph,
      guid_,
      std::move(inputs),
      {{output_shape, dtype, 0}},
      rand_params.get(),
      size);
  syn_out(0) = std::move(rand[0]);
}

bool RandDSSTMeta(
    habana_helpers::IShapeList& inputs,
    habana_helpers::IShapeList& outputs) {
  PT_BRIDGE_DEBUG("RandDSSTMeta called");
  static_cast<void>(outputs);
  if (!inputs[2].isTensor()) {
    auto t_size = inputs[1].getTensorShape();
    PT_BRIDGE_DEBUG("RandDSSTMeta constant shape ", t_size);
    habana_helpers::UpdateSTShapeInfo(t_size);
  }

  return true;
}

HabanaRand::HabanaRand(int device_id, c10::ScalarType scalar_type)
    : HabanaRandomBase(device_id, "random_uniform", scalar_type, {1}) {
  SetFillParams(FillHabanaRandParams);
  SetOutputMetaFn(HabanaRandOutputMeta);
  SetSTMetaFn(RandDSSTMeta);
}

HabanaRandn::HabanaRandn(int device_id, c10::ScalarType scalar_type)
    : HabanaRandomBase(device_id, "random_normal", scalar_type, {1}) {
  SetFillParams(FillHabanaRandnParams);
  SetOutputMetaFn(HabanaRandOutputMeta);
  SetSTMetaFn(RandDSSTMeta);
}

bool RandIntDSSTMeta(
    habana_helpers::IShapeList& inputs,
    habana_helpers::IShapeList& outputs) {
  PT_BRIDGE_DEBUG("RandIntDSSTMeta called");
  static_cast<void>(outputs);

  if (inputs[3].isTensor()) {
    auto t_size = inputs[3].getTensorShape();
    PT_BRIDGE_DEBUG("RandIntDSSTMeta constant shape ", t_size);
    habana_helpers::UpdateSTShapeInfo(t_size);
  } else {
    PT_BRIDGE_DEBUG("Rand Int DS meta not supported non tensor input !!!");
    return false;
  }

  return true;
}

HabanaRandint::HabanaRandint(int device_id, c10::ScalarType scalar_type)
    : HabanaRandomBase(device_id, "random_uniform", scalar_type, {1}) {
  SetOutputMetaFn(HabanaRandintOutputMeta);
  SetSTMetaFn(RandIntDSSTMeta);
}

void HabanaRandint::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  const auto output_meta = GetOutputMetaData()[0];
  const auto& outshape = output_meta.shape;
  const auto& dtype = output_meta.dtype;

  size_t size = 0;
  auto rand_params = FillHabanaRandintParams(stack, size);

  update_guid_dtype(guid_, dtype);

  std::vector<synTensor> inputs{syn_in(0)};
  CreateShapeTensorInput(graph, dtype, outshape, inputs);

  std::string post_op_guid = "";
  NodeAttr::NodeOutputAttr out_attr = {outshape, dtype};
  const bool need_convert_i16 = dtype == c10::ScalarType::Byte ||
      dtype == c10::ScalarType::Char || dtype == c10::ScalarType::Bool;
  if (need_convert_i16) {
    post_op_guid =
        dtype == at::ScalarType::Byte ? "cast_i16_to_u8" : "cast_i16_to_i8";
    update_guid_dtype(guid_, "i16");
    out_attr.dtype = c10::ScalarType::Short;
  } else if (c10::isFloatingType(dtype)) {
    post_op_guid = get_guid_with_precision("floor_fwd"sv, dtype);
  } else {
    out_attr.final_result_index = 0;
  }

  auto rand = OpBackend::BuildOp(
      graph, guid_, std::move(inputs), {out_attr}, rand_params.get(), size);
  if (need_convert_i16) {
    PARAMS_STUB(ns_CastKernel::Params);
    // Round down so that the upper limit is not included in the generated seq.
    // The assumption is that the float vaues dont include the upper limit.
    params->round_mode = CAST_ROUND_DOWN;
    syn_out(0) = std::move(OpBackend::BuildOp(
        graph,
        post_op_guid,
        {rand[0].get()},
        {{outshape, dtype, 0}},
        params.get(),
        size)[0]);
  } else if (c10::isFloatingType(dtype)) {
    syn_out(0) = std::move(OpBackend::BuildOp(
        graph, post_op_guid, {rand[0].get()}, {{outshape, dtype, 0}})[0]);
  } else {
    syn_out(0) = std::move(rand[0]);
  }
}

HabanaUniform::HabanaUniform(int device_id, c10::ScalarType scalar_type)
    : HabanaRandomBase(device_id, "philox_random_uniform", scalar_type, {1}) {
  SetOutputMetaFn(HabanaUniformOutputMeta);
  SetFillParams(FillHabanaUniformParams);
  SetSTMetaFn(RandDSSTMeta);
}

bool RandSeedGeneratorDSSTMeta(
    habana_helpers::IShapeList& inputs,
    habana_helpers::IShapeList& outputs) {
  PT_BRIDGE_DEBUG("RandSeedGeneratorDSSTMeta called");
  static_cast<void>(outputs);
  if (inputs[1].isScalar()) {
    auto t_size = inputs[1].getScalar().toInt();
    PT_BRIDGE_DEBUG("RandSeedGeneratorDSSTMeta constant shape ", t_size);
    std::vector<int64_t> shape(1, t_size);
    habana_helpers::UpdateSTShapeInfo(shape);
    return true;
  }

  return false;
}

HabanaSeedGenerator::HabanaSeedGenerator(
    int device_id,
    c10::ScalarType scalar_type)
    : HabanaRandomBase(device_id, "habana_seed_generator", scalar_type, {0}) {
  SetOutputMetaFn(HabanaSeedGeneratorOutputMeta);
  SetFillParams(FillHabanaSeedGeneratorParams);
  SetSTMetaFn(RandSeedGeneratorDSSTMeta);
}

} // namespace habana

static const auto& HabanaRandomKernelRegistry =
    habana::KernelRegistry()
        .REGISTER_HABANA_RANDOM_OP(seed_generator, SeedGenerator)
        .REGISTER_HABANA_RANDOM_OP(rand, Rand)
        .REGISTER_HABANA_RANDOM_OP(rand_st, Rand)
        .REGISTER_HABANA_RANDOM_OP(randn, Randn)
        .REGISTER_HABANA_RANDOM_OP(randn_st, Randn)
        .REGISTER_HABANA_RANDOM_OP(randint, Randint)
        .REGISTER_HABANA_RANDOM_OP(randint_st, Randint)
        .REGISTER_HABANA_RANDOM_OP(uniform, Uniform);
