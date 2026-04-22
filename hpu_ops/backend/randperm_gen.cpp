/**
 * Copyright (c) 2021-2026 Intel Corporation
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
#include "generated/backend/randperm.h"
#include "hpu_ops/backend/arange.h"
#include "hpu_ops/habana_random_ops.h"
#include "hpu_ops/hpu_op_helper.h"
#include "pytorch_helpers/habana_helpers/conversion.h"
#include "pytorch_helpers/habana_helpers/logging.h"

using namespace std::literals;

namespace habana {
synapse_helpers::tensor RandPermCommon(
    OpBackend* op,
    synapse_helpers::graph& graph,
    std::optional<synTensor> arange_synin,
    synTensor seed_tensor,
    c10::ScalarType out_dtype,
    std::vector<int64_t> out_shape,
    int n,
    int final_result_index = 0,
    bool is_compile = false) {
  c10::ScalarType tpc_supported_randperm_dtype =
      ((common::IsInt64Supported() && (out_dtype == c10::ScalarType::Long))
           ? c10::ScalarType::Long
           : c10::ScalarType::Int);

  auto params = FillArangeParamsInternal(0, n, 1, tpc_supported_randperm_dtype);
  int start = 0;
  int end = n;
  int step = 1;
  using namespace std::literals;
  auto arange_op = ArangeCommon(
      op,
      graph,
      start,
      end,
      step,
      tpc_supported_randperm_dtype,
      arange_synin,
      std::nullopt,
      out_shape,
      params,
      std::nullopt,
      is_compile);

  std::vector<synTensor> inputs;
  inputs.emplace_back(arange_op.get());
  inputs.emplace_back(seed_tensor);

  if (out_dtype == tpc_supported_randperm_dtype) {
    auto randperm = OpBackend::BuildNode(
        op,
        graph,
        {get_guid_with_precision(
             "random_shuffle"sv, tpc_supported_randperm_dtype),
         std::move(inputs),
         {{out_shape, tpc_supported_randperm_dtype, final_result_index}}});
    return std::move(randperm[0]);
  } else {
    auto randperm = OpBackend::BuildNode(
        op,
        graph,
        {get_guid_with_precision(
             "random_shuffle"sv, tpc_supported_randperm_dtype),
         std::move(inputs),
         {{out_shape, tpc_supported_randperm_dtype}}});
    if ((out_dtype == c10::ScalarType::Long) &&
        (tpc_supported_randperm_dtype == c10::ScalarType::Int)) {
      // Current handling of Long in the cast builder utilities
      // has issues handling cast_i32_to_i64 though this cast guid
      // doesn't have any documented dependencies on Synapse int64 support
      // and hence the env variable PT_ENABLE_INT64_SUPPORT.
      const std::string cast_guid = "cast_i32_to_i64";
      ns_CastKernel::Params params;
      params.round_mode = CAST_ROUND_ZERO;
      NodeAttr castnode{
          cast_guid,
          {randperm[0].get()},
          {{out_shape, out_dtype, final_result_index}},
          &params,
          sizeof(params)};
      auto castop = OpBackend::BuildNode(op, graph, std::move(castnode));
      return std::move(castop[0]);
    } else {
      auto castop = OpBackend::BuildCast(
          op,
          graph,
          randperm[0].get(),
          out_shape,
          tpc_supported_randperm_dtype,
          out_dtype,
          final_result_index);
      return castop;
    }
  }
}

OutputMetaDataVector RandPermMeta(const at::Stack& stack) {
  OutputMetaDataVector metaVec(1);
  auto& meta = metaVec.front();
  meta.shape = {stack.at(0).toInt()};
  if (stack.size() > 3) {
    unsigned dtype_index;
    if (stack.size() == 5) {
      dtype_index = 1;
    } else {
      dtype_index = 2;
    }
    /*
    Integral dtype conditions
    case-1: When dtype param is not explicitly passed, then pytorch expects
    output type to be Long case-2: When dtype param is passed and if it is
    specified as int32, then that needs to be accounted for.
    */
    meta.dtype = stack.at(dtype_index)
                     .toOptional<at::ScalarType>()
                     .value_or(c10::ScalarType::Long);
  } else {
    c10::ScalarType randperm_dtype =
        (common::IsInt64Supported() ? c10::ScalarType::Long
                                    : c10::ScalarType::Int);
    meta.dtype = randperm_dtype;
  }
  return metaVec;
}

SharedMetaDataVector RandPermSharedMeta(
    const at::Stack& stack,
    habana_helpers::HabanaExecutionMode /*unused*/) {
  auto dtype = c10::ScalarType::Int;

  SharedMetaDataVector meta;
  meta.reserve(2);
  auto& range = meta.emplace_back("range");
  range.outputs_data.emplace_back(1, dtype);

  auto seedRank = stack.at(1).isTensor() ? stack_tensor(stack, 1).dim() : 1;
  auto& randomShuffle = meta.emplace_back("random_shuffle_fwd");
  randomShuffle.inputs_data.push_back(range.outputs_data[0]);
  randomShuffle.inputs_data.emplace_back(seedRank, dtype);
  randomShuffle.outputs_data.emplace_back(1, dtype);

  return meta;
}

void RandPermOp::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  auto n = safe_convert<int>(stack.at(0).toInt());
  bool is_compile =
      GetExecutionMode() == habana_helpers::HabanaFrontendTypes::COMPILE;
  const auto meta = RandPermMeta(stack)[0];
  auto out_dtype = meta.dtype;
  auto out_shape = meta.shape;

  synTensor seedTensor;
  if (stack.at(1).isTensor()) {
    seedTensor = syn_in(0);
  } else {
    seedTensor = syn_seed();
  }

  syn_out(0) = RandPermCommon(
      this,
      graph,
      std::nullopt,
      seedTensor,
      out_dtype,
      out_shape,
      n,
      0,
      is_compile);
}

//===----------------------------------------------------------------------===//
// This is the implementation of custom RandPerm op in `torch.compile`
//===----------------------------------------------------------------------===//
OutputMetaDataVector HabanaRandPermMeta(const at::Stack& stack) {
  OutputMetaDataVector metaVec(1);
  auto& meta = metaVec.front();

  meta.shape = {stack.at(1).toInt()};

  unsigned dtype_index = 2;
  meta.dtype = stack.at(dtype_index)
                   .toOptional<at::ScalarType>()
                   .value_or(c10::ScalarType::Long);
  return metaVec;
}

void HabanaRandPerm::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  bool is_compile =
      GetExecutionMode() == habana_helpers::HabanaFrontendTypes::COMPILE;
  HABANA_ASSERT(
      stack.at(0).isTensor(),
      "For a custom schema(Randperm) seed tensor should be the",
      "first argument.");
  auto n = safe_convert<int>(stack.at(1).toInt());
  const auto meta = HabanaRandPermMeta(stack)[0];
  auto out_dtype = meta.dtype;
  auto out_shape = meta.shape;
  syn_out(0) = RandPermCommon(
      this,
      graph,
      std::nullopt,
      syn_in(0),
      out_dtype,
      out_shape,
      n,
      0,
      is_compile);
}

HabanaRandPerm::HabanaRandPerm(int device_id, c10::ScalarType scalar_type)
    : HabanaRandomBase(device_id, "randperm", scalar_type, {0}) {
  SetOutputMetaFn(HabanaRandPermMeta);
}

size_t GetMInMaxSifOffsetRP(bool dry_run, size_t data_size) {
  size_t sif_offset = 0;
  if (dry_run &&
      habana::ShapeInference::GetCurrentPass() ==
          habana::ShapeInfo::InferencePass::MIN_SHAPE) {
    sif_offset = data_size;
  }
  return sif_offset;
}

template <typename T>
std::vector<T> GetArangeH2DParams(at::Tensor& params_t, bool dry_run) {
  std::vector<T> params_data;
  auto data_size = safe_convert<size_t>(params_t.sizes()[0]);
  auto* tmeta{get_tensor_extra_meta(params_t)};
  void* host_ptr = nullptr;
  if (dry_run) {
    host_ptr = tmeta->get_compile_host_ptr();
  } else {
    host_ptr = tmeta->get_host_ptr();
  }

  T* h2d_data = static_cast<T*>(host_ptr);
  size_t sif_offset = GetMInMaxSifOffsetRP(dry_run, data_size);
  h2d_data = h2d_data + sif_offset;
  params_data.reserve(data_size);
  for (size_t i = 0; i < data_size; i++) {
    params_data.push_back(*h2d_data++);
  }
  return params_data;
}

OutputMetaDataVector HabanaRandPermMetaDS(const at::Stack& stack) {
  OutputMetaDataVector metaVec(1);
  auto& meta = metaVec.front();
  // DS Compile Flow
  std::vector<int32_t> params_data;
  at::Tensor params_t = stack[1].toTensor();
  if ((habana::ShapeInference::GetCurrentPass() ==
       habana::ShapeInfo::InferencePass::MIN_SHAPE) ||
      (habana::ShapeInference::GetCurrentPass() ==
       habana::ShapeInfo::InferencePass::MAX_SHAPE)) {
    params_data = GetArangeH2DParams<int32_t>(params_t, true);
  } else {
    params_data = GetArangeH2DParams<int32_t>(params_t, false);
  }
  meta.shape = {params_data[1]};
  unsigned dtype_index = 3;
  meta.dtype = stack.at(dtype_index)
                   .toOptional<at::ScalarType>()
                   .value_or(c10::ScalarType::Long);
  return metaVec;
}

void HabanaRandPermDS::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  HABANA_ASSERT(
      stack.at(0).isTensor(),
      "For a custom schema(Randperm) seed tensor should be the",
      "first argument.");
  const auto meta = HabanaRandPermMetaDS(stack)[0];
  c10::ScalarType tpc_supported_randperm_dtype =
      ((GET_ENV_FLAG_NEW(PT_ENABLE_INT64_SUPPORT) &&
        (meta.dtype == c10::ScalarType::Long))
           ? c10::ScalarType::Long
           : c10::ScalarType::Int);

  auto out_dtype = meta.dtype;
  auto out_shape = meta.shape;
  auto params = FillArangeParamsInternal(0, 1, 1, tpc_supported_randperm_dtype);
  std::vector<int32_t> params_data;
  at::Tensor params_t = stack[1].toTensor();
  if ((habana::ShapeInference::GetCurrentPass() ==
       habana::ShapeInfo::InferencePass::MIN_SHAPE) ||
      (habana::ShapeInference::GetCurrentPass() ==
       habana::ShapeInfo::InferencePass::MAX_SHAPE)) {
    params_data = GetArangeH2DParams<int32_t>(params_t, true);
  } else {
    params_data = GetArangeH2DParams<int32_t>(params_t, false);
  }
  int end = params_data[1];
  if (p_context_->syn_inputs_.size() == 3) {
    EraseSynInput(2);
  }
  syn_out(0) = RandPermCommon(
      this, graph, syn_in(1), syn_in(0), out_dtype, out_shape, end, 0, true);
}

HabanaRandPermDS::HabanaRandPermDS(int device_id, c10::ScalarType scalar_type)
    : HabanaRandomBase(device_id, "random_uniform", scalar_type, {0}) {
  SetOutputMetaFn(HabanaRandPermMetaDS);
}

} // namespace habana

static const auto& HabanaRandomKernelRegistry =
    habana::KernelRegistry()
        .REGISTER_HABANA_RANDOM_OP(randperm, RandPerm)
        .REGISTER_HABANA_RANDOM_OP(randperm_ht, RandPermDS);
