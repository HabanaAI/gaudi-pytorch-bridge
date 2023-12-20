/******************************************************************************
 * Copyright (C) 2023 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */
#include "generated/backend/arange.h"
#include "generated/backend/randperm.h"
#include "habana_kernels/random_gen_kernels.h"
#include "hpu_ops/backend/arange.h"
#include "hpu_ops/common/arange_gen.h"

namespace habana {
OutputMetaDataVector RandPermMeta(const at::Stack& stack) {
  OutputMetaData meta;
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
        (GET_ENV_FLAG_NEW(PT_ENABLE_INT64_SUPPORT) ? c10::ScalarType::Long
                                                   : c10::ScalarType::Int);
    meta.dtype = randperm_dtype;
  }
  return {meta};
}

void RandPermOp::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  int n = stack.at(0).toInt();
  const auto meta = RandPermMeta(stack)[0];
  size_t size = 0;
  at::Stack arange_stack = {};
  c10::ScalarType tpc_supported_randperm_dtype =
      ((GET_ENV_FLAG_NEW(PT_ENABLE_INT64_SUPPORT) &&
        (meta.dtype == c10::ScalarType::Long))
           ? c10::ScalarType::Long
           : c10::ScalarType::Int);

  auto params =
      FillArangeParamsInternal(0, n, 1, tpc_supported_randperm_dtype, size);
  int start = 0;
  int end = n;
  int step = 1;
  std::optional<synTensor> syn_in0 = std::nullopt;
  std::optional<synTensor> syn_in1 = std::nullopt;
  auto arange_op = ArangeCommon(
      this,
      graph,
      start,
      end,
      step,
      tpc_supported_randperm_dtype,
      syn_in0, // TBD: NOTE: This needs to be changed for torch.compile DS
      syn_in1, // TBD: NOTE: This needs to be changed for torch.compile DS
      get_guid_with_precision("range", tpc_supported_randperm_dtype),
      meta.shape,
      params,
      size,
      c10::nullopt);

  std::vector<synTensor> inputs;
  inputs.push_back(std::move(arange_op.get()));
  if (stack.at(1).isTensor()) {
    inputs.push_back(syn_in(0));
  } else {
    inputs.push_back(syn_seed());
  }
  if (meta.dtype == tpc_supported_randperm_dtype) {
    auto randperm = BuildOp(
        graph,
        get_guid_with_precision("random_shuffle", tpc_supported_randperm_dtype),
        std::move(inputs),
        {{meta.shape, tpc_supported_randperm_dtype, 0}});
    syn_out(0) = std::move(randperm[0]);
  } else {
    auto randperm = BuildOp(
        graph,
        get_guid_with_precision("random_shuffle", tpc_supported_randperm_dtype),
        std::move(inputs),
        {{meta.shape, tpc_supported_randperm_dtype}});
    if ((meta.dtype == c10::ScalarType::Long) &&
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
          {{meta.shape, meta.dtype, 0}},
          &params,
          sizeof(params)};
      auto castop = BuildNode(this, graph, std::move(castnode));
      syn_out(0) = std::move(castop[0]);
    } else {
      auto castop = BuildCast(
          this,
          graph,
          randperm[0].get(),
          meta.shape,
          tpc_supported_randperm_dtype,
          meta.dtype,
          0);
      syn_out(0) = std::move(castop);
    }
  }
}

} // namespace habana
