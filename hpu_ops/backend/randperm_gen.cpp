/******************************************************************************
 * Copyright (C) 2023-2024 HabanaLabs, Ltd.
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
#include "hpu_ops/habana_random_ops.h"

namespace habana {
synapse_helpers::tensor RandPermCommon(
    OpBackend* op,
    synapse_helpers::graph& graph,
    synTensor seed_tensor,
    c10::ScalarType out_dtype,
    std::vector<int64_t> out_shape,
    int n) {
  size_t size = 0;
  c10::ScalarType tpc_supported_randperm_dtype =
      ((common::IsInt64Supported() && (out_dtype == c10::ScalarType::Long))
           ? c10::ScalarType::Long
           : c10::ScalarType::Int);

  auto params =
      FillArangeParamsInternal(0, n, 1, tpc_supported_randperm_dtype, size);
  int start = 0;
  int end = n;
  int step = 1;
  auto arange_op = ArangeCommon(
      op,
      graph,
      start,
      end,
      step,
      tpc_supported_randperm_dtype,
      std::nullopt, // TBD: NOTE: This needs to be changed for torch.compile DS
      std::nullopt, // TBD: NOTE: This needs to be changed for torch.compile DS
      get_guid_with_precision("range", tpc_supported_randperm_dtype),
      out_shape,
      params,
      size,
      c10::nullopt);

  std::vector<synTensor> inputs;
  inputs.emplace_back(arange_op.get());
  inputs.emplace_back(seed_tensor);

  if (out_dtype == tpc_supported_randperm_dtype) {
    auto randperm = OpBackend::BuildNode(
        op,
        graph,
        {std::move(get_guid_with_precision(
             "random_shuffle", tpc_supported_randperm_dtype)),
         std::move(inputs),
         {{out_shape, tpc_supported_randperm_dtype, 0}}});
    return std::move(randperm[0]);
  } else {
    auto randperm = OpBackend::BuildNode(
        op,
        graph,
        {std::move(get_guid_with_precision(
             "random_shuffle", tpc_supported_randperm_dtype)),
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
          {{out_shape, out_dtype, 0}},
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
          0);
      return castop;
    }
  }
}

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
        (common::IsInt64Supported() ? c10::ScalarType::Long
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
  auto out_dtype = meta.dtype;
  auto out_shape = meta.shape;

  synTensor seedTensor;
  if (stack.at(1).isTensor()) {
    seedTensor = syn_in(0);
  } else {
    seedTensor = syn_seed();
  }

  syn_out(0) = RandPermCommon(this, graph, seedTensor, out_dtype, out_shape, n);
}

//===----------------------------------------------------------------------===//
// This is the implementation of custom RandPerm op in `torch.compile`
//===----------------------------------------------------------------------===//
OutputMetaDataVector HabanaRandPermMeta(const at::Stack& stack) {
  OutputMetaData meta;

  meta.shape = {stack.at(1).toInt()};

  unsigned dtype_index = 2;
  meta.dtype = stack.at(dtype_index)
                   .toOptional<at::ScalarType>()
                   .value_or(c10::ScalarType::Long);
  return {meta};
}

void HabanaRandPermOp::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  HABANA_ASSERT(
      stack.at(0).isTensor(),
      "For a custom schema(Randperm) seed tensor should be the",
      "first argument.");
  int n = stack.at(1).toInt();
  const auto meta = HabanaRandPermMeta(stack)[0];
  auto out_dtype = meta.dtype;
  auto out_shape = meta.shape;
  syn_out(0) = RandPermCommon(this, graph, syn_in(0), out_dtype, out_shape, n);
}

HabanaRandPermOp::HabanaRandPermOp(int device_id, c10::ScalarType scalar_type)
    : OpBackend(device_id, "randperm", scalar_type, {0}, {}, {}, false) {
  SetOutputMetaFn(HabanaRandPermMeta);
}
} // namespace habana

static const auto& HabanaRandomKernelRegistry = habana::KernelRegistry().add(
    "hpu::habana_randperm",
    KERNEL_FN_GLOBAL(habana::HabanaRandPermOp));