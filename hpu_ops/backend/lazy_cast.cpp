/*******************************************************************************
 * Copyright (C) 2022-2023 Habana Labs, Ltd. an Intel Company
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

#include "hpu_ops/lazy_cast.h"
#include "backend/helpers/cast_sequence.h"
#include "hpu_ops/hpu_op_helper.h"

namespace habana {

LazyCast::LazyCast(int device_id, c10::ScalarType scalar_type)
    : OpBackend(device_id, "lazy_cast_guid", scalar_type, {0}, {}, {}, false) {}

void LazyCast::AddNode(synapse_helpers::graph& graph, const at::Stack& stack) {
  TORCH_CHECK(
      stack.size() >= 2 && stack.size() <= 4,
      "Incorrect size of inputs expected for cast operator");

  TORCH_CHECK(
      stack[0].isTensor(),
      "Input arg1 expected to be tensor for toDtype operator");

  auto self = stack_tensor(stack, 0);
  auto src_type = self.scalar_type();
  auto dst_type = stack.at(1).toScalarType();
  auto sizes = self.sizes();
  bool stochastic_rounding_override = false;
  int sr_seed{0};

  if (stack.size() > 2) {
    TORCH_CHECK(
        stack[2].isBool(), "Input arg2 expected to be Bool for cast operator");
    stochastic_rounding_override = stack[2].toBool();
  }

  if (stack.size() > 3) {
    TORCH_CHECK(
        stack[3].isInt(), "Input arg3 expected to be Int for cast operator");
    sr_seed = stack[3].toInt();
  }

  auto src_type_cast_type = habana_helpers::DataTypeToCastType(src_type);
  auto dst_type_cast_type = habana_helpers::DataTypeToCastType(dst_type);

  if (src_type_cast_type == dst_type_cast_type) {
    auto out =
        BuildOp(graph, "identity", {syn_in(0)}, {{sizes, ScalarType(), 0}});
    syn_out(0) = std::move(out.at(0));
  } else {
    auto out = CastHelper(
        graph,
        syn_in(0),
        sizes,
        src_type,
        dst_type,
        0,
        stochastic_rounding_override,
        sr_seed);
    syn_out(0) = std::move(out);
  }
}

CopyFrom::CopyFrom(int device_id, c10::ScalarType scalar_type)
    : OpBackend(device_id, {}, scalar_type, {}, {}, {}, true) {}

void CopyFrom::AddNode(synapse_helpers::graph& graph, const at::Stack& stack) {
  auto src = stack_tensor(stack, 0);
  auto src_type = src.scalar_type();
  auto dst_type = stack_tensor(stack, 1).scalar_type();

  auto src_type_cast_type = habana_helpers::DataTypeToCastType(src_type);
  auto dst_type_cast_type = habana_helpers::DataTypeToCastType(dst_type);

  if (src_type_cast_type == dst_type_cast_type) {
    auto out = BuildOp(
        graph, "identity", {syn_in(0)}, {{src.sizes(), ScalarType(), 0}});
    syn_out(0) = std::move(out.at(0));
  } else {
    auto out = CastHelper(graph, syn_in(0), src.sizes(), src_type, dst_type, 0);
    syn_out(0) = std::move(out);
  }
}
} // namespace habana

static const auto& CastKernelRegistry =
    habana::KernelRegistry()
        .add("hpu::_copy_from", KERNEL_FN_GLOBAL(habana::CopyFrom))
        .add("hpu::cast", KERNEL_FN_GLOBAL(habana::LazyCast))
        .add("hpu::habana_cast_sr_mode", KERNEL_FN_GLOBAL(habana::LazyCast));
