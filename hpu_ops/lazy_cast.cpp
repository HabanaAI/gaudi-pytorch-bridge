/******************************************************************************
 * Copyright (C) 2022 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */

#include "hpu_ops/lazy_cast.h"
#include "habana_helpers/cast_sequence.h"

namespace habana {

LazyCast::LazyCast(int device_id, c10::ScalarType scalar_type)
    : OpBackend(device_id, "lazy_cast_guid", scalar_type, {0}, {}, {}, false) {
  SetOutputTypeStackIdx(1);
}

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

} // namespace habana

static auto& CastKernelRegistry =
    habana::KernelRegistry()
        .add("hpu::cast", KERNEL_FN_GLOBAL(habana::LazyCast))
        .add("hpu::habana_cast_sr_mode", KERNEL_FN_GLOBAL(habana::LazyCast));