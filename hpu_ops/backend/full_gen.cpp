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

#include "hpu_ops/full.h"
namespace habana {

unsigned constexpr SIZE_INDEX = 0;
unsigned constexpr FILL_VALUE_INDEX = 1;
unsigned constexpr DTYPE_INDEX = 2;
unsigned constexpr DEVICE_INDEX = 4;

OutputMetaDataVector FullMeta(const at::Stack& stack) {
  auto dtype = stack.at(DTYPE_INDEX)
                   .toOptional<at::ScalarType>()
                   .value_or(at::get_default_dtype_as_scalartype());

  auto device =
      stack.at(DEVICE_INDEX).toOptional<at::Device>().value_or(at::kHPU);
  TORCH_INTERNAL_ASSERT(device.is_hpu());

  OutputMetaData meta;
  meta.dtype = dtype;
  meta.shape = stack.at(SIZE_INDEX).toIntVector();
  meta.mem_format = at::MemoryFormat::Contiguous;
  return {meta};
}

FullBE::FullBE(int device_id, c10::ScalarType scalar_type)
    : OpBackend(device_id, "None_", scalar_type, {0}, {}, {}, false) {
  SetOutputMetaFn(FullMeta);
}

void FullBE::AddNode(synapse_helpers::graph& graph, const at::Stack& stack) {
  auto fillValue = stack.at(FILL_VALUE_INDEX).toScalar();
  const auto& meta = GetOutputMetaData(0);
  auto result = ConstantHelper(graph, fillValue, meta.dtype, meta.shape, 0);
  syn_out(0) = std::move(result);
}

} // namespace habana

static const auto& FullKernelRegistry = habana::KernelRegistry().add(
    "aten::full",
    KERNEL_FN_GLOBAL(habana::FullBE));
