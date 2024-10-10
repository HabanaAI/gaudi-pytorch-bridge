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
#pragma once
#include <ATen/Tensor.h>
#include <absl/types/optional.h>
#include <perf_lib_layer_params.h>
#include <synapse_common_types.h>
#include <cstdint>

namespace habana_helpers {

// clang-format off
#define CAST_TYPE_DATA                        \
  /* TPC kernel infix / enum class entry */   \
  ENTRY( f32)                                 \
  ENTRY(bf16)                                 \
  ENTRY(  i8)                                 \
  ENTRY( i16)                                 \
  ENTRY( i32)                                 \
  ENTRY( i64)                                 \
  ENTRY(  u8)                                 \
  ENTRY(  f8)                                 \
  ENTRY( hf8)                                 \
  ENTRY(fp16)
// clang-format on

#define ENTRY(TPC_T) TPC_T,
enum class CastType : uint8_t { CAST_TYPE_DATA __count };
#undef ENTRY

std::ostream& operator<<(std::ostream& os, const CastType& obj);

struct CastTypes {
  CastType from_;
  CastType to_;

  bool operator==(CastTypes rhs) const {
    return (from_ == rhs.from_) && (to_ == rhs.to_);
  }
  bool operator!=(CastTypes rhs) const {
    return !(*this == rhs);
  }
};

CastType DataTypeToCastType(const at::ScalarType& dt);
at::ScalarType CastTypeToDataType(CastType ct);

std::vector<CastTypes> get_cast_sequence(CastTypes cast_types);

CastF32RoundMode_t get_cast_rounding_mode(c10::ScalarType dst_dtype);
} // namespace habana_helpers
