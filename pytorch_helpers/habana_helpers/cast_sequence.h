/******************************************************************************
 * Copyright (C) 2022 Habana Labs, Ltd. an Intel Company
 * All Rights Reserved.
 *
 * Unauthorized copying of this file or any element(s) within it, via any medium
 * is strictly prohibited.
 * This file contains Habana Labs, Ltd. proprietary and confidential information
 * and is subject to the confidentiality and license agreements under which it
 * was provided.
 *
 ******************************************************************************
 */
#pragma once
#include <ATen/Tensor.h>
#include <absl/types/optional.h>
#include <synapse_common_types.h>
#include <cstdint>

namespace habana_helpers {

// clang-format off
#define CAST_TYPE_DATA                                     \
  /* 1. TPC kernel infix / enum class entry */             \
  /* 2. synDataType suffix */                              \
  /* 3. C++ type for type_traits (is_floating_point<T>) */ \
  /* 4. type to which identity node is possible */         \
  /*      1.       2.        3.    4.*/                    \
  ENTRY( f32,   float,    float,  f32)                     \
  ENTRY(bf16,    bf16,    float, bf16)                     \
  ENTRY(  i8,    int8,      int,   u8)                     \
  ENTRY( i16,   int16,      int,  u16)                     \
  ENTRY( i32,   int32,      int,  u32)                     \
  ENTRY( i64,   int64,      int,  u64)                     \
  ENTRY(  u8,   uint8, unsigned,   i8)                     \
  ENTRY( u16,  uint16, unsigned,  i16)                     \
  ENTRY( u32,  uint32, unsigned,  i32)                     \
  ENTRY( u64,  uint64, unsigned,  i64)                     \
  ENTRY(  f8, fp8_152,    float,   f8)
// clang-format on

#define ENTRY(TPC_T, SYN_T, T, I) TPC_T,
enum class CastType : uint8_t { CAST_TYPE_DATA __count };
#undef ENTRY

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

} // namespace habana_helpers