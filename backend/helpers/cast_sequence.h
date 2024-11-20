/**
* Copyright (c) 2021-2024 Intel Corporation
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
