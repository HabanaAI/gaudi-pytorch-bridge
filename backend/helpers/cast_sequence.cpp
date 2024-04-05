/*******************************************************************************
 * Copyright (C) 2022-2024 Habana Labs, Ltd. an Intel Company
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

#include "backend/helpers/cast_sequence.h"
#include <vector>

#include "backend/habana_device/hpu_cached_devices.h"
#include "backend/helpers/enum_mapping_table.h"
#include "common/utils.h"
#include "pytorch_helpers/habana_helpers/dtype_helpers.h"
#include "pytorch_helpers/habana_helpers/pt_version_check.h"

namespace habana_helpers {

namespace {

using CastStage = absl::optional<CastType>;

CastStage get_cast_stage(CastTypes cast_types, synDeviceType syn_device_type) {
  // DON'T EDIT THE FOLLOWING TABLES MANUALLY UNLESS YOU HAVE TO
  // Use pytorch-integration/scripts/generate_cast_node_cpp_get_cast_stage.py
  //
  // tpc_kernels/src/kernel_factory_gaudi.cpp
  // CastKernel::SRC_to_DST
  //
  // ======== gaudi ========

  // cast
  // fr/to f32 bf16 i8 i16 i32 i64 u8
  // f32     *    X  X   -   X   -  -
  // bf16    X    *  -   -   -   -  -
  // i8      X    X  *   X   X   -  -
  // i16     -    -  X   *   X   -  -
  // i32     X    X  X   X   *   X  X
  // i64     -    -  -   -   X   *  -
  // u8      X    -  -   -   X   -  *

  // clang-format off
#define OK  CastStage {}
#define F32 CastStage { CastType::f32 }
#define I32 CastStage { CastType::i32 }
  // clang-format on

  // TODO: SW-35847 Remove indirect casting
  using LineT = EnumMappingTable<CastType, CastStage>;
  static const EnumMappingTable<CastType, LineT> cast_stage_matrix_gaudi = {
      // clang-format off
      //              to:    f32  bf16   i8  i16  i32  i64   u8
      /* from  f32 */ LineT{  OK,   OK,  OK, I32,  OK, I32, I32 },
      /* from bf16 */ LineT{  OK,   OK, F32, F32, F32, F32, F32 },
      /* from   i8 */ LineT{  OK,   OK,  OK,  OK,  OK, I32, I32 },
      /* from  i16 */ LineT{ I32,  I32,  OK,  OK,  OK, I32, I32 },
      /* from  i32 */ LineT{  OK,   OK,  OK,  OK,  OK,  OK,  OK },
      /* from  i64 */ LineT{ I32,  I32, I32, I32,  OK,  OK, I32 },
      /* from   u8 */ LineT{  OK,  F32, I32, I32,  OK, I32,  OK },
      // clang-format on
  };

#undef I32
#undef F32
#undef OK

  // ======== gaudi2 ========

  // cast
  // fr/to f32 bf16 i8 i16 i32 i64 u8 f8 hf8 f16
  // f32     *    X  X   X   X   -  X  X   X   X
  // bf16    X    *  X   X   X   -  X  X   X   X
  // i8      X    X  *   X   X   -  X  -   -   X
  // i16     X    X  -   *   X   -  -  -   -   X
  // i32     X    X  X   X   *   X  X  -   -   X
  // i64     -    -  -   -   X   *  -  -   -   -
  // u8      X    X  X   -   X   -  *  -   -   X
  // f8      X    X  -   -   -   -  -  *   -   -
  // hf8     X    X  -   -   -   -  -  -   *   -
  // f16     X    X  X   X   X   -  X  -   -   *

  // clang-format off
#define OK   CastStage {}
#define BF16 CastStage { CastType::bf16 }
#define F32  CastStage { CastType::f32  }
#define I32  CastStage { CastType::i32  }
  // clang-format on

  // TODO: SW-35847 Remove indirect casting
  using LineT = EnumMappingTable<CastType, CastStage>;
  static const EnumMappingTable<CastType, LineT> cast_stage_matrix_gaudi2 = {
      // clang-format off
      //              to:    f32  bf16    i8   i16  i32  i64    u8    f8   hf8   f16
      /* from  f32 */ LineT{  OK,   OK,   OK,   OK,  OK, I32,   OK,   OK,   OK,   OK },
      /* from bf16 */ LineT{  OK,   OK,   OK,   OK,  OK, I32,   OK,   OK,   OK,   OK },
      /* from   i8 */ LineT{  OK,   OK,   OK,   OK,  OK, I32,   OK, BF16, BF16,   OK },
      /* from  i16 */ LineT{  OK,   OK,  I32,   OK,  OK, I32,  I32, BF16, BF16,   OK },
      /* from  i32 */ LineT{  OK,   OK,   OK,   OK,  OK,  OK,   OK,  F32,  F32,   OK },
      /* from  i64 */ LineT{ I32,  I32,  I32,  I32,  OK,  OK,  I32,  I32,  I32,  I32 },
      /* from   u8 */ LineT{  OK,   OK,   OK,  I32,  OK, I32,   OK, BF16, BF16,   OK },
      /* from   f8 */ LineT{  OK,   OK, BF16, BF16, F32, F32, BF16,   OK, BF16, BF16 },
      /* from  hf8 */ LineT{  OK,   OK, BF16, BF16, F32, F32, BF16, BF16,   OK, BF16 },
      /* from  f16 */ LineT{  OK,   OK,   OK,   OK,  OK, I32,   OK, BF16, BF16,   OK },
      // clang-format on
  };

#undef I32
#undef F32
#undef BF16
#undef OK

  // ======== gaudi3 ========

  // cast
  // fr/to f32 bf16 i8 i16 i32 i64 u8 f8 hf8 f16
  // f32     *    X  X   X   X   -  X  X   X   X
  // bf16    X    *  X   X   X   -  X  X   X   X
  // i8      X    X  *   X   X   -  X  -   -   X
  // i16     X    X  -   *   X   -  -  -   -   X
  // i32     X    X  X   X   *   X  X  -   -   X
  // i64     -    -  -   -   X   *  -  -   -   -
  // u8      X    X  X   -   X   -  *  -   -   X
  // f8      X    X  -   -   -   -  -  *   -   -
  // hf8     X    X  -   -   -   -  -  -   *   -
  // f16     X    X  X   X   X   -  X  -   -   *

  // clang-format off
#define OK   CastStage {}
#define BF16 CastStage { CastType::bf16 }
#define F32  CastStage { CastType::f32  }
#define I32  CastStage { CastType::i32  }
  // clang-format on

  // TODO: SW-35847 Remove indirect casting
  using LineT = EnumMappingTable<CastType, CastStage>;
  static const EnumMappingTable<CastType, LineT> cast_stage_matrix_gaudi3 = {
      // clang-format off
      //              to:    f32  bf16    i8   i16  i32  i64    u8    f8   hf8   f16
      /* from  f32 */ LineT{  OK,   OK,   OK,   OK,  OK, I32,   OK,   OK,   OK,   OK },
      /* from bf16 */ LineT{  OK,   OK,   OK,   OK,  OK, I32,   OK,   OK,   OK,   OK },
      /* from   i8 */ LineT{  OK,   OK,   OK,   OK,  OK, I32,   OK, BF16, BF16,   OK },
      /* from  i16 */ LineT{  OK,   OK,  I32,   OK,  OK, I32,  I32, BF16, BF16,   OK },
      /* from  i32 */ LineT{  OK,   OK,   OK,   OK,  OK,  OK,   OK,  F32,  F32,   OK },
      /* from  i64 */ LineT{ I32,  I32,  I32,  I32,  OK,  OK,  I32,  I32,  I32,  I32 },
      /* from   u8 */ LineT{  OK,   OK,   OK,  I32,  OK, I32,   OK, BF16, BF16,   OK },
      /* from   f8 */ LineT{  OK,   OK, BF16, BF16, F32, F32, BF16,   OK, BF16, BF16 },
      /* from  hf8 */ LineT{  OK,   OK, BF16, BF16, F32, F32, BF16, BF16,   OK, BF16 },
      /* from  f16 */ LineT{  OK,   OK,   OK,   OK,  OK, I32,   OK, BF16, BF16,   OK },
      // clang-format on
  };

#undef I32
#undef F32
#undef BF16
#undef OK

  EnumMappingTable<CastType, LineT> cast_stage_matrix;
  switch (syn_device_type) {
    case synDeviceType::synDeviceGaudi:
      cast_stage_matrix = cast_stage_matrix_gaudi;
      break;
    case synDeviceType::synDeviceGaudi2:
      cast_stage_matrix = cast_stage_matrix_gaudi2;
      break;
    case synDeviceType::synDeviceGaudi3:
      cast_stage_matrix = cast_stage_matrix_gaudi3;
      break;
    default:
      HABANA_ASSERT(false, "Unknown device: ", syn_device_type);
      break;
  }

  return cast_stage_matrix[cast_types.from_][cast_types.to_];
}

auto get_cast_type_for_long() {
  if (common::IsInt64Supported()) {
    return CastType::i64;
  }
  return CastType::i32;
}

} // namespace

std::ostream& operator<<(std::ostream& os, const CastType& obj) {
  os << static_cast<std::underlying_type<CastType>::type>(obj);
  return os;
}

CastType DataTypeToCastType(const at::ScalarType& dt) {
  switch (dt) {
    case at::ScalarType::Double:
    case at::ScalarType::Float:
      return CastType::f32;
    case at::ScalarType::BFloat16:
      return CastType::bf16;
    case at::ScalarType::Half:
      return CastType::fp16;
#if HAVE_FP8_SUPPORT
    case at::ScalarType::Float8_e5m2:
      return CastType::f8;
    case at::ScalarType::Float8_e4m3fn:
      return CastType::hf8;
#endif
    case at::ScalarType::Char:
    case at::ScalarType::Bool:
      return CastType::i8;
    case at::ScalarType::Short:
      return CastType::i16;
    case at::ScalarType::Int:
      return CastType::i32;
    case at::ScalarType::Long:
      return get_cast_type_for_long();
    case at::ScalarType::Byte:
      return CastType::u8;
    default:
      HABANA_ASSERT(false, "Unknown data type: ", dt);
      return CastType::u8;
  }
}

at::ScalarType CastTypeToDataType(CastType ct) {
  switch (ct) {
    case CastType::f32:
      return at::ScalarType::Float;
    case CastType::bf16:
      return at::ScalarType::BFloat16;
    case CastType::fp16:
      return at::ScalarType::Half;
#if HAVE_FP8_SUPPORT
    case CastType::f8:
      return at::ScalarType::Float8_e5m2;
    case CastType::hf8:
      return at::ScalarType::Float8_e4m3fn;
#endif
    case CastType::i8:
      return at::ScalarType::Char;
    case CastType::i16:
      return at::ScalarType::Short;
    case CastType::i32:
      return at::ScalarType::Int;
    case CastType::i64:
      return at::ScalarType::Long;
    case CastType::u8:
      return at::ScalarType::Byte;
    default:
      HABANA_ASSERT(false, "Unknown data type: ", ct);
      return at::ScalarType::Undefined;
  }
}

std::vector<CastTypes> get_cast_sequence(
    CastTypes cast_types,
    synDeviceType syn_device_type) {
  std::vector<CastTypes> vec;
  while (true) {
    CastStage cast_stage = get_cast_stage(cast_types, syn_device_type);
    if (cast_stage) {
      vec.emplace_back(CastTypes{cast_types.from_, *cast_stage});
      cast_types.from_ = *cast_stage;
    } else {
      break;
    }
  }
  vec.emplace_back(cast_types);
  return vec;
}

std::vector<CastTypes> get_cast_sequence(CastTypes cast_types) {
  auto& device = habana::HPURegistrar::get_device();
  return get_cast_sequence(cast_types, device.type());
}

CastF32RoundMode_t get_cast_rounding_mode(c10::ScalarType dst_dtype) {
  if (c10::isIntegralType(dst_dtype, true)) {
    return CAST_ROUND_ZERO;
  }

  return CAST_ROUND_HALF_NE;
}

} // namespace habana_helpers
