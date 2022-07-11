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

#include "cast_sequence.h"
#include <vector>
#include "enum_mapping_table.h"
#include "logging.h"
#include "pytorch_helpers/habana_device/hpu_cached_devices.h"

namespace habana_helpers {

namespace {

using CastStage = absl::optional<CastType>;

CastStage get_cast_stage(CastTypes cast_types, synDeviceType syn_device_type) {
  // DON'T EDIT THE FOLLOWING TABLES MANUALLY UNLESS YOU HAVE TO
  // Use tensorflow-training/tools/generate_cast_node_cpp_get_cast_stage.py
  //
  // tpc_kernels/src/kernel_factory_gaudi.cpp
  // CastKernel::SRC_to_DST
  //
  // ======== gaudi ========

  // cast
  // fr/to f32 bf16 i8 i16 i32 i64 u8 u16 u32 u64
  // f32     *    X  X   -   X   -  -   -   X   -
  // bf16    X    *  -   -   -   -  -   -   -   -
  // i8      X    X  *   X   X   -  I   -   -   -
  // i16     -    -  X   *   X   -  -   I   -   -
  // i32     X    X  X   X   *   X  X   -   I   -
  // i64     -    -  -   -   X   *  -   -   -   I
  // u8      X    -  I   -   X   -  *   X   X   -
  // u16     -    -  -   I   -   -  X   *   X   -
  // u32     X    -  -   -   I   -  X   X   *   X
  // u64     -    -  -   -   -   I  -   -   X   *

  // clang-format off
#define OK  CastStage {}
#define F32 CastStage { CastType::f32 }
#define I16 CastStage { CastType::i16 }
#define I32 CastStage { CastType::i32 }
#define I64 CastStage { CastType::i64 }
#define U16 CastStage { CastType::u16 }
#define U32 CastStage { CastType::u32 }
#define U64 CastStage { CastType::u64 }
  // clang-format on

  // TODO: SW-35847 Remove indirect casting
  using LineT = EnumMappingTable<CastType, CastStage>;
  static const EnumMappingTable<CastType, LineT> cast_stage_matrix_gaudi = {
      // clang-format off
      //              to:    f32  bf16   i8  i16  i32  i64   u8  u16  u32  u64
      /* from  f32 */ LineT{  OK,   OK,  OK, I32,  OK, I32, I32, U32,  OK, U32 },
      /* from bf16 */ LineT{  OK,   OK, F32, F32, F32, F32, F32, F32, F32, F32 },
      /* from   i8 */ LineT{  OK,   OK,  OK,  OK,  OK, I32,  OK, I16, I32, I32 },
      /* from  i16 */ LineT{ I32,  I32,  OK,  OK,  OK, I32, U16,  OK, I32, I32 },
      /* from  i32 */ LineT{  OK,   OK,  OK,  OK,  OK,  OK,  OK, U32,  OK, I64 },
      /* from  i64 */ LineT{ I32,  I32, I32, I32,  OK,  OK, I32, I32, U64,  OK },
      /* from   u8 */ LineT{  OK,  F32,  OK, U16,  OK, I32,  OK,  OK,  OK, U32 },
      /* from  u16 */ LineT{ U32,  U32, I16,  OK, U32, U32,  OK,  OK,  OK, U32 },
      /* from  u32 */ LineT{  OK,  F32, I32, I32,  OK, U64,  OK,  OK,  OK,  OK },
      /* from  u64 */ LineT{ U32,  U32, I64, I64, I64,  OK, U32, U32,  OK,  OK },
      // clang-format on
  };

#undef U64
#undef U32
#undef U16
#undef I64
#undef I32
#undef I16
#undef F32
#undef OK

  // ======== gaudi2 ========

  // cast
  // fr/to f32 bf16 i8 i16 i32 i64 u8 u16 u32 u64 f8
  // f32     *    X  X   X   X   -  X   X   X   -  X
  // bf16    X    *  X   X   X   -  X   X   X   -  X
  // i8      X    X  *   X   X   -  I   X   X   -  -
  // i16     X    X  -   *   X   -  -   I   X   -  -
  // i32     X    X  X   X   *   X  X   X   I   -  -
  // i64     -    -  -   -   X   *  -   -   -   I  -
  // u8      X    X  I   -   X   -  *   X   X   -  -
  // u16     X    X  X   I   X   -  X   *   X   -  -
  // u32     X    X  X   X   I   -  X   X   *   X  -
  // u64     -    -  -   -   -   I  -   -   X   *  -
  // f8      X    X  -   -   -   -  -   -   -   -  *

  // clang-format off
#define OK   CastStage {}
#define BF16 CastStage { CastType::bf16 }
#define F32  CastStage { CastType::f32  }
#define I32  CastStage { CastType::i32  }
#define I64  CastStage { CastType::i64  }
#define U16  CastStage { CastType::u16  }
#define U32  CastStage { CastType::u32  }
#define U64  CastStage { CastType::u64  }
  // clang-format on

  // TODO: SW-35847 Remove indirect casting
  using LineT = EnumMappingTable<CastType, CastStage>;
  static const EnumMappingTable<CastType, LineT> cast_stage_matrix_gaudi2 = {
      // clang-format off
      //              to:    f32  bf16    i8   i16  i32  i64    u8   u16  u32  u64    f8
      /* from  f32 */ LineT{  OK,   OK,   OK,   OK,  OK, I32,   OK,   OK,  OK, U32,   OK },
      /* from bf16 */ LineT{  OK,   OK,   OK,   OK,  OK, I32,   OK,   OK,  OK, U32,   OK },
      /* from   i8 */ LineT{  OK,   OK,   OK,   OK,  OK, I32,   OK,   OK,  OK, U32, BF16 },
      /* from  i16 */ LineT{  OK,   OK,  I32,   OK,  OK, I32,  U16,   OK,  OK, U32, BF16 },
      /* from  i32 */ LineT{  OK,   OK,   OK,   OK,  OK,  OK,   OK,   OK,  OK, I64,  F32 },
      /* from  i64 */ LineT{ I32,  I32,  I32,  I32,  OK,  OK,  I32,  I32, U64,  OK,  I32 },
      /* from   u8 */ LineT{  OK,   OK,   OK,  U16,  OK, I32,   OK,   OK,  OK, U32, BF16 },
      /* from  u16 */ LineT{  OK,   OK,   OK,   OK,  OK, I32,   OK,   OK,  OK, U32, BF16 },
      /* from  u32 */ LineT{  OK,   OK,   OK,   OK,  OK, U64,   OK,   OK,  OK,  OK,  F32 },
      /* from  u64 */ LineT{ U32,  U32,  U32,  U32, I64,  OK,  U32,  U32,  OK,  OK,  U32 },
      /* from   f8 */ LineT{  OK,   OK, BF16, BF16, F32, F32, BF16, BF16, F32, F32,   OK },
      // clang-format on
  };

#undef U64
#undef U32
#undef U16
#undef I64
#undef I32
#undef F32
#undef BF16
#undef OK

  auto cast_stage_matrix = syn_device_type == synDeviceType::synDeviceGaudi2
      ? cast_stage_matrix_gaudi2
      : cast_stage_matrix_gaudi;
  return cast_stage_matrix[cast_types.from_][cast_types.to_];
}

} // namespace

CastType DataTypeToCastType(const at::ScalarType& dt) {
  switch (dt) {
    case at::ScalarType::Float:
      return CastType::f32;
    case at::ScalarType::BFloat16:
      return CastType::bf16;
    case at::ScalarType::Char:
      return CastType::i8;
    case at::ScalarType::Bool:
      return CastType::i8;
    case at::ScalarType::Short:
      return CastType::i16;
    case at::ScalarType::Int:
      return CastType::i32;
    case at::ScalarType::Long:
      return CastType::i64;
    case at::ScalarType::Byte:
      return CastType::u8;
    default:
      HABANA_ASSERT(false, "Unknown data type");
      return CastType::u8;
  }
}

at::ScalarType CastTypeToDataType(CastType ct) {
  switch (ct) {
    case CastType::f32:
      return at::ScalarType::Float;
    case CastType::bf16:
      return at::ScalarType::BFloat16;
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
      HABANA_ASSERT(false, "Unknown data type");
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
  auto& device = synapse_helpers::HPURegistrar::get_device();
  return get_cast_sequence(cast_types, device.type());
}

} // namespace habana_helpers