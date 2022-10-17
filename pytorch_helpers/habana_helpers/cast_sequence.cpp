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

#include <vector>

#include "cast_sequence.h"
#include "enum_mapping_table.h"
#include "logging.h"
#include "pt_version_check.h"
#include "pytorch_helpers/habana_device/hpu_cached_devices.h"

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

  // clang-format off
#define OK  CastStage {}
#define N   CastStage {}
#define F32 CastStage { CastType::f32 }
#define I32 CastStage { CastType::i32 }
#define U32 CastStage { CastType::u32 }
#define U8  CastStage { CastType::u8  }
  // clang-format on

  // TODO: SW-35847 Remove indirect casting
  using LineT = EnumMappingTable<CastType, CastStage>;
  static const EnumMappingTable<CastType, LineT> cast_stage_matrix_gaudi = {
      // clang-format off
      //              to:    f32  bf16   i8  i16  i32  i64   u8  u16  u32  u64
      /* from  f32 */ LineT{   N,   OK,  OK, I32,  OK, I32, I32, U32,  OK, U32 },
      /* from bf16 */ LineT{  OK,    N, F32,   N, F32,   N,   N,   N, F32,   N },
      /* from   i8 */ LineT{  OK,   OK,   N,  OK,  OK, I32, I32,   N, I32,   N },
      /* from  i16 */ LineT{ I32,  I32,  OK,   N,  OK, I32, I32,   N, I32,   N },
      /* from  i32 */ LineT{  OK,   OK,  OK,  OK,   N,  OK,  OK, U32,  OK, U32 },
      /* from  i64 */ LineT{ I32,  I32, I32, I32,  OK,   N, I32,   N, I32,   N },
      /* from   u8 */ LineT{  OK,  F32, I32, I32,  OK, I32,   N,  OK,  OK, U32 },
      /* from  u16 */ LineT{ U32,    N,   N,   N, U32,   N,  OK,   N,  OK, U32 },
      /* from  u32 */ LineT{  OK,  F32, I32, I32,  OK, I32,  OK,  OK,   N,  OK },
      /* from  u64 */ LineT{ U32,    N,   N,   N, U32,   N, U32, U32,  OK,   N },
      // clang-format on
  };

#undef U8
#undef U32
#undef I32
#undef F32
#undef OK
#undef N

  // ======== gaudi2 ========

  // clang-format off
#define OK   CastStage {}
#define N    CastStage {}
#define BF16 CastStage { CastType::bf16 }
#define F32  CastStage { CastType::f32  }
#define I32  CastStage { CastType::i32  }
#define U16  CastStage { CastType::u16  }
#define U32  CastStage { CastType::u32  }
  // clang-format on

  // TODO: SW-35847 Remove indirect casting
  using LineT = EnumMappingTable<CastType, CastStage>;
  static const EnumMappingTable<CastType, LineT> cast_stage_matrix_gaudi2 = {
      // clang-format off
      //              to:    f32  bf16    i8   i16  i32  i64    u8   u16  u32  u64    f8   f16
      /* from  f32 */ LineT{   N,   OK,   OK,   OK,  OK, I32,   OK,   OK,  OK, U32,   OK,   OK },
      /* from bf16 */ LineT{  OK,    N,   OK,   OK,  OK, I32,   OK,   OK,  OK, U32,   OK,   OK },
      /* from   i8 */ LineT{  OK,   OK,    N,   OK,  OK, I32,   OK,   OK,  OK, U32, BF16,   OK },
      /* from  i16 */ LineT{  OK,   OK,   OK,    N,  OK, I32,  U16,   OK,  OK, U32, BF16,   OK },
      /* from  i32 */ LineT{  OK,   OK,   OK,   OK,   N,  OK,   OK,   OK,  OK, U32,  F32,   OK },
      /* from  i64 */ LineT{ I32,  I32,  I32,  I32,  OK,   N,  I32,  I32, I32,   N,    N,  I32 },
      /* from   u8 */ LineT{  OK,   OK,   OK,  U16,  OK, I32,    N,   OK,  OK, U32, BF16,   OK },
      /* from  u16 */ LineT{  OK,   OK,   OK,   OK,  OK, I32,   OK,    N,  OK, U32, BF16,   OK },
      /* from  u32 */ LineT{  OK,   OK,   OK,   OK,  OK, I32,   OK,   OK,   N,  OK,  F32,   OK },
      /* from  u64 */ LineT{ U32,  U32,  U32,  U32, U32,   N,  U32,  U32,  OK,   N,    N,  U32 },
      /* from   f8 */ LineT{  OK,   OK, BF16, BF16, F32,   N, BF16, BF16, F32,   N,    N, BF16 },
      /* from  f16 */ LineT{  OK,   OK,   OK,   OK,  OK, I32,   OK,   OK,  OK, U32, BF16,    N },
      // clang-format on
  };

#undef U32
#undef U16
#undef I32
#undef F32
#undef BF16
#undef OK
#undef N

  // ======== greco ========

  // clang-format off
#define OK  CastStage {}
#define N   CastStage {}
#define F32 CastStage { CastType::f32 }
#define I32 CastStage { CastType::i32 }
  // clang-format on

  // TODO: SW-35847 Remove indirect casting
  using LineT = EnumMappingTable<CastType, CastStage>;
  static const EnumMappingTable<CastType, LineT> cast_stage_matrix_greco = {
      // clang-format off
      //              to:    f32  bf16  f16   i8  i16  i32   u8  u16  u32
      /* from  f32 */ LineT{   N,   OK,  OK,  OK,  OK,  OK,  OK,  OK, I32 },
      /* from bf16 */ LineT{  OK,    N,  OK,  OK, F32,  OK,  OK, F32, I32 },
      /* from  f16 */ LineT{  OK,   OK,   N,  OK, I32,  OK,  OK, F32, I32 },
      /* from   i8 */ LineT{  OK,   OK,  OK,   N, I32,  OK,  OK, F32, I32 },
      /* from  i16 */ LineT{  OK,  F32, I32, I32,   N,  OK, I32,  OK, I32 },
      /* from  i32 */ LineT{  OK,   OK,  OK,  OK,  OK,   N,  OK, F32,  OK },
      /* from   u8 */ LineT{  OK,   OK,  OK,  OK, I32,  OK,   N, F32, I32 },
      /* from  u16 */ LineT{  OK,  F32, F32, F32,  OK, F32, F32,   N,   N },
      /* from  u32 */ LineT{ I32,  I32, I32, I32, I32,  OK, I32,   N,   N },
      // clang-format on
  };

#undef I32
#undef F32
#undef OK
#undef N

  EnumMappingTable<CastType, LineT> cast_stage_matrix;
  switch (syn_device_type) {
    case synDeviceType::synDeviceGaudi:
    case synDeviceType::synDeviceGaudiM:
      cast_stage_matrix = cast_stage_matrix_gaudi;
      break;
    case synDeviceType::synDeviceGaudi2:
      cast_stage_matrix = cast_stage_matrix_gaudi2;
      break;
    case synDeviceType::synDeviceGreco:
      cast_stage_matrix = cast_stage_matrix_greco;
      break;
    default:
      HABANA_ASSERT(false, "Unknown device: ", syn_device_type);
      break;
  }

  return cast_stage_matrix[cast_types.from_][cast_types.to_];
}

} // namespace

std::ostream& operator<<(std::ostream& os, const CastType& obj) {
  os << static_cast<std::underlying_type<CastType>::type>(obj);
  return os;
}

CastType DataTypeToCastType(const at::ScalarType& dt) {
  switch (dt) {
    case at::ScalarType::Float:
      return CastType::f32;
    case at::ScalarType::BFloat16:
      return CastType::bf16;
    case at::ScalarType::Half:
      return CastType::fp16;
#if IS_PYTORCH_FORK_AT_LEAST(1, 0)
    case at::ScalarType::Fp8r152:
      return CastType::f8;
#endif
    case at::ScalarType::Char:
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
#if IS_PYTORCH_FORK_AT_LEAST(1, 0)
    case CastType::f8:
      return at::ScalarType::Fp8r152;
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
  auto& device = synapse_helpers::HPURegistrar::get_device();
  return get_cast_sequence(cast_types, device.type());
}

} // namespace habana_helpers