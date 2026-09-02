/**
 * Copyright (c) 2026 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 */
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace habana::triton_gaudi {

inline constexpr std::uint32_t kLaunchParamsMagic = 0x31475452U;
inline constexpr std::uint16_t kLaunchAbiMajor = 1;
inline constexpr std::uint16_t kLaunchAbiMinor = 8;
inline constexpr std::uint16_t kBridgeLaunchAbiMajor = 2;
inline constexpr std::uint16_t kBridgeLaunchAbiMinor = 0;
inline constexpr std::size_t kArtifactHashChars = 64;
inline constexpr std::size_t kMaxIndexSpaceRank = 5;
inline constexpr std::size_t kMaxScalarParams = 32;
inline constexpr std::size_t kMaxLaunchTensors = 64;
inline constexpr char kKernelGuid[] = "triton_gaudi2_v1";
inline constexpr char kBridgeKernelGuid[] = "triton_gaudi2_v2";
inline constexpr char kLaunchPacketMagic[4] = {'G', 'A', 'L', '2'};
inline constexpr char kArtifactManifestMagic[8] = {
    'G', 'T', 'R', 'M', 'A', 'N', '0', '1'};
inline constexpr std::uint16_t kArtifactManifestAbiMajor = 1;
inline constexpr std::uint16_t kArtifactManifestAbiMinor = 0;

enum class KernelKind : std::uint32_t {
  Elementwise = 0,
  FusedAddRmsNorm = 1,
  SiluAndMul = 2,
  GdnDecodePacked = 3,
  GdnDecodeConvPacked = 4,
  GdnQkConvPacked = 5,
  GdnDecodeValueConvPacked = 6,
  DynamicQuant = 7,
};

struct LaunchParamsV1 {
  std::uint32_t magic{kLaunchParamsMagic};
  std::uint16_t abi_major{kLaunchAbiMajor};
  std::uint16_t abi_minor{kLaunchAbiMinor};
  std::uint16_t input_count{0};
  std::uint16_t output_count{0};
  std::uint16_t scalar_count{0};
  std::uint16_t index_space_rank{1};
  std::uint32_t block_size{0};
  std::uint32_t logical_size{0};
  std::uint32_t tensor_dtype{0};
  KernelKind kernel_kind{KernelKind::Elementwise};
  std::array<std::uint64_t, kMaxIndexSpaceRank> grid{};
  std::array<std::uint32_t, kMaxScalarParams> scalar_params{};
  std::array<char, kArtifactHashChars + 1> artifact_hash{};
};

enum class DTypeV2 : std::uint8_t {
  BFloat16 = 1,
  Float16 = 2,
  Float32 = 3,
  Float64 = 4,
  Int8 = 5,
  Int16 = 6,
  Int32 = 7,
  Int64 = 8,
  UInt8 = 9,
  UInt16 = 10,
  UInt32 = 11,
  UInt64 = 12,
  Float8E4NV = 13,
  Float8E5 = 14,
  Bool = 15,
};

enum class TensorRoleV2 : std::uint8_t {
  Input = 1,
  Output = 2,
  MutableInput = 3,
};

#pragma pack(push, 1)

struct ArtifactIndexMappingV1 {
  std::uint32_t index_space_dim;
  float a;
  float start_b;
  float end_b;
  std::uint32_t all_required;
};

struct ArtifactTensorAccessV1 {
  std::uint32_t flags;
  ArtifactIndexMappingV1 mappings[kMaxIndexSpaceRank];
};

struct ArtifactManifestV1 {
  char magic[8];
  std::uint16_t abi_major;
  std::uint16_t abi_minor;
  std::uint32_t struct_size;
  std::uint16_t index_space_rank;
  std::uint16_t input_count;
  std::uint16_t output_count;
  std::uint16_t reserved;
  ArtifactTensorAccessV1 tensor_access[kMaxLaunchTensors];
};

struct LaunchPacketHeaderV2 {
  char magic[4];
  std::uint16_t abi_major;
  std::uint16_t abi_minor;
  std::uint32_t struct_size;
  char artifact_hash[kArtifactHashChars];
  std::uint16_t grid_rank;
  std::uint16_t tensor_count;
  std::uint16_t scalar_count;
  std::uint8_t reserved[2];
  std::uint64_t grid[kMaxIndexSpaceRank];
};

struct TensorBindingV2 {
  std::uint16_t argument_index;
  DTypeV2 dtype;
  TensorRoleV2 role;
  std::uint32_t flags;
};

struct ScalarBindingV2 {
  std::uint16_t argument_index;
  DTypeV2 dtype;
  std::uint8_t reserved;
  std::uint64_t bits;
};

#pragma pack(pop)

static_assert(std::is_standard_layout_v<LaunchParamsV1>);
static_assert(std::is_trivially_copyable_v<LaunchParamsV1>);
static_assert(sizeof(LaunchParamsV1) == 272);
static_assert(sizeof(ArtifactIndexMappingV1) == 20);
static_assert(sizeof(ArtifactTensorAccessV1) == 104);
static_assert(sizeof(ArtifactManifestV1) == 6680);
static_assert(sizeof(LaunchPacketHeaderV2) == 124);
static_assert(sizeof(TensorBindingV2) == 8);
static_assert(sizeof(ScalarBindingV2) == 12);

} // namespace habana::triton_gaudi
