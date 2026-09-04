/**
 * Copyright (c) 2026 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 */

#include "backend/triton_gaudi_runtime.h"

#include <fcntl.h>
#include <openssl/evp.h>
#include <synapse_api.h>
#include <sys/stat.h>
#include <unistd.h>

#include <array>
#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <memory>
#include <mutex>
#include <nlohmann/json.hpp>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <unordered_map>
#include <utility>

#include "backend/habana_device/HPUDevice.h"
#include "backend/habana_device/HPUStream.h"
#include "backend/helpers/create_tensor.h"
#include "backend/synapse_helpers/device.h"
#include "backend/synapse_helpers/graph.h"
#include "habana_lazy/hpu_lazy_tensors.h"
#include "habana_lazy/lazy_executor.h"
#include "include/habanalabs/triton_gaudi_launch.h"

namespace habana::triton_gaudi {
namespace {

namespace fs = std::filesystem;

struct CompiledRecipe {
  std::shared_ptr<synapse_helpers::graph::recipe_handle> handle;
  std::uint64_t workspace_size{0};
  std::vector<std::string> tensor_names;
  std::vector<std::uint64_t> tensor_ids;
};

struct Artifact {
  std::string hash;
  int device_id{0};
  std::uint16_t input_count{0};
  std::uint16_t output_count{0};
  std::uint16_t scalar_count{0};
  std::uint16_t index_space_rank{1};
  std::uint32_t block_size{0};
  std::uint32_t logical_size{0};
  int bound_scalar_position{-1};
  KernelKind kernel_kind{KernelKind::Elementwise};
  at::ScalarType dtype{at::kFloat};
  std::vector<at::ScalarType> tensor_dtypes;
  std::vector<DTypeV2> tensor_dtype_codes;
  std::vector<std::uint16_t> tensor_argument_indices;
  std::vector<TensorRoleV2> tensor_roles;
  std::vector<std::uint16_t> scalar_argument_indices;
  std::vector<std::string> scalar_dtypes;
  ArtifactManifestV1 perf_manifest{};
  std::vector<std::size_t> mutable_input_positions;
  std::uint32_t ref_count{1};
  std::mutex recipe_mutex;
  std::unordered_map<std::string, std::shared_ptr<CompiledRecipe>> recipes;
};

struct Registry {
  std::mutex mutex;
  std::atomic<std::uint64_t> next_handle{1};
  std::unordered_map<std::uint64_t, std::shared_ptr<Artifact>> by_handle;
  std::unordered_map<std::string, std::uint64_t> by_hash;
};

Registry& registry() {
  // Synapse tears down the HPU device context before all shared-library and
  // Python module destructors have run.  Process-lifetime recipe ownership
  // avoids invoking recipe destructors against that partially destroyed
  // context; normal unregister calls still release entries during execution.
  static Registry* value = new Registry();
  return *value;
}

void require(bool condition, const std::string& message) {
  if (!condition) {
    throw std::runtime_error("Triton Gaudi runtime: " + message);
  }
}

bool is_lower_hex_hash(std::string_view hash) {
  if (hash.size() != kArtifactHashChars) {
    return false;
  }
  for (char value : hash) {
    if (!((value >= '0' && value <= '9') ||
          (value >= 'a' && value <= 'f'))) {
      return false;
    }
  }
  return true;
}

DTypeV2 dtype_code(const std::string& dtype) {
  if (dtype == "i1") {
    return DTypeV2::Bool;
  }
  if (dtype == "bf16") {
    return DTypeV2::BFloat16;
  }
  if (dtype == "f16") {
    return DTypeV2::Float16;
  }
  if (dtype == "f32") {
    return DTypeV2::Float32;
  }
  if (dtype == "f64") {
    return DTypeV2::Float64;
  }
  if (dtype == "i8") {
    return DTypeV2::Int8;
  }
  if (dtype == "i16") {
    return DTypeV2::Int16;
  }
  if (dtype == "i32") {
    return DTypeV2::Int32;
  }
  if (dtype == "i64") {
    return DTypeV2::Int64;
  }
  if (dtype == "u8") {
    return DTypeV2::UInt8;
  }
  if (dtype == "u16") {
    return DTypeV2::UInt16;
  }
  if (dtype == "u32") {
    return DTypeV2::UInt32;
  }
  if (dtype == "u64") {
    return DTypeV2::UInt64;
  }
  if (dtype == "fp8e4nv") {
    return DTypeV2::Float8E4NV;
  }
  if (dtype == "fp8e5") {
    return DTypeV2::Float8E5;
  }
  throw std::runtime_error(
      "Triton Gaudi runtime: unsupported launch dtype " + dtype);
}

TensorRoleV2 tensor_role(const std::string& role) {
  if (role == "input") {
    return TensorRoleV2::Input;
  }
  if (role == "output") {
    return TensorRoleV2::Output;
  }
  if (role == "mutable_input") {
    return TensorRoleV2::MutableInput;
  }
  throw std::runtime_error(
      "Triton Gaudi runtime: unsupported tensor role " + role);
}

fs::path artifact_directory() {
  const char* configured = std::getenv("TRITON_GAUDI_ARTIFACT_DIR");
  require(
      configured != nullptr && *configured != '\0',
      "TRITON_GAUDI_ARTIFACT_DIR must name a private cache directory");
  const fs::path directory(configured);
  std::error_code error;
  fs::create_directories(directory, error);
  require(!error, "cannot create artifact directory: " + error.message());
  require(
      fs::is_directory(directory, error) && !fs::is_symlink(directory, error),
      "artifact cache must be a real directory, not a symlink");
  fs::permissions(
      directory,
      fs::perms::owner_all,
      fs::perm_options::replace,
      error);
  require(!error, "cannot secure artifact directory: " + error.message());
  return directory;
}

void write_all(int fd, const std::vector<std::uint8_t>& data) {
  std::size_t written = 0;
  while (written < data.size()) {
    const ssize_t result =
        ::write(fd, data.data() + written, data.size() - written);
    if (result < 0 && errno == EINTR) {
      continue;
    }
    require(result > 0, "failed to write artifact cache entry");
    written += static_cast<std::size_t>(result);
  }
}

void materialize_artifact_file(
    const std::string& artifact_hash,
    const std::string& suffix,
    const std::vector<std::uint8_t>& data) {
  const fs::path directory = artifact_directory();
  const fs::path destination = directory / (artifact_hash + suffix);
  std::error_code error;
  const auto matches_expected = [&]() {
    if (!fs::is_regular_file(destination, error) ||
        fs::is_symlink(destination, error) ||
        fs::file_size(destination, error) != data.size()) {
      return false;
    }
    std::ifstream input(destination, std::ios::binary);
    std::vector<std::uint8_t> existing{
        std::istreambuf_iterator<char>(input),
        std::istreambuf_iterator<char>()};
    return existing == data;
  };
  if (fs::exists(destination, error)) {
    require(matches_expected(), "an incompatible artifact already exists in the cache");
    return;
  }

  const fs::path temporary = directory /
      (artifact_hash + suffix + ".tmp." + std::to_string(::getpid()) + "." +
       std::to_string(
           registry().next_handle.load(std::memory_order_relaxed)));
  const int fd = ::open(
      temporary.c_str(),
      O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC | O_NOFOLLOW,
      S_IRUSR | S_IWUSR);
  require(fd >= 0, "cannot create temporary artifact cache entry");
  bool fd_open = true;
  try {
    write_all(fd, data);
    require(::fsync(fd) == 0, "cannot flush temporary artifact cache entry");
    require(::close(fd) == 0, "cannot close temporary artifact cache entry");
    fd_open = false;
    if (::link(temporary.c_str(), destination.c_str()) != 0 &&
        errno != EEXIST) {
      throw std::runtime_error("Triton Gaudi runtime: cannot publish artifact");
    }
    fs::remove(temporary, error);
    require(matches_expected(), "published artifact does not match the requested ELF");
  } catch (...) {
    if (fd_open) {
      ::close(fd);
    }
    fs::remove(temporary, error);
    throw;
  }
}

void materialize_elf(
    const std::string& artifact_hash,
    const std::vector<std::uint8_t>& elf) {
  require(
      elf.size() >= 4 && elf[0] == 0x7f && elf[1] == 'E' &&
          elf[2] == 'L' && elf[3] == 'F',
      "artifact payload is not an ELF object");
  require(elf.size() <= 64ULL * 1024ULL * 1024ULL, "TPC ELF is too large");
  materialize_artifact_file(artifact_hash, ".elf", elf);
}

void materialize_manifest(
    const std::string& artifact_hash,
    const ArtifactManifestV1& manifest) {
  const auto* begin = reinterpret_cast<const std::uint8_t*>(&manifest);
  std::vector<std::uint8_t> bytes(begin, begin + sizeof(manifest));
  materialize_artifact_file(artifact_hash, ".manifest", bytes);
}

#pragma pack(push, 1)
struct ArtifactEnvelopeHeaderV2 {
  char magic[4];
  std::uint16_t major;
  std::uint16_t minor;
  std::uint32_t manifest_size;
  std::uint64_t payload_size;
};
#pragma pack(pop)

static_assert(sizeof(ArtifactEnvelopeHeaderV2) == 20);

std::string sha256_hex(
    const std::vector<std::pair<const void*, std::size_t>>& chunks) {
  using DigestContext =
      std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)>;
  DigestContext context(EVP_MD_CTX_new(), &EVP_MD_CTX_free);
  require(context != nullptr, "cannot allocate a SHA-256 context");
  require(
      EVP_DigestInit_ex(context.get(), EVP_sha256(), nullptr) == 1,
      "cannot initialize SHA-256");
  for (const auto& [data, size] : chunks) {
    require(
        EVP_DigestUpdate(context.get(), data, size) == 1,
        "cannot update SHA-256");
  }
  std::array<unsigned char, EVP_MAX_MD_SIZE> digest{};
  unsigned int digest_size = 0;
  require(
      EVP_DigestFinal_ex(context.get(), digest.data(), &digest_size) == 1 &&
          digest_size == 32,
      "cannot finalize SHA-256");
  constexpr char alphabet[] = "0123456789abcdef";
  std::string result(2 * digest_size, '0');
  for (std::size_t index = 0; index < digest_size; ++index) {
    result[2 * index] = alphabet[digest[index] >> 4];
    result[2 * index + 1] = alphabet[digest[index] & 0x0f];
  }
  return result;
}

std::string sha256_hex(const std::vector<std::uint8_t>& data) {
  return sha256_hex({{data.data(), data.size()}});
}

bool has_suffix(std::string_view value, std::string_view suffix) {
  return value.size() >= suffix.size() &&
      value.substr(value.size() - suffix.size()) == suffix;
}

void reject_artifact_paths(const nlohmann::json& value) {
  if (value.is_object()) {
    for (const auto& [key, child] : value.items()) {
      require(
          key != "path" && !has_suffix(key, "_path") &&
              !has_suffix(key, "_directory"),
          "Gaudi artifact manifests cannot contain filesystem paths");
      reject_artifact_paths(child);
    }
  } else if (value.is_array()) {
    for (const auto& child : value) {
      reject_artifact_paths(child);
    }
  }
}

std::array<std::uint8_t, 2> little_u16(std::uint16_t value) {
  return {
      static_cast<std::uint8_t>(value & 0xff),
      static_cast<std::uint8_t>((value >> 8) & 0xff)};
}

std::array<std::uint8_t, 8> little_u64(std::uint64_t value) {
  std::array<std::uint8_t, 8> result{};
  for (std::size_t index = 0; index < result.size(); ++index) {
    result[index] = static_cast<std::uint8_t>((value >> (8 * index)) & 0xff);
  }
  return result;
}

struct DecodedArtifactV2 {
  std::string hash;
  std::vector<std::uint8_t> elf;
  std::string legacy_manifest;
};

DecodedArtifactV2 decode_artifact_v2(
    const std::vector<std::uint8_t>& envelope) {
  constexpr std::size_t kMaxManifestBytes = 4ULL * 1024ULL * 1024ULL;
  constexpr std::size_t kMaxPayloadBytes = 256ULL * 1024ULL * 1024ULL;
  require(
      envelope.size() >= sizeof(ArtifactEnvelopeHeaderV2),
      "truncated GaudiKernelArtifactV2 header");
  ArtifactEnvelopeHeaderV2 header{};
  std::memcpy(&header, envelope.data(), sizeof(header));
  require(
      std::memcmp(header.magic, "GAK2", 4) == 0 && header.major == 2 &&
          header.minor == 0,
      "unsupported GaudiKernelArtifactV2 envelope");
  require(
      header.manifest_size <= kMaxManifestBytes &&
          header.payload_size <= kMaxPayloadBytes,
      "GaudiKernelArtifactV2 section exceeds the size limit");
  const std::uint64_t expected = sizeof(header) +
      static_cast<std::uint64_t>(header.manifest_size) +
      header.payload_size;
  require(
      expected == envelope.size(),
      "truncated or trailing data in GaudiKernelArtifactV2");

  const auto* manifest_begin = envelope.data() + sizeof(header);
  const std::string manifest_text(
      reinterpret_cast<const char*>(manifest_begin), header.manifest_size);
  auto manifest = nlohmann::json::parse(manifest_text);
  require(
      manifest.is_object() &&
          manifest.value("abi", "") == "GaudiKernelArtifactV2" &&
          manifest.value("abi_major", 0) == 2 &&
          manifest.value("abi_minor", 1) == 0,
      "invalid GaudiKernelArtifactV2 manifest ABI");
  require(
      manifest.value("target", "") == "gaudi2" &&
          manifest.value("mode", "") == "strict",
      "GaudiKernelArtifactV2 must be a strict Gaudi2 artifact");
  reject_artifact_paths(manifest);
  const std::string artifact_hash = manifest.value("artifact_hash", "");
  require(
      is_lower_hex_hash(artifact_hash),
      "GaudiKernelArtifactV2 has an invalid artifact hash");

  nlohmann::json descriptor;
  std::string payload_name;
  try {
    require(
        manifest.contains("payloads") && manifest.at("payloads").is_array(),
        "GaudiKernelArtifactV2 payload descriptors must be an array");
    const auto& payloads = manifest.at("payloads");
    require(
        payloads.size() == 1,
        "Bridge ABI v2 currently accepts one TPC payload only");
    descriptor = payloads.at(0);
    require(
        descriptor.is_object(),
        "GaudiKernelArtifactV2 payload descriptors must be objects");
    payload_name = descriptor.at("name").get<std::string>();
    require(
        !payload_name.empty() && payload_name.size() <= 0xffff &&
            descriptor.value("kind", "") == "tpc_elf" &&
            descriptor.value("offset", 1ULL) == 0 &&
            descriptor.value("size", 0ULL) == header.payload_size,
        "Bridge ABI v2 requires one contiguous TPC ELF payload");
  } catch (const nlohmann::json::exception& error) {
    throw std::runtime_error(
        "Triton Gaudi runtime: invalid Artifact V2 payload descriptor JSON: " +
        std::string(error.what()));
  }
  const auto* payload_begin = manifest_begin + header.manifest_size;
  std::vector<std::uint8_t> elf(
      payload_begin, payload_begin + header.payload_size);
  const std::string payload_digest = descriptor.value("sha256", "");
  require(
      is_lower_hex_hash(payload_digest) && payload_digest == sha256_hex(elf),
      "GaudiKernelArtifactV2 TPC payload digest mismatch");

  try {
    require(
        manifest.contains("execution_plan") &&
            manifest.at("execution_plan").is_object(),
        "GaudiKernelArtifactV2 execution_plan must be an object");
    const auto& plan = manifest.at("execution_plan");
    require(
        plan.contains("nodes") && plan.at("nodes").is_array(),
        "GaudiKernelArtifactV2 execution-plan nodes must be an array");
    const auto& nodes = plan.at("nodes");
    require(
        plan.value("version", 0) == 1 && nodes.size() == 1 &&
            nodes.at(0).is_object() &&
            nodes.at(0).value("engine", "") == "tpc" &&
            nodes.at(0).value("payload", "") == payload_name,
        "Bridge ABI v2 cannot drop multi-engine execution-plan nodes");
  } catch (const nlohmann::json::exception& error) {
    throw std::runtime_error(
        "Triton Gaudi runtime: invalid Artifact V2 execution-plan JSON: " +
        std::string(error.what()));
  }

  auto unhashed = manifest;
  unhashed.erase("artifact_hash");
  const std::string canonical_manifest = unhashed.dump(-1, ' ', true);
  constexpr char prefix[] = "GaudiKernelArtifactV2";
  const auto name_size = little_u16(
      static_cast<std::uint16_t>(payload_name.size()));
  const auto payload_size = little_u64(header.payload_size);
  const std::string computed_hash = sha256_hex(
      {{prefix, sizeof(prefix)},
       {canonical_manifest.data(), canonical_manifest.size()},
       {name_size.data(), name_size.size()},
       {payload_size.data(), payload_size.size()},
       {payload_name.data(), payload_name.size()},
       {elf.data(), elf.size()}});
  require(
      computed_hash == artifact_hash,
      "GaudiKernelArtifactV2 content digest mismatch");

  manifest.erase("abi");
  manifest.erase("abi_major");
  manifest.erase("abi_minor");
  manifest.erase("payloads");
  manifest.erase("execution_plan");
  manifest.erase("mode");
  manifest["abi"] = "GaudiKernelArtifactV1";
  manifest["abi_major"] = 1;
  manifest["abi_minor"] = 0;
  manifest["artifact_hash"] = artifact_hash;
  manifest["elf_sha256"] = payload_digest;
  return {artifact_hash, std::move(elf), manifest.dump()};
}

std::shared_ptr<Artifact> parse_artifact(
    const std::string& artifact_hash,
    const std::string& manifest_json,
    int device_id) {
  require(is_lower_hex_hash(artifact_hash), "artifact hash must be 64 lowercase hex characters");
  const auto manifest = nlohmann::json::parse(manifest_json);
  require(
      manifest.value("abi", "") == "GaudiKernelArtifactV1" &&
          manifest.value("abi_major", 0) == 1,
      "unsupported kernel artifact ABI");
  require(
      manifest.value("target", "") == "gaudi2" &&
          manifest.value("engine", "") == "tpc",
      "only Gaudi2 TPC artifacts can be registered");
  require(
      manifest.value("artifact_hash", "") == artifact_hash,
      "manifest and registration hashes differ");

  const auto& arguments = manifest.at("arguments");
  const auto& input_args = manifest.at("input_args");
  const auto output_args = manifest.contains("output_args")
      ? manifest.at("output_args").get<std::vector<std::size_t>>()
      : std::vector<std::size_t>{
            manifest.at("output_arg").get<std::size_t>()};
  std::vector<std::size_t> tensor_order;
  std::vector<std::size_t> scalar_order;
  std::vector<std::string> scalar_dtypes;
  std::size_t scalar_count = 0;
  std::size_t argument_position = 0;
  for (const auto& argument : arguments) {
    const auto index = argument.at("index").get<std::size_t>();
    require(
        index == argument_position++,
        "kernel argument indices must be contiguous and ordered");
    const auto kind = argument.at("kind").get<std::string>();
    const auto dtype = argument.at("dtype").get<std::string>();
    if (kind == "tensor") {
      require(
          dtype == "f32" || dtype == "bf16" || dtype == "i32" ||
              dtype == "fp8e4nv",
          "launch ABI supports FP32, BF16, I32, and Gaudi2 E4M3 tensors only");
      tensor_order.push_back(index);
    } else {
      require(
          dtype == "i32" || dtype == "u32" || dtype == "f32",
          "launch ABI supports i32/u32/f32 scalar parameters only");
      scalar_order.push_back(index);
      scalar_dtypes.push_back(dtype);
      ++scalar_count;
    }
  }
  std::vector<std::size_t> expected_order =
      input_args.get<std::vector<std::size_t>>();
  expected_order.insert(
      expected_order.end(), output_args.begin(), output_args.end());
  require(
      tensor_order == expected_order,
      "TPC tensor parameters must be ordered as inputs followed by output");
  require(
      scalar_count <= kMaxScalarParams,
      "artifact exceeds the scalar parameter ABI limit");
  require(
      input_args.size() <= std::numeric_limits<std::uint16_t>::max() &&
          output_args.size() <= std::numeric_limits<std::uint16_t>::max(),
      "artifact tensor count exceeds the launch ABI limit");

  const auto index_space = manifest.at("index_space");
  auto artifact = std::make_shared<Artifact>();
  artifact->hash = artifact_hash;
  artifact->device_id = device_id;
  artifact->input_count = static_cast<std::uint16_t>(input_args.size());
  artifact->output_count = static_cast<std::uint16_t>(output_args.size());
  artifact->scalar_count = static_cast<std::uint16_t>(scalar_count);
  artifact->tensor_argument_indices.reserve(tensor_order.size());
  for (const auto index : tensor_order) {
    require(
        index <= std::numeric_limits<std::uint16_t>::max(),
        "tensor argument index exceeds the launch ABI limit");
    artifact->tensor_argument_indices.push_back(
        static_cast<std::uint16_t>(index));
  }
  artifact->scalar_argument_indices.reserve(scalar_order.size());
  for (const auto index : scalar_order) {
    require(
        index <= std::numeric_limits<std::uint16_t>::max(),
        "scalar argument index exceeds the launch ABI limit");
    artifact->scalar_argument_indices.push_back(
        static_cast<std::uint16_t>(index));
  }
  artifact->scalar_dtypes = scalar_dtypes;
  try {
    const auto& access_patterns = manifest.at("access_patterns");
    require(
        access_patterns.is_array(),
        "artifact access_patterns must be an array");
    std::unordered_map<std::size_t, const nlohmann::json*> accesses;
    for (const auto& access : access_patterns) {
      require(access.is_object(), "tensor access descriptors must be objects");
      const auto index = access.at("arg").get<std::size_t>();
      const auto inserted = accesses.emplace(index, &access);
      require(inserted.second, "tensor access roles must be unique");
    }
    artifact->tensor_roles.reserve(tensor_order.size());
    for (std::size_t position = 0; position < tensor_order.size(); ++position) {
      const auto index = tensor_order[position];
      const auto access = accesses.find(index);
      require(
          access != accesses.end(),
          "every tensor argument requires an access role");
      const auto role = tensor_role(
          access->second->at("role").get<std::string>());
      require(
          position < artifact->input_count
              ? role == TensorRoleV2::Input ||
                  role == TensorRoleV2::MutableInput
              : role == TensorRoleV2::Output,
          "tensor access roles must agree with input/output argument lists");
      artifact->tensor_roles.push_back(role);
    }
    artifact->index_space_rank =
        static_cast<std::uint16_t>(index_space.value("rank", 1));
    require(
        artifact->index_space_rank > 0 &&
            artifact->index_space_rank <= kMaxIndexSpaceRank,
        "artifact index-space rank exceeds the perf-library ABI");

    std::memcpy(
        artifact->perf_manifest.magic,
        kArtifactManifestMagic,
        sizeof(artifact->perf_manifest.magic));
    artifact->perf_manifest.abi_major = kArtifactManifestAbiMajor;
    artifact->perf_manifest.abi_minor = kArtifactManifestAbiMinor;
    artifact->perf_manifest.struct_size = sizeof(ArtifactManifestV1);
    artifact->perf_manifest.index_space_rank = artifact->index_space_rank;
    artifact->perf_manifest.input_count = artifact->input_count;
    artifact->perf_manifest.output_count = artifact->output_count;
    for (auto& tensor_access : artifact->perf_manifest.tensor_access) {
      for (auto& mapping : tensor_access.mappings) {
        mapping.all_required = 1;
      }
    }
    const auto finite_float = [](const nlohmann::json& value) {
      require(value.is_number(), "tensor access coefficients must be numeric");
      const double number = value.get<double>();
      require(
          std::isfinite(number) &&
              std::abs(number) <= std::numeric_limits<float>::max(),
          "tensor access coefficient cannot be represented by the TPC ABI");
      return static_cast<float>(number);
    };
    for (std::size_t position = 0; position < tensor_order.size(); ++position) {
      auto& destination = artifact->perf_manifest.tensor_access[position];
      const auto& mappings =
          accesses.at(tensor_order[position])->at("mapping");
      require(
          mappings.is_array() && mappings.size() <= kMaxIndexSpaceRank,
          "tensor access mappings exceed the perf-library ABI");
      std::array<bool, kMaxIndexSpaceRank> mapped_dimensions{};
      for (const auto& mapping : mappings) {
        require(mapping.is_object(), "tensor access mappings must be objects");
        const auto tensor_dim =
            mapping.at("tensor_dim").get<std::size_t>();
        const auto index_dim =
            mapping.at("index_space_dim").get<std::size_t>();
        require(
            tensor_dim < kMaxIndexSpaceRank &&
                !mapped_dimensions[tensor_dim],
            "tensor access dimensions must be unique and in range");
        require(
            index_dim < artifact->index_space_rank,
            "tensor access mapping references an invalid index-space dimension");
        auto& output = destination.mappings[tensor_dim];
        output.index_space_dim = static_cast<std::uint32_t>(index_dim);
        output.a = finite_float(mapping.at("a"));
        output.start_b = finite_float(mapping.at("start_b"));
        output.end_b = finite_float(mapping.at("end_b"));
        output.all_required = 0;
        mapped_dimensions[tensor_dim] = true;
      }
    }
  } catch (const nlohmann::json::exception& error) {
    throw std::runtime_error(
        "Triton Gaudi runtime: invalid access_patterns/index_space JSON: " +
        std::string(error.what()));
  }
  artifact->block_size = index_space.at("block_size").get<std::uint32_t>();
  artifact->logical_size = artifact->block_size;
  if (!manifest.at("bound_arg").is_null()) {
    const auto bound_arg = manifest.at("bound_arg").get<std::size_t>();
    const auto bound = std::find(
        scalar_order.begin(), scalar_order.end(), bound_arg);
    require(
        bound != scalar_order.end(),
        "bound_arg must identify an i32/u32 scalar argument");
    artifact->bound_scalar_position =
        static_cast<int>(std::distance(scalar_order.begin(), bound));
    require(
        scalar_dtypes.at(
            static_cast<std::size_t>(artifact->bound_scalar_position)) !=
            "f32",
        "bound_arg must identify an i32/u32 scalar argument");
  }
  const auto kind = manifest.value("kind", "elementwise");
  if (kind == "fused_add_rms_norm") {
    artifact->kernel_kind = KernelKind::FusedAddRmsNorm;
    const auto& parameters = manifest.at("parameters");
    artifact->logical_size = parameters.at("n_cols").get<std::uint32_t>();
    require(
        input_args.get<std::vector<std::size_t>>() ==
                std::vector<std::size_t>({0, 1, 2}) &&
            output_args == std::vector<std::size_t>({3, 4}) &&
            scalar_order == std::vector<std::size_t>({5}) &&
            scalar_dtypes == std::vector<std::string>({"f32"}) &&
            artifact->bound_scalar_position < 0,
        "fused add+RMSNorm artifact has an incompatible tensor or scalar ABI");
  } else if (kind == "silu_and_mul_dynamic_quant") {
    artifact->kernel_kind = KernelKind::SiluAndMulDynamicQuant;
    const auto& parameters = manifest.at("parameters");
    artifact->logical_size = parameters.at("n_cols").get<std::uint32_t>();
    require(
        input_args.get<std::vector<std::size_t>>() ==
                std::vector<std::size_t>({0}) &&
            output_args == std::vector<std::size_t>({1, 2}) &&
            scalar_order.empty() && artifact->bound_scalar_position < 0 &&
            parameters.at("input_row_stride").get<std::uint32_t>() ==
                2 * artifact->logical_size &&
            parameters.at("fp8_max").get<double>() == 240.0 &&
            std::abs(parameters.at("scale_epsilon").get<double>() - 1.0e-8) <=
                1.0e-14,
        "fused SiLU-and-mul dynamic quantization artifact has an incompatible ABI");
  } else if (kind == "dynamic_quant") {
    artifact->kernel_kind = KernelKind::DynamicQuant;
    const auto& parameters = manifest.at("parameters");
    artifact->logical_size = parameters.at("n_cols").get<std::uint32_t>();
    require(
        input_args.get<std::vector<std::size_t>>() ==
                std::vector<std::size_t>({0}) &&
            output_args == std::vector<std::size_t>({1, 2}) &&
            scalar_order.empty() && artifact->bound_scalar_position < 0 &&
            parameters.at("fp8_max").get<double>() == 240.0 &&
            std::abs(parameters.at("scale_epsilon").get<double>() - 1.0e-8) <=
                1.0e-14,
        "dynamic FP8 quantization artifact has an incompatible tensor or scalar ABI");
  } else if (kind == "silu_and_mul") {
    artifact->kernel_kind = KernelKind::SiluAndMul;
    const auto& parameters = manifest.at("parameters");
    artifact->logical_size = parameters.at("n_cols").get<std::uint32_t>();
    require(
        input_args.get<std::vector<std::size_t>>() ==
                std::vector<std::size_t>({0}) &&
            output_args == std::vector<std::size_t>({1}) &&
            scalar_order.empty() && artifact->bound_scalar_position < 0 &&
            parameters.at("input_row_stride").get<std::uint32_t>() ==
                2 * artifact->logical_size &&
            parameters.at("chunk_size").get<std::uint32_t>() ==
                artifact->block_size &&
            parameters.at("chunks_per_row").get<std::uint32_t>() ==
                (artifact->logical_size + artifact->block_size - 1) /
                    artifact->block_size,
        "SiLU-and-mul artifact has an incompatible tensor or scalar ABI");
  } else if (kind == "gdn_decode_packed") {
    artifact->kernel_kind = KernelKind::GdnDecodePacked;
    const auto& parameters = manifest.at("parameters");
    artifact->logical_size = parameters.at("value_dim").get<std::uint32_t>();
    artifact->mutable_input_positions = {
        parameters.at("mutates_arg").get<std::size_t>()};
    require(
        input_args.get<std::vector<std::size_t>>() ==
                std::vector<std::size_t>({0, 1, 2, 3, 4, 5, 6}) &&
            output_args == std::vector<std::size_t>({7}) &&
            scalar_order == std::vector<std::size_t>({8}) &&
            scalar_dtypes == std::vector<std::string>({"i32"}) &&
            artifact->bound_scalar_position < 0 &&
            artifact->mutable_input_positions ==
                std::vector<std::size_t>({0}) &&
            parameters.at("key_heads").get<std::uint32_t>() == 16 &&
            parameters.at("value_heads").get<std::uint32_t>() == 48 &&
            parameters.at("key_dim").get<std::uint32_t>() == 128 &&
            parameters.at("value_dim").get<std::uint32_t>() == 128 &&
            parameters.at("packed_width").get<std::uint32_t>() == 10240 &&
            parameters.at("value_tile").get<std::uint32_t>() ==
                artifact->block_size &&
            parameters.at("state_slots_arg").get<std::size_t>() == 8,
        "packed GDN decode artifact has an incompatible tensor or scalar ABI");
  } else if (kind == "gdn_decode_conv_packed") {
    artifact->kernel_kind = KernelKind::GdnDecodeConvPacked;
    const auto& parameters = manifest.at("parameters");
    artifact->logical_size = parameters.at("value_dim").get<std::uint32_t>();
    artifact->mutable_input_positions =
        parameters.at("mutates_args").get<std::vector<std::size_t>>();
    require(
        input_args.get<std::vector<std::size_t>>() ==
                std::vector<std::size_t>({0, 1, 2, 3, 4, 5, 6, 7, 8}) &&
            output_args == std::vector<std::size_t>({9}) &&
            scalar_order == std::vector<std::size_t>({10, 11}) &&
            scalar_dtypes == std::vector<std::string>({"i32", "i32"}) &&
            artifact->bound_scalar_position < 0 &&
            artifact->mutable_input_positions ==
                std::vector<std::size_t>({0, 1}) &&
            parameters.at("key_heads").get<std::uint32_t>() == 16 &&
            parameters.at("value_heads").get<std::uint32_t>() == 48 &&
            parameters.at("key_dim").get<std::uint32_t>() == 128 &&
            parameters.at("value_dim").get<std::uint32_t>() == 128 &&
            parameters.at("packed_width").get<std::uint32_t>() == 10240 &&
            parameters.at("conv_width").get<std::uint32_t>() == 4 &&
            parameters.at("conv_slots_arg").get<std::size_t>() == 10 &&
            parameters.at("state_slots_arg").get<std::size_t>() == 11,
        "fused conv+GDN artifact has an incompatible tensor or scalar ABI");
  } else if (kind == "gdn_qk_conv_packed") {
    artifact->kernel_kind = KernelKind::GdnQkConvPacked;
    const auto& parameters = manifest.at("parameters");
    artifact->logical_size = parameters.at("channels").get<std::uint32_t>();
    artifact->mutable_input_positions =
        parameters.at("mutates_args").get<std::vector<std::size_t>>();
    require(
        input_args.get<std::vector<std::size_t>>() ==
                std::vector<std::size_t>({0, 1, 2, 3}) &&
            output_args == std::vector<std::size_t>({4}) &&
            scalar_order == std::vector<std::size_t>({5}) &&
            scalar_dtypes == std::vector<std::string>({"i32"}) &&
            artifact->bound_scalar_position < 0 &&
            artifact->mutable_input_positions ==
                std::vector<std::size_t>({0}) &&
            artifact->logical_size == 4096 &&
            parameters.at("packed_width").get<std::uint32_t>() == 10240 &&
            parameters.at("conv_width").get<std::uint32_t>() == 4 &&
            parameters.at("conv_slots_arg").get<std::size_t>() == 5,
        "packed Q/K convolution artifact has an incompatible ABI");
  } else if (kind == "gdn_decode_value_conv_packed") {
    artifact->kernel_kind = KernelKind::GdnDecodeValueConvPacked;
    const auto& parameters = manifest.at("parameters");
    artifact->logical_size = parameters.at("value_dim").get<std::uint32_t>();
    artifact->mutable_input_positions =
        parameters.at("mutates_args").get<std::vector<std::size_t>>();
    require(
        input_args.get<std::vector<std::size_t>>() ==
                std::vector<std::size_t>({0, 1, 2, 3, 4, 5, 6, 7, 8, 9}) &&
            output_args == std::vector<std::size_t>({10}) &&
            scalar_order == std::vector<std::size_t>({11, 12}) &&
            scalar_dtypes == std::vector<std::string>({"i32", "i32"}) &&
            artifact->bound_scalar_position < 0 &&
            artifact->mutable_input_positions ==
                std::vector<std::size_t>({0, 1}) &&
            parameters.at("key_heads").get<std::uint32_t>() == 16 &&
            parameters.at("value_heads").get<std::uint32_t>() == 48 &&
            parameters.at("key_dim").get<std::uint32_t>() == 128 &&
            parameters.at("value_dim").get<std::uint32_t>() == 128 &&
            parameters.at("packed_width").get<std::uint32_t>() == 10240 &&
            parameters.at("qk_width").get<std::uint32_t>() == 4096 &&
            parameters.at("conv_width").get<std::uint32_t>() == 4 &&
            parameters.at("value_tile").get<std::uint32_t>() ==
                artifact->block_size &&
            parameters.at("conv_slots_arg").get<std::size_t>() == 11 &&
            parameters.at("state_slots_arg").get<std::size_t>() == 12,
        "fused value-conv + GDN artifact has an incompatible ABI");
  } else {
    require(kind == "elementwise", "manifest has an unsupported TPC kernel kind");
  }
  const auto tensor_dtype = manifest.value("tensor_dtype", "");
  artifact->dtype = tensor_dtype == "bf16" ? at::kBFloat16
      : tensor_dtype == "fp8e4nv"          ? at::kFloat8_e4m3fn
                                           : at::kFloat;
  require(
      tensor_dtype == "f32" || tensor_dtype == "bf16" ||
          tensor_dtype == "fp8e4nv",
      "manifest has an unsupported tensor dtype");
  artifact->tensor_dtypes.reserve(tensor_order.size());
  artifact->tensor_dtype_codes.reserve(tensor_order.size());
  for (const auto& argument : arguments) {
    if (argument.at("kind").get<std::string>() == "tensor") {
      const auto argument_dtype = argument.at("dtype").get<std::string>();
      artifact->tensor_dtype_codes.push_back(dtype_code(argument_dtype));
      artifact->tensor_dtypes.push_back(
          argument_dtype == "bf16"          ? at::kBFloat16
              : argument_dtype == "i32"     ? at::kInt
              : argument_dtype == "fp8e4nv" ? at::kFloat8_e4m3fn
                                            : at::kFloat);
      if (artifact->kernel_kind != KernelKind::GdnDecodePacked &&
          artifact->kernel_kind != KernelKind::GdnDecodeConvPacked &&
          artifact->kernel_kind != KernelKind::GdnQkConvPacked &&
          artifact->kernel_kind != KernelKind::GdnDecodeValueConvPacked &&
          artifact->kernel_kind != KernelKind::DynamicQuant &&
          artifact->kernel_kind != KernelKind::SiluAndMulDynamicQuant) {
        require(
            argument_dtype == tensor_dtype,
            "all generic TPC tensor arguments must use the manifest dtype");
      }
    }
  }
  if (artifact->kernel_kind == KernelKind::FusedAddRmsNorm) {
    require(
        artifact->dtype == at::kBFloat16 &&
            artifact->logical_size > 0 &&
            artifact->logical_size <= artifact->block_size &&
            (artifact->logical_size == 1 ||
             artifact->logical_size > artifact->block_size / 2) &&
            artifact->block_size <= 8192 &&
            (artifact->block_size & (artifact->block_size - 1)) == 0,
        "fused add+RMSNorm requires BF16 and a supported power-of-two block size");
  } else if (
      artifact->kernel_kind == KernelKind::SiluAndMulDynamicQuant) {
    const std::vector<at::ScalarType> expected_dtypes{
        at::kBFloat16, at::kFloat8_e4m3fn, at::kFloat};
    require(
        artifact->dtype == at::kFloat8_e4m3fn &&
            artifact->tensor_dtypes == expected_dtypes &&
            artifact->logical_size > 0 && artifact->logical_size <= 4096 &&
            artifact->logical_size <= artifact->block_size &&
            (artifact->logical_size == 1 ||
             artifact->logical_size > artifact->block_size / 2) &&
            artifact->block_size <= 8192 &&
            (artifact->block_size & (artifact->block_size - 1)) == 0,
        "fused SiLU-and-mul dynamic quantization requires BF16 to E4M3/f32 and a supported block size");
  } else if (artifact->kernel_kind == KernelKind::DynamicQuant) {
    const std::vector<at::ScalarType> expected_dtypes{
        at::kBFloat16, at::kFloat8_e4m3fn, at::kFloat};
    require(
        artifact->dtype == at::kFloat8_e4m3fn &&
            artifact->tensor_dtypes == expected_dtypes &&
            artifact->logical_size > 0 &&
            artifact->logical_size <= artifact->block_size &&
            (artifact->logical_size == 1 ||
             artifact->logical_size > artifact->block_size / 2) &&
            artifact->block_size <= 16384 &&
            (artifact->block_size & (artifact->block_size - 1)) == 0,
        "dynamic quantization requires BF16 to E4M3/f32 and a supported block size");
  } else if (artifact->kernel_kind == KernelKind::SiluAndMul) {
    require(
        artifact->dtype == at::kBFloat16 &&
            artifact->logical_size > 0 &&
            artifact->logical_size <= 65536 &&
            artifact->block_size >= 128 &&
            artifact->block_size <= 1024 &&
            (artifact->block_size & (artifact->block_size - 1)) == 0,
        "SiLU-and-mul requires BF16 and a supported power-of-two block size");
  } else if (artifact->kernel_kind == KernelKind::GdnDecodePacked) {
    const std::vector<at::ScalarType> expected_dtypes{
        at::kFloat,
        at::kBFloat16,
        at::kBFloat16,
        at::kBFloat16,
        at::kFloat,
        at::kFloat,
        at::kInt,
        at::kBFloat16};
    require(
        artifact->dtype == at::kBFloat16 &&
            artifact->tensor_dtypes == expected_dtypes &&
            artifact->logical_size == 128 &&
            (artifact->block_size == 16 || artifact->block_size == 32 ||
             artifact->block_size == 64 || artifact->block_size == 128),
        "packed GDN decode requires the canonical mixed dtypes and value tile");
  } else if (artifact->kernel_kind == KernelKind::GdnDecodeConvPacked) {
    const std::vector<at::ScalarType> expected_dtypes{
        at::kBFloat16,
        at::kFloat,
        at::kBFloat16,
        at::kBFloat16,
        at::kBFloat16,
        at::kFloat,
        at::kFloat,
        at::kInt,
        at::kBFloat16,
        at::kBFloat16};
    require(
        artifact->dtype == at::kBFloat16 &&
            artifact->tensor_dtypes == expected_dtypes &&
            artifact->logical_size == 128 && artifact->block_size == 128,
        "fused conv+GDN requires the canonical mixed dtypes and ownership tile");
  } else if (artifact->kernel_kind == KernelKind::GdnQkConvPacked) {
    const std::vector<at::ScalarType> expected_dtypes{
        at::kBFloat16,
        at::kBFloat16,
        at::kInt,
        at::kBFloat16,
        at::kBFloat16};
    require(
        artifact->dtype == at::kBFloat16 &&
            artifact->tensor_dtypes == expected_dtypes &&
            artifact->logical_size == 4096 &&
            (artifact->block_size == 128 || artifact->block_size == 256 ||
             artifact->block_size == 512),
        "packed Q/K convolution requires canonical dtypes and channel tile");
  } else if (artifact->kernel_kind == KernelKind::GdnDecodeValueConvPacked) {
    const std::vector<at::ScalarType> expected_dtypes{
        at::kBFloat16,
        at::kFloat,
        at::kBFloat16,
        at::kBFloat16,
        at::kBFloat16,
        at::kBFloat16,
        at::kFloat,
        at::kFloat,
        at::kInt,
        at::kBFloat16,
        at::kBFloat16};
    require(
        artifact->dtype == at::kBFloat16 &&
            artifact->tensor_dtypes == expected_dtypes &&
            artifact->logical_size == 128 &&
            (artifact->block_size == 16 || artifact->block_size == 32 ||
             artifact->block_size == 64 || artifact->block_size == 128),
        "fused value-conv + GDN requires canonical dtypes and value tile");
  }
  require(
      artifact->input_count > 0 && artifact->output_count > 0 &&
          artifact->block_size > 0 &&
          (artifact->kernel_kind == KernelKind::SiluAndMul
               ? artifact->index_space_rank == 2
               : artifact->kernel_kind == KernelKind::GdnDecodePacked
               ? artifact->index_space_rank == 3
               : artifact->kernel_kind == KernelKind::GdnDecodeConvPacked
               ? artifact->index_space_rank == 2
               : artifact->kernel_kind == KernelKind::GdnQkConvPacked
               ? artifact->index_space_rank == 2
               : artifact->kernel_kind == KernelKind::GdnDecodeValueConvPacked
               ? artifact->index_space_rank == 3
               : artifact->index_space_rank == 1),
      "invalid initial TPC index-space metadata");
  return artifact;
}

std::shared_ptr<Artifact> lookup(std::uint64_t handle) {
  auto& state = registry();
  std::lock_guard<std::mutex> guard(state.mutex);
  const auto iterator = state.by_handle.find(handle);
  require(iterator != state.by_handle.end(), "unknown or released artifact handle");
  return iterator->second;
}

void validate_tensor(
    const at::Tensor& tensor,
    at::ScalarType dtype,
    int device_id) {
  require(tensor.defined(), "kernel tensor argument is undefined");
  require(tensor.device().type() == at::kHPU, "kernel tensors must reside on HPU");
  require(
      tensor.device().index() == device_id,
      "kernel tensors must reside on the artifact's HPU device");
  require(
      tensor.scalar_type() == dtype,
      "kernel tensor dtype does not match the compiled artifact");
  require(tensor.is_contiguous(), "initial launch ABI requires contiguous tensors");
  require(tensor.numel() > 0, "zero-sized tensors are not supported by the initial launch ABI");
  require(
      tensor.numel() <= std::numeric_limits<unsigned>::max(),
      "tensor is too large for the Synapse tensor descriptor");
}

LaunchParamsV1 make_launch_params(
    const Artifact& artifact,
    const std::vector<std::uint64_t>& grid,
    const std::vector<std::uint32_t>& scalar_params) {
  LaunchParamsV1 params{};
  params.input_count = artifact.input_count;
  params.output_count = artifact.output_count;
  params.scalar_count = artifact.scalar_count;
  params.index_space_rank = artifact.index_space_rank;
  params.block_size = artifact.block_size;
  params.logical_size = artifact.logical_size;
  params.tensor_dtype = artifact.dtype == at::kBFloat16
      ? 1U << 8 // tpc_lib_api::DATA_BF16
      : artifact.dtype == at::kFloat8_e4m3fn
      ? 1U << 5 // tpc_lib_api::DATA_F8_143
      : 1U << 12; // tpc_lib_api::DATA_F32
  params.kernel_kind = artifact.kernel_kind;
  std::copy_n(
      grid.begin(), artifact.index_space_rank, params.grid.begin());
  std::copy(
      scalar_params.begin(), scalar_params.end(), params.scalar_params.begin());
  std::copy(
      artifact.hash.begin(), artifact.hash.end(), params.artifact_hash.begin());
  return params;
}

std::string recipe_key(
    const std::vector<std::uint64_t>& grid,
    const std::vector<at::Tensor>& tensors,
    const std::vector<std::uint32_t>& scalar_params) {
  std::ostringstream key;
  key << "g";
  for (const auto value : grid) {
    key << ':' << value;
  }
  key << "|n";
  for (const auto& tensor : tensors) {
    key << ':' << tensor.scalar_type() << '[';
    for (const auto size : tensor.sizes()) {
      key << size << ',';
    }
    key << ']';
  }
  key << "|s";
  for (const auto value : scalar_params) {
    key << ':' << value;
  }
  return key.str();
}

std::shared_ptr<CompiledRecipe> compile_recipe(
    const std::shared_ptr<Artifact>& artifact,
    const std::vector<std::uint64_t>& grid,
    const std::vector<at::Tensor>& tensors,
    const std::vector<std::uint32_t>& scalar_params) {
  auto& device = HPUDeviceContext::get_device();
  auto graph = synapse_helpers::graph::create(
      device, "triton_gaudi_" + artifact->hash.substr(0, 16));

  std::vector<synapse_helpers::tensor> tensor_owners;
  std::vector<synTensor> inputs;
  std::vector<synTensor> outputs;
  tensor_owners.reserve(tensors.size());
  inputs.reserve(artifact->input_count);
  outputs.reserve(artifact->output_count);

  auto compiled = std::make_shared<CompiledRecipe>();
  compiled->tensor_names.reserve(tensors.size());
  for (std::size_t index = 0; index < tensors.size(); ++index) {
    std::vector<std::int64_t> shape;
    std::vector<std::int64_t> stride;
    if (artifact->kernel_kind == KernelKind::SiluAndMul ||
        artifact->kernel_kind == KernelKind::DynamicQuant ||
        artifact->kernel_kind == KernelKind::SiluAndMulDynamicQuant ||
        artifact->kernel_kind == KernelKind::GdnDecodePacked ||
        artifact->kernel_kind == KernelKind::GdnDecodeConvPacked ||
        artifact->kernel_kind == KernelKind::GdnQkConvPacked ||
        artifact->kernel_kind == KernelKind::GdnDecodeValueConvPacked) {
      shape = tensors[index].sizes().vec();
      stride = tensors[index].strides().vec();
    } else {
      shape = {tensors[index].numel()};
      stride = {1};
    }
    auto syn_tensor = habana_helpers::create_tensor(
        c10::IntArrayRef(shape),
        c10::IntArrayRef(stride),
        graph,
        true,
        false,
        static_cast<int>(device.id()),
        artifact->tensor_dtypes.at(index),
        "triton_arg_" + std::to_string(index));
    compiled->tensor_names.push_back(syn_tensor.name());
    if (index < artifact->input_count) {
      inputs.push_back(syn_tensor.get());
    } else {
      outputs.push_back(syn_tensor.get());
    }
    tensor_owners.push_back(std::move(syn_tensor));
  }

  auto params = make_launch_params(*artifact, grid, scalar_params);
  graph.add_node(
      std::move(inputs),
      std::move(outputs),
      &params,
      sizeof(params),
      kKernelGuid,
      nullptr,
      nullptr,
      nullptr,
      false);
  compiled->handle = graph.compile();
  require(compiled->handle != nullptr, "Synapse produced an empty TPC recipe");
  compiled->workspace_size =
      synapse_helpers::graph::query_workspace_size(*compiled->handle);

  std::vector<const char*> names;
  names.reserve(compiled->tensor_names.size());
  for (const auto& name : compiled->tensor_names) {
    names.push_back(name.c_str());
  }
  compiled->tensor_ids.resize(names.size());
  const synStatus status = synTensorRetrieveIds(
      compiled->handle->syn_recipe_handle_,
      names.data(),
      compiled->tensor_ids.data(),
      static_cast<std::uint32_t>(names.size()));
  require(
      status == synSuccess,
      "synTensorRetrieveIds failed with status " + std::to_string(status));
  return compiled;
}

std::shared_ptr<CompiledRecipe> get_or_compile_recipe(
    const std::shared_ptr<Artifact>& artifact,
    const std::vector<std::uint64_t>& grid,
    const std::vector<at::Tensor>& tensors,
    const std::vector<std::uint32_t>& scalar_params) {
  const auto key = recipe_key(grid, tensors, scalar_params);
  std::lock_guard<std::mutex> guard(artifact->recipe_mutex);
  if (const auto iterator = artifact->recipes.find(key);
      iterator != artifact->recipes.end()) {
    return iterator->second;
  }
  auto compiled = compile_recipe(artifact, grid, tensors, scalar_params);
  artifact->recipes.emplace(key, compiled);
  return compiled;
}

struct LaunchResources {
  std::vector<at::Tensor> tensors;
  std::shared_ptr<Artifact> artifact;
  std::shared_ptr<CompiledRecipe> recipe;
  std::unique_ptr<synapse_helpers::device_ptr_lock> address_lock;
};

} // namespace

std::uint64_t register_artifact(
    const std::string& artifact_hash,
    const std::vector<std::uint8_t>& elf,
    const std::string& manifest_json,
    int device_id) {
  require(
      device_id >= 0 &&
          device_id ==
              static_cast<int>(HPUDeviceContext::get_device().id()),
      "artifact device does not match the active HPU");
  auto artifact = parse_artifact(artifact_hash, manifest_json, device_id);
  materialize_elf(artifact_hash, elf);
  materialize_manifest(artifact_hash, artifact->perf_manifest);

  auto& state = registry();
  std::lock_guard<std::mutex> guard(state.mutex);
  const std::string artifact_key =
      artifact_hash + ":" + std::to_string(device_id);
  if (const auto existing = state.by_hash.find(artifact_key);
      existing != state.by_hash.end()) {
    auto registered = state.by_handle.at(existing->second);
    ++registered->ref_count;
    return existing->second;
  }
  const std::uint64_t handle =
      state.next_handle.fetch_add(1, std::memory_order_relaxed);
  state.by_handle.emplace(handle, std::move(artifact));
  state.by_hash.emplace(artifact_key, handle);
  return handle;
}

std::uint64_t register_artifact_v2(
    const std::vector<std::uint8_t>& envelope,
    int device_id) {
  DecodedArtifactV2 decoded;
  try {
    decoded = decode_artifact_v2(envelope);
  } catch (const nlohmann::json::exception& error) {
    throw std::runtime_error(
        "Triton Gaudi runtime: invalid Artifact V2 envelope JSON: " +
        std::string(error.what()));
  }
  try {
    return register_artifact(
        decoded.hash,
        decoded.elf,
        decoded.legacy_manifest,
        device_id);
  } catch (const nlohmann::json::exception& error) {
    throw std::runtime_error(
        "Triton Gaudi runtime: invalid projected launch manifest JSON: " +
        std::string(error.what()));
  }
}

void unregister_artifact(std::uint64_t handle) {
  auto& state = registry();
  std::lock_guard<std::mutex> guard(state.mutex);
  const auto iterator = state.by_handle.find(handle);
  if (iterator == state.by_handle.end()) {
    return;
  }
  if (--iterator->second->ref_count != 0) {
    return;
  }
  state.by_hash.erase(
      iterator->second->hash + ":" +
      std::to_string(iterator->second->device_id));
  state.by_handle.erase(iterator);
}

void launch(
    std::uint64_t handle,
    const std::vector<std::uint64_t>& grid,
    std::uint64_t hpu_stream,
    const std::vector<at::Tensor>& tensors,
    const std::vector<std::uint32_t>& scalar_params) {
  const auto artifact = lookup(handle);
  require(grid.size() == 3, "launch grid must have three dimensions");
  if (artifact->kernel_kind == KernelKind::SiluAndMul) {
    require(
        grid[0] > 0 && grid[1] > 0 && grid[2] == 1,
        "SiLU-and-mul requires a non-empty two-dimensional grid");
  } else if (artifact->kernel_kind == KernelKind::GdnDecodePacked) {
    require(
        grid[0] > 0 && grid[1] == 48 && grid[2] > 0,
        "packed GDN decode requires a non-empty three-dimensional grid");
  } else if (artifact->kernel_kind == KernelKind::GdnDecodeConvPacked) {
    require(
        grid[0] == 16 && grid[1] > 0 && grid[2] == 1,
        "fused conv+GDN requires key-head by batch index space");
  } else if (artifact->kernel_kind == KernelKind::GdnQkConvPacked) {
    require(
        grid[0] == 4096 / artifact->block_size && grid[1] > 0 &&
            grid[2] == 1,
        "packed Q/K convolution requires channel-block by batch index space");
  } else if (
      artifact->kernel_kind == KernelKind::GdnDecodeValueConvPacked) {
    require(
        grid[0] > 0 && grid[1] == 48 && grid[2] > 0,
        "fused value-conv + GDN requires tile by head by batch index space");
  } else {
    require(
        grid[0] > 0 && grid[1] == 1 && grid[2] == 1,
        "initial TPC launch supports a non-empty one-dimensional grid only");
  }
  require(
      tensors.size() == artifact->input_count + artifact->output_count,
      "tensor argument count does not match the artifact");
  require(
      scalar_params.size() == artifact->scalar_count,
      "scalar argument count does not match the artifact");
  require(
      grid[0] <= std::numeric_limits<std::uint32_t>::max() &&
          grid[1] <= std::numeric_limits<std::uint32_t>::max() &&
          grid[2] <= std::numeric_limits<std::uint32_t>::max(),
      "TPC index-space geometry exceeds the Gaudi2 ABI");
  require(
      artifact->device_id ==
          static_cast<int>(HPUDeviceContext::get_device().id()),
      "artifact device does not match the active HPU");
  if (artifact->kernel_kind == KernelKind::FusedAddRmsNorm) {
    const auto n_cols = artifact->logical_size;
    const auto matrix_elements = grid[0] * static_cast<std::uint64_t>(n_cols);
    require(
        tensors.size() == 5 &&
            static_cast<std::uint64_t>(tensors[0].numel()) == matrix_elements &&
            static_cast<std::uint64_t>(tensors[1].numel()) == matrix_elements &&
            static_cast<std::uint64_t>(tensors[2].numel()) == n_cols &&
            static_cast<std::uint64_t>(tensors[3].numel()) == matrix_elements &&
            static_cast<std::uint64_t>(tensors[4].numel()) == matrix_elements,
        "fused add+RMSNorm tensor storage does not match grid rows and n_cols");
  } else if (
      artifact->kernel_kind == KernelKind::SiluAndMulDynamicQuant) {
    const auto n_cols = artifact->logical_size;
    const auto rows = static_cast<std::int64_t>(grid[0]);
    const auto matrix_elements = grid[0] * static_cast<std::uint64_t>(n_cols);
    require(
        tensors.size() == 3 && tensors[0].dim() == 2 &&
            tensors[1].dim() == 2 && tensors[2].dim() == 2 &&
            tensors[0].sizes() == at::IntArrayRef({rows, 2 * n_cols}) &&
            tensors[1].sizes() == at::IntArrayRef({rows, n_cols}) &&
            tensors[2].sizes() == at::IntArrayRef({rows, 1}) &&
            static_cast<std::uint64_t>(tensors[0].numel()) ==
                2 * matrix_elements &&
            static_cast<std::uint64_t>(tensors[1].numel()) == matrix_elements &&
            static_cast<std::uint64_t>(tensors[2].numel()) == grid[0],
        "fused SiLU-and-mul dynamic quantization tensor storage does not match grid rows and n_cols");
  } else if (artifact->kernel_kind == KernelKind::DynamicQuant) {
    const auto n_cols = artifact->logical_size;
    const auto rows = static_cast<std::int64_t>(grid[0]);
    const auto matrix_elements = grid[0] * static_cast<std::uint64_t>(n_cols);
    require(
        tensors.size() == 3 && tensors[0].dim() == 2 &&
            tensors[1].dim() == 2 && tensors[2].dim() == 2 &&
            tensors[0].sizes() == at::IntArrayRef({rows, n_cols}) &&
            tensors[1].sizes() == at::IntArrayRef({rows, n_cols}) &&
            tensors[2].sizes() == at::IntArrayRef({rows, 1}) &&
            static_cast<std::uint64_t>(tensors[0].numel()) == matrix_elements &&
            static_cast<std::uint64_t>(tensors[1].numel()) == matrix_elements &&
            static_cast<std::uint64_t>(tensors[2].numel()) == grid[0],
        "dynamic quantization tensor storage does not match grid rows and n_cols");
  } else if (artifact->kernel_kind == KernelKind::SiluAndMul) {
    const auto n_cols = artifact->logical_size;
    const auto rows = grid[1];
    const auto rows_i64 = static_cast<std::int64_t>(rows);
    const auto matrix_elements = rows * static_cast<std::uint64_t>(n_cols);
    require(
        grid[0] == (n_cols + artifact->block_size - 1) / artifact->block_size &&
            tensors.size() == 2 && tensors[0].dim() == 2 &&
            tensors[1].dim() == 2 && tensors[0].size(0) == rows_i64 &&
            tensors[0].size(1) == 2 * n_cols &&
            tensors[1].size(0) == rows_i64 && tensors[1].size(1) == n_cols &&
            static_cast<std::uint64_t>(tensors[0].numel()) == 2 * matrix_elements &&
            static_cast<std::uint64_t>(tensors[1].numel()) == matrix_elements,
        "SiLU-and-mul tensor storage does not match grid rows and n_cols");
  } else if (artifact->kernel_kind == KernelKind::GdnDecodePacked) {
    const auto batch = static_cast<std::int64_t>(grid[2]);
    const auto state_slots = static_cast<std::int64_t>(scalar_params.at(0));
    require(
        grid[0] == 128 / artifact->block_size &&
            tensors.size() == 8 && state_slots > 0 &&
            tensors[0].sizes() ==
                at::IntArrayRef({state_slots, 48, 128, 128}) &&
            tensors[1].sizes() == at::IntArrayRef({batch, 10240}) &&
            tensors[2].sizes() == at::IntArrayRef({batch, 48}) &&
            tensors[3].sizes() == at::IntArrayRef({batch, 48}) &&
            tensors[4].sizes() == at::IntArrayRef({48}) &&
            tensors[5].sizes() == at::IntArrayRef({48}) &&
            tensors[6].sizes() == at::IntArrayRef({batch}) &&
            tensors[7].sizes() == at::IntArrayRef({batch, 48, 128}),
        "packed GDN decode tensor storage does not match the Qwen3.5 specialization");
  } else if (artifact->kernel_kind == KernelKind::GdnDecodeConvPacked) {
    const auto batch = static_cast<std::int64_t>(grid[1]);
    const auto conv_slots = static_cast<std::int64_t>(scalar_params.at(0));
    const auto state_slots = static_cast<std::int64_t>(scalar_params.at(1));
    require(
        tensors.size() == 10 && conv_slots > 0 && state_slots > 0 &&
            tensors[0].sizes() ==
                at::IntArrayRef({conv_slots, 3, 10240}) &&
            tensors[1].sizes() ==
                at::IntArrayRef({state_slots, 48, 128, 128}) &&
            tensors[2].sizes() == at::IntArrayRef({batch, 10240}) &&
            tensors[3].sizes() == at::IntArrayRef({batch, 48}) &&
            tensors[4].sizes() == at::IntArrayRef({batch, 48}) &&
            tensors[5].sizes() == at::IntArrayRef({48}) &&
            tensors[6].sizes() == at::IntArrayRef({48}) &&
            tensors[7].sizes() == at::IntArrayRef({batch}) &&
            tensors[8].sizes() == at::IntArrayRef({4, 10240}) &&
            tensors[9].sizes() == at::IntArrayRef({batch, 48, 128}),
        "fused conv+GDN tensor storage does not match the Qwen3.5 specialization");
  } else if (artifact->kernel_kind == KernelKind::GdnQkConvPacked) {
    const auto batch = static_cast<std::int64_t>(grid[1]);
    const auto conv_slots = static_cast<std::int64_t>(scalar_params.at(0));
    require(
        tensors.size() == 5 && conv_slots > 0 &&
            tensors[0].sizes() ==
                at::IntArrayRef({conv_slots, 3, 10240}) &&
            tensors[1].sizes() == at::IntArrayRef({batch, 10240}) &&
            tensors[2].sizes() == at::IntArrayRef({batch}) &&
            tensors[3].sizes() == at::IntArrayRef({4, 10240}) &&
            tensors[4].sizes() == at::IntArrayRef({batch, 4096}),
        "packed Q/K convolution tensor storage does not match Qwen3.5");
  } else if (artifact->kernel_kind == KernelKind::GdnDecodeValueConvPacked) {
    const auto batch = static_cast<std::int64_t>(grid[2]);
    const auto conv_slots = static_cast<std::int64_t>(scalar_params.at(0));
    const auto state_slots = static_cast<std::int64_t>(scalar_params.at(1));
    require(
        grid[0] == 128 / artifact->block_size && tensors.size() == 11 &&
            conv_slots > 0 && state_slots > 0 &&
            tensors[0].sizes() ==
                at::IntArrayRef({conv_slots, 3, 10240}) &&
            tensors[1].sizes() ==
                at::IntArrayRef({state_slots, 48, 128, 128}) &&
            tensors[2].sizes() == at::IntArrayRef({batch, 4096}) &&
            tensors[3].sizes() == at::IntArrayRef({batch, 10240}) &&
            tensors[4].sizes() == at::IntArrayRef({batch, 48}) &&
            tensors[5].sizes() == at::IntArrayRef({batch, 48}) &&
            tensors[6].sizes() == at::IntArrayRef({48}) &&
            tensors[7].sizes() == at::IntArrayRef({48}) &&
            tensors[8].sizes() == at::IntArrayRef({batch}) &&
            tensors[9].sizes() == at::IntArrayRef({4, 10240}) &&
            tensors[10].sizes() == at::IntArrayRef({batch, 48, 128}),
        "fused value-conv + GDN tensor storage does not match Qwen3.5");
  } else if (artifact->bound_scalar_position >= 0) {
    const auto logical_elements = scalar_params.at(
        static_cast<std::size_t>(artifact->bound_scalar_position));
    require(logical_elements > 0, "masked TPC launch requires a positive runtime bound");
    const auto expected_grid =
        (static_cast<std::uint64_t>(logical_elements) + artifact->block_size - 1) /
        artifact->block_size;
    require(
        grid[0] == expected_grid,
        "launch grid does not cover the masked runtime bound exactly");
    for (const auto& tensor : tensors) {
      require(
          logical_elements <= static_cast<std::uint64_t>(tensor.numel()),
          "runtime bound exceeds a kernel tensor's storage");
    }
  } else {
    const auto covered_elements = grid[0] * artifact->block_size;
    for (const auto& tensor : tensors) {
      require(
          covered_elements <= static_cast<std::uint64_t>(tensor.numel()),
          "unmasked launch grid exceeds a kernel tensor's storage");
    }
  }
  for (std::size_t index = 0; index < tensors.size(); ++index) {
    validate_tensor(
        tensors[index], artifact->tensor_dtypes.at(index), artifact->device_id);
  }

  auto* execution_context = habana_lazy::get_device_lazy_execution_context();
  require(
      execution_context->getCaptureGraph() == nullptr,
      "direct TPC launch is not HPUGraph-capturable; use GaudiCompiledGraphOp");
  // Materialize pending lazy producers before submitting the external recipe.
  // In eager mode producer events already describe the dependency and calling
  // StepMarker would add avoidable host overhead.
  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) == 1) {
    habana_lazy::HbLazyTensor::StepMarker({});
  }

  const auto recipe =
      get_or_compile_recipe(artifact, grid, tensors, scalar_params);
  auto resources = std::make_shared<LaunchResources>();
  resources->tensors = tensors;
  resources->artifact = artifact;
  resources->recipe = recipe;
  std::vector<synapse_helpers::device_ptr> addresses;
  std::vector<synapse_helpers::device_ptr> input_addresses;
  std::vector<synapse_helpers::device_ptr> output_addresses;
  addresses.reserve(tensors.size());
  for (std::size_t index = 0; index < tensors.size(); ++index) {
    const auto address = reinterpret_cast<synapse_helpers::device_ptr>(
        tensors[index].data_ptr());
    addresses.push_back(address);
    if (index < artifact->input_count) {
      input_addresses.push_back(address);
    } else {
      output_addresses.push_back(address);
    }
  }
  for (const auto position : artifact->mutable_input_positions) {
    output_addresses.push_back(
        addresses.at(position));
  }

  auto& device = HPUDeviceContext::get_device();
  auto& stream = device.get_stream(hpu_stream, synapse_helpers::COMPUTE);
  device.add_wait_events_on_stream(input_addresses, stream);
  std::vector<synLaunchTensorInfo> launch_info;
  launch_info.reserve(tensors.size());
  for (std::size_t index = 0; index < tensors.size(); ++index) {
    synLaunchTensorInfo info{};
    info.tensorName = recipe->tensor_names[index].c_str();
    info.pTensorAddress = addresses[index];
    info.tensorType = DATA_TENSOR;
    if (artifact->kernel_kind == KernelKind::SiluAndMul ||
        artifact->kernel_kind == KernelKind::GdnDecodePacked ||
        artifact->kernel_kind == KernelKind::GdnDecodeConvPacked ||
        artifact->kernel_kind == KernelKind::GdnQkConvPacked ||
        artifact->kernel_kind == KernelKind::GdnDecodeValueConvPacked) {
      for (std::size_t dimension = 0;
           dimension < static_cast<std::size_t>(tensors[index].dim());
           ++dimension) {
        info.tensorSize[dimension] = static_cast<std::uint64_t>(
            tensors[index].size(tensors[index].dim() - dimension - 1));
      }
    } else {
      info.tensorSize[0] = static_cast<std::uint64_t>(tensors[index].numel());
    }
    info.tensorId = recipe->tensor_ids[index];
    launch_info.push_back(info);
  }
  std::vector<synapse_helpers::shared_event> external_events;
  synapse_helpers::graph::launch(
      device,
      *recipe->handle,
      recipe->workspace_size,
      launch_info,
      resources->address_lock,
      external_events,
      stream);

  // Binding the completion event to outputs lets later Bridge work consume the
  // result without a host synchronization.  The callback retains every input,
  // the recipe and the allocator address lock until device completion.
  device.register_producer_on_stream(
      std::move(output_addresses),
      stream,
      [resources = std::move(resources)]() mutable { resources.reset(); });
}

void launch_v2(
    std::uint64_t handle,
    std::uint64_t hpu_stream,
    const std::vector<at::Tensor>& tensors,
    const std::vector<std::uint8_t>& packet) {
  const auto artifact = lookup(handle);
  require(
      packet.size() >= sizeof(LaunchPacketHeaderV2),
      "truncated launch ABI v2 packet");
  LaunchPacketHeaderV2 header{};
  std::memcpy(&header, packet.data(), sizeof(header));
  require(
      std::memcmp(header.magic, kLaunchPacketMagic, sizeof(header.magic)) == 0 &&
          header.abi_major == kBridgeLaunchAbiMajor &&
          header.abi_minor <= kBridgeLaunchAbiMinor,
      "unsupported launch ABI v2 packet");
  require(
      header.reserved[0] == 0 && header.reserved[1] == 0,
      "launch ABI v2 reserved bytes must be zero");
  require(
      header.grid_rank >= 1 && header.grid_rank <= 3 &&
          header.tensor_count <= kMaxLaunchTensors &&
          header.scalar_count <= kMaxScalarParams,
      "launch ABI v2 packet exceeds the current TPC limits");
  const std::size_t expected_size = sizeof(header) +
      static_cast<std::size_t>(header.tensor_count) *
          sizeof(TensorBindingV2) +
      static_cast<std::size_t>(header.scalar_count) *
          sizeof(ScalarBindingV2);
  require(
      header.struct_size == expected_size && packet.size() == expected_size,
      "truncated or trailing data in launch ABI v2 packet");
  for (std::size_t index = header.grid_rank;
       index < kMaxIndexSpaceRank;
       ++index) {
    require(header.grid[index] == 0, "launch ABI v2 grid padding must be zero");
  }
  require(
      std::string(header.artifact_hash, kArtifactHashChars) == artifact->hash,
      "launch packet and registered artifact hashes differ");
  require(
      header.tensor_count == tensors.size() &&
          header.tensor_count == artifact->tensor_argument_indices.size(),
      "launch ABI v2 tensor binding count mismatch");
  require(
      header.scalar_count == artifact->scalar_argument_indices.size(),
      "launch ABI v2 scalar binding count mismatch");

  std::size_t offset = sizeof(header);
  for (std::size_t index = 0; index < header.tensor_count; ++index) {
    TensorBindingV2 binding{};
    std::memcpy(&binding, packet.data() + offset, sizeof(binding));
    require(
        binding.argument_index == artifact->tensor_argument_indices.at(index) &&
            binding.dtype == artifact->tensor_dtype_codes.at(index) &&
            binding.role == artifact->tensor_roles.at(index) &&
            binding.flags == 0,
        "launch ABI v2 tensor binding does not match the artifact manifest");
    offset += sizeof(binding);
  }

  std::vector<std::uint32_t> scalar_params;
  scalar_params.reserve(header.scalar_count);
  for (std::size_t index = 0; index < header.scalar_count; ++index) {
    ScalarBindingV2 binding{};
    std::memcpy(&binding, packet.data() + offset, sizeof(binding));
    const auto& expected_dtype = artifact->scalar_dtypes.at(index);
    require(
        binding.argument_index == artifact->scalar_argument_indices.at(index) &&
            binding.dtype == dtype_code(expected_dtype) &&
            binding.reserved == 0,
        "launch ABI v2 scalar binding does not match the artifact manifest");
    require(
        (expected_dtype == "i32" || expected_dtype == "u32" ||
         expected_dtype == "f32") &&
            binding.bits <= std::numeric_limits<std::uint32_t>::max(),
        "the current TPC node-params ABI accepts 32-bit scalars only");
    scalar_params.push_back(static_cast<std::uint32_t>(binding.bits));
    offset += sizeof(binding);
  }

  std::vector<std::uint64_t> grid{
      header.grid[0],
      header.grid_rank >= 2 ? header.grid[1] : 1,
      header.grid_rank >= 3 ? header.grid[2] : 1};
  launch(handle, grid, hpu_stream, tensors, scalar_params);
}

} // namespace habana::triton_gaudi
