/**
 * Copyright (c) 2026 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 */
#pragma once

#include <ATen/core/Tensor.h>

#include <cstdint>
#include <string>
#include <vector>

namespace habana::triton_gaudi {

std::uint64_t register_artifact(
    const std::string& artifact_hash,
    const std::vector<std::uint8_t>& elf,
    const std::string& manifest_json,
    int device_id);

std::uint64_t register_artifact_v2(
    const std::vector<std::uint8_t>& artifact,
    int device_id);

void unregister_artifact(std::uint64_t handle);

void launch(
    std::uint64_t handle,
    const std::vector<std::uint64_t>& grid,
    std::uint64_t hpu_stream,
    const std::vector<at::Tensor>& tensors,
    const std::vector<std::uint32_t>& scalar_params);

void launch_v2(
    std::uint64_t handle,
    std::uint64_t hpu_stream,
    const std::vector<at::Tensor>& tensors,
    const std::vector<std::uint8_t>& packet);

} // namespace habana::triton_gaudi
