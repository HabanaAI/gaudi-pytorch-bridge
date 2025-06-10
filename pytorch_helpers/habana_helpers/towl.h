/**
 * Copyright (c) 2021-2025 Intel Corporation
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

#include <cstdint>
#include "backend/synapse_helpers/device_types.h"
#include "backend/synapse_helpers/graph.h"
#include "logging.h"
namespace towl {

namespace impl {

struct TowlEnabled {
  static bool flag;
};

void emitDeviceMemoryAllocated(
    void* ptr,
    std::size_t size,
    std::uint64_t stream,
    bool is_physical = false);
void emitDeviceMemoryDeallocated(void* ptr, bool is_physical = false);
void emitDeviceMemoryAllocSuccess(
    void* ptr,
    std::size_t size,
    bool is_workspace);
void emitDeviceMemoryAllocFailed(std::size_t size, bool is_workspace);
void emitDeviceMemorySnapshot();
void emitRecipeLaunch(
    const synapse_helpers::graph::recipe_handle& recipe_handle,
    uint64_t workspace_size,
    const std::vector<std::uint64_t>& addresses,
    const std::vector<synLaunchTensorInfo>& tensors,
    bool is_physical = false);
void emitRecipeFinished(
    const synapse_helpers::graph::recipe_handle* recipe_handle);
void emitCollectiveLaunch(const std::string& info);
void emitCollectiveFinished(const std::string& info);

void emitDefragLaunch(const std::string& info);
void emitDefragFinished(const std::string& info);

void emitPythonString(const std::string& s);

void emitDeviceMemorySummary(const char* tag);

void emitCopyLaunch(const char* tag, void* src, void* dst, size_t bytes);
void emitCopyFinished(const char* tag, void* src, void* dst);

void emitCopyMultipleLaunch(
    const char* tag,
    const uint64_t* srcs,
    const uint64_t* dsts,
    const uint64_t* sizes,
    size_t num_copies);
void emitCopyMultipleFinished(
    const char* tag,
    std::shared_ptr<synapse_helpers::device_ptr_lock>& locked);

void emitRecipeCompileSuccess(
    const synapse_helpers::graph::recipe_handle& recipe_handle,
    uint64_t workspace_size,
    const std::string& name,
    double compile_duration);

void emitRecipeCompileFailed(
    const std::string& error_info,
    double compile_duration);

void emitTimeDurationJit(const std::string& name, float time);
void emitTimeDurationFX(const std::string& name, float time);

void emitMetrics(const std::string& name, float value);

void emitRecipeName(const std::string& param_data);
void emitRecipeHandle(synRecipeHandle recipe_handle);
void emitRecipeRequireWorkspace(const std::string& workspace);
void emitRecipeTensorToUse(const std::string& dtensorinfo_dump);
} // namespace impl

/*
 * Entrypoints check directly if towl is enabled. To reduce performance
 * penalty by existence of loggers we directly check the flag before
 * entering actual implementation.
 */
#define _MAKE_TOWL_ENTRYPOINT(name, DEF_ARGS, CALL_ARGS) \
  inline void name DEF_ARGS {                            \
    if (::towl::impl::TowlEnabled::flag) {               \
      ::towl::impl::name CALL_ARGS;                      \
    }                                                    \
  }

namespace {
_MAKE_TOWL_ENTRYPOINT(
    emitDeviceMemoryAllocated,
    (void* ptr,
     std::size_t size,
     std::uint64_t stream,
     bool is_physical = false),
    (ptr, size, stream, is_physical))
_MAKE_TOWL_ENTRYPOINT(
    emitDeviceMemoryDeallocated,
    (void* ptr, bool is_physical = false),
    (ptr, is_physical))
_MAKE_TOWL_ENTRYPOINT(
    emitRecipeLaunch,
    (const synapse_helpers::graph::recipe_handle& recipe_handle,
     uint64_t workspace_size,
     const std::vector<std::uint64_t>& locked_addresses,
     const std::vector<synLaunchTensorInfo>& tensors,
     bool is_physical = false),
    (recipe_handle, workspace_size, locked_addresses, tensors, is_physical))
_MAKE_TOWL_ENTRYPOINT(
    emitRecipeFinished,
    (const synapse_helpers::graph::recipe_handle* recipe_handle),
    (recipe_handle));
_MAKE_TOWL_ENTRYPOINT(emitCollectiveLaunch, (const std::string& info), (info));
_MAKE_TOWL_ENTRYPOINT(
    emitCollectiveFinished,
    (const std::string& info),
    (info));

_MAKE_TOWL_ENTRYPOINT(emitDefragLaunch, (const std::string& info), (info));
_MAKE_TOWL_ENTRYPOINT(emitDefragFinished, (const std::string& info), (info));

_MAKE_TOWL_ENTRYPOINT(emitPythonString, (const std::string& s), (s));
_MAKE_TOWL_ENTRYPOINT(emitDeviceMemorySummary, (const char* tag), (tag));

_MAKE_TOWL_ENTRYPOINT(
    emitCopyLaunch,
    (const char* tag, void* src, void* dst, size_t bytes),
    (tag, src, dst, bytes));
_MAKE_TOWL_ENTRYPOINT(
    emitCopyFinished,
    (const char* tag, void* src, void* dst),
    (tag, src, dst));

_MAKE_TOWL_ENTRYPOINT(
    emitCopyMultipleLaunch,
    (const char* tag,
     const uint64_t* srcs,
     const uint64_t* dsts,
     const uint64_t* sizes,
     size_t num_copies),
    (tag, srcs, dsts, sizes, num_copies));
_MAKE_TOWL_ENTRYPOINT(
    emitCopyMultipleFinished,
    (const char* tag,
     std::shared_ptr<synapse_helpers::device_ptr_lock>& locked),
    (tag, locked));
_MAKE_TOWL_ENTRYPOINT(
    emitMetrics,
    (const std::string& name, float value),
    (name, value));

_MAKE_TOWL_ENTRYPOINT(
    emitTimeDurationJit,
    (const std::string& name, float value),
    (name, value));

_MAKE_TOWL_ENTRYPOINT(
    emitTimeDurationFX,
    (const std::string& name, float value),
    (name, value));

_MAKE_TOWL_ENTRYPOINT(
    emitRecipeCompileSuccess,
    (const synapse_helpers::graph::recipe_handle& recipe_handle,
     uint64_t workspace_size,
     const std::string& name,
     double compile_duration),
    (recipe_handle, workspace_size, name, compile_duration));

_MAKE_TOWL_ENTRYPOINT(
    emitRecipeCompileFailed,
    (const std::string& error_info, double compile_duration),
    (error_info, compile_duration));

_MAKE_TOWL_ENTRYPOINT(
    emitDeviceMemoryAllocSuccess,
    (void* ptr, std::size_t size, bool is_workspace),
    (ptr, size, is_workspace));

_MAKE_TOWL_ENTRYPOINT(
    emitDeviceMemoryAllocFailed,
    (std::size_t size, bool is_workspace),
    (size, is_workspace));

_MAKE_TOWL_ENTRYPOINT(
    emitRecipeName,
    (const std::string& param_data),
    (param_data));

_MAKE_TOWL_ENTRYPOINT(
    emitRecipeHandle,
    (synRecipeHandle recipe_handle),
    (recipe_handle));

_MAKE_TOWL_ENTRYPOINT(
    emitRecipeRequireWorkspace,
    (const std::string& workspace),
    (workspace));

_MAKE_TOWL_ENTRYPOINT(
    emitRecipeTensorToUse,
    (const std::string& dtensorinfo_dump),
    (dtensorinfo_dump));
} // namespace

void configure(bool enable, std::string config);

#undef _MAKE_TOWL_FRONTEND
} // namespace towl
