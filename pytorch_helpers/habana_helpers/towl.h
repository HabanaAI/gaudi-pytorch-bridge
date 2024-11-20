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

#include <hl_logger/hllog.hpp>
#include <cstdint>
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
    std::uint64_t stream);
void emitDeviceMemoryDeallocated(void* ptr);
void emitDeviceMemorySnapshot();
void emitRecipeLaunch(
    const synapse_helpers::graph::recipe_handle& recipe_handle,
    uint64_t workspace_size,
    const std::vector<std::uint64_t>& addresses,
    const std::vector<synLaunchTensorInfo>& tensors);
void emitRecipeFinished(
    const synapse_helpers::graph::recipe_handle* recipe_handle);

void emitPythonString(const std::string& s);

void emitDeviceMemorySummary(const char* tag);
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
    (void* ptr, std::size_t size, std::uint64_t stream),
    (ptr, size, stream))
_MAKE_TOWL_ENTRYPOINT(emitDeviceMemoryDeallocated, (void* ptr), (ptr))
_MAKE_TOWL_ENTRYPOINT(
    emitRecipeLaunch,
    (const synapse_helpers::graph::recipe_handle& recipe_handle,
     uint64_t workspace_size,
     const std::vector<std::uint64_t>& locked_addresses,
     const std::vector<synLaunchTensorInfo>& tensors),
    (recipe_handle, workspace_size, locked_addresses, tensors))
_MAKE_TOWL_ENTRYPOINT(
    emitRecipeFinished,
    (const synapse_helpers::graph::recipe_handle* recipe_handle),
    (recipe_handle));
_MAKE_TOWL_ENTRYPOINT(emitPythonString, (const std::string& s), (s));
_MAKE_TOWL_ENTRYPOINT(emitDeviceMemorySummary, (const char* tag), (tag));
} // namespace

void configure(bool enable, std::string config);

#undef _MAKE_TOWL_FRONTEND
} // namespace towl
