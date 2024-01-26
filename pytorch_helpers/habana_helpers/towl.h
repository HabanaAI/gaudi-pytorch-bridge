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
