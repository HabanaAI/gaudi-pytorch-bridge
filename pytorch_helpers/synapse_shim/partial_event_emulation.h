/******************************************************************************
 * Copyright (C) 2021 Habana Labs, Ltd. an Intel Company
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

// TODO: The emulator should be removed once feature is ready on Synapse side.

#pragma once

#include <synapse_api.h> // IWYU pragma: keep
#include <mutex>
#include <unordered_map>
#include <unordered_set>

class PartialEventEmulation {
 public:
  ~PartialEventEmulation() = default;

  PartialEventEmulation(PartialEventEmulation const&) = delete;
  void operator=(PartialEventEmulation const&) = delete;
  PartialEventEmulation(PartialEventEmulation&&) = delete;
  void operator=(PartialEventEmulation&&) = delete;

  static PartialEventEmulation& Instance() {
    static PartialEventEmulation instance;
    return instance;
  }

  synStatus synTensorExtExtractExecutionOrder(
      const synRecipeHandle recipeHandle,
      uint32_t numOfExternalTensors,
      uint64_t* tensorIds);
  synStatus synLaunchWithExternalEvents(
      const synStreamHandle streamHandle,
      const synLaunchTensorInfo* launchTensorsInfoExt,
      const uint32_t numberOfTensors,
      uint64_t pWorkspace,
      const synRecipeHandle pRecipeHandle,
      synEventHandle* eventHandleList,
      const uint32_t numberOfEvents,
      uint32_t flags);
  synStatus synEventMapTensor(
      synEventHandle* eventHandle,
      size_t numOfEvents,
      const synLaunchTensorInfo* launchTensorsInfo,
      const synRecipeHandle recipeHandle);
  synStatus synTensorSetExternal(synTensor tensor, bool isExternal);
  synStatus synTensorGetExternal(const synTensor tensor, bool* isExternal);

 private:
  PartialEventEmulation() = default;

  std::unordered_set<synTensor> external_tensors_;
  std::mutex mutex_;
};

bool UsePartialEventEmulation();
