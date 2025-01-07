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
