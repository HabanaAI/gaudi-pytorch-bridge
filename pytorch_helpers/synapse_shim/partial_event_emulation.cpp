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

#include "synapse_shim/partial_event_emulation.h"

synStatus PartialEventEmulation::synTensorExtExtractExecutionOrder(
    const synRecipeHandle /*unused*/,
    uint32_t /*unused*/,
    uint64_t* /*unused*/) {
  return synSuccess;
}

synStatus PartialEventEmulation::synLaunchWithExternalEvents(
    const synStreamHandle streamHandle,
    const synLaunchTensorInfo* launchTensorsInfo,
    const uint32_t numberOfTensors,
    uint64_t pWorkspace,
    const synRecipeHandle pRecipeHandle,
    synEventHandle* eventHandleList,
    const uint32_t numberOfEvents,
    uint32_t flags) {
  auto status = synLaunch(
      streamHandle,
      launchTensorsInfo,
      numberOfTensors,
      pWorkspace,
      pRecipeHandle,
      flags);
  if (status != synSuccess) {
    return status;
  }

  for (uint32_t i = 0; i < numberOfEvents; ++i) {
    status = synEventRecord(eventHandleList[i], streamHandle);
    if (status != synSuccess) {
      return status;
    }
  }

  return status;
}

synStatus PartialEventEmulation::synEventMapTensor(
    synEventHandle* /*unused*/,
    size_t /*unused*/,
    const synLaunchTensorInfo* /*unused*/,
    const synRecipeHandle /*unused*/) {
  return synSuccess;
}

synStatus PartialEventEmulation::synTensorSetExternal(
    synTensor tensor,
    bool isExternal) {
  std::unique_lock<std::mutex> lock(mutex_);
  if (isExternal) {
    external_tensors_.insert(tensor);
  } else {
    external_tensors_.erase(tensor);
  }
  return synSuccess;
}

synStatus PartialEventEmulation::synTensorGetExternal(
    const synTensor tensor,
    bool* isExternal) {
  std::unique_lock<std::mutex> lock(mutex_);
  *isExternal = external_tensors_.count(tensor);
  return synSuccess;
}

bool UsePartialEventEmulation() {
  const char* emulateSignalingEnv =
      std::getenv("PT_HPU_EMULATE_SIGNALING_FROM_ENCAP_OP");
  if (!emulateSignalingEnv)
    return false;
  return (std::string_view(emulateSignalingEnv) == "true");
}
