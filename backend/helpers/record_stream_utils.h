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

#ifndef RECORD_STREAM_UTILS_H
#define RECORD_STREAM_UTILS_H

#include <vector>
#include "backend/synapse_helpers/device.h"

namespace stream_utils {

template <typename T>
inline void GenericRecordStream(
    synapse_helpers::device& device,
    synapse_helpers::hpuStream_t hpu_stream,
    const std::vector<T>& pointers) {
  if (pointers.empty()) {
    return;
  }

  auto& memory = device.get_device_memory();
  for (auto data_ptr : pointers) {
    memory.recordStream(reinterpret_cast<void*>(data_ptr), hpu_stream);
  }
}

} // namespace stream_utils

#endif // RECORD_STREAM_UTILS_H
