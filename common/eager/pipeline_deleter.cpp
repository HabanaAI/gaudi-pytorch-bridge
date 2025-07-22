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

#include "common/pipeline_deleter.h"
#include <sys/syscall.h>
#include "backend/habana_device/HPUAllocator.h"
#include "backend/synapse_helpers/env_flags.h"
#include "common/utils.h"
#include "habana_eager/eager_pipeline_utils.h"
namespace common {

using namespace ::habana;

namespace {

long get_thread_tid() {
  return syscall(static_cast<long>(SYS_gettid));
}

bool is_enabled() {
  return IsStreamAllocatorEnabled();
}
} // namespace

void PipelineDeleter::install() {
  if (not is_enabled()) {
    return;
  }
  m_marked_tid = get_thread_tid();
  HPUDeviceAllocator::deleter_hook = [this](void* ptr) {
    delete_function(ptr);
  };
}

void PipelineDeleter::uninstall() {
  m_marked_tid = 0;
  HPUDeviceAllocator::deleter_hook = nullptr;
}

void PipelineDeleter::delete_function(void* ptr) {
  if (get_thread_tid() == m_marked_tid or m_marked_tid == 0) {
    HPUDeviceAllocator::real_deleter(ptr);
    return;
  }

  eager::PipelineTaskAllThreads(
      std::move(ptr),
      [](void*&) {},
      [](void*&) {},
      HPUDeviceAllocator::real_deleter);
}

} // namespace common
