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

namespace common {

/*
 * PipelineDeleter is utility to delegate device memory deallocations
 * to execution thread through pipeline.
 *
 * Execution threads calls `mark_calling_thread_as_deleter` during
 * its initialization. Later all memory deallocations done in this
 * thread are real deallocations while deallocations done in other
 * threads are delegate to pipeline.
 *
 * Deleter created by method `make_deleter` is installed inside
 * HPUAllocator.
 */
class PipelineDeleter {
 public:
  void install();
  void uninstall();

  static PipelineDeleter& instance() {
    static PipelineDeleter _instance;
    return _instance;
  }

 private:
  PipelineDeleter() {}

  void delete_function(void*) const;

  long m_marked_tid = 0;
};

} // namespace common
