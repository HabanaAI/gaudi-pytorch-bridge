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

#pragma once
#include <sys/types.h>
#include <string_view>

namespace synapse_logger {
class SynapseLoggerObserver {
 public:
  virtual ~SynapseLoggerObserver(){};
  virtual void on_log(
      std::string_view name,
      std::string_view args,
      pid_t pid,
      pid_t tid,
      int64_t dtime,
      bool is_begin) = 0;
  virtual bool enabled(std::string_view name) = 0;
};

extern "C" void register_synapse_logger_oberver(
    SynapseLoggerObserver* synapse_logger_observer);
}; // namespace synapse_logger