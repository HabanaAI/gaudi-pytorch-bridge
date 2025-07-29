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

#include <string>

class CacheVersion {
 public:
  // returns string containing Hash calculated for content of:
  //    - habana_device, synapse_helpers, Synapse and tpc_kernels (binary
  //    content)
  //    - env variables impacting the compilation of synGraph to synRecipe
  static std::string libs_env_hash();
  // return <PID>_<MAC_ADDR> (twelve 0s (zero) for MAC,
  // if no valid network interface is found)
  static std::string combined_pid_mac_addr();
};
