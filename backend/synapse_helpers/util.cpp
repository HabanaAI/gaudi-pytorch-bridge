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

#include <iomanip>
#include <sstream>

#include "util.h"

namespace synapse_helpers {
std::string uint64_to_hex_string(uint64_t number) {
  std::ostringstream ss;
  ss << "0x" << std::setfill('0') << std::setw(12) << std::hex << number;
  return ss.str();
}
} // namespace synapse_helpers