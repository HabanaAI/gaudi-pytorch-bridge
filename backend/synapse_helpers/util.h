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

#include <ios>
#include <ostream>
#include <unordered_map>

namespace synapse_helpers {

std::string uint64_to_hex_string(uint64_t number);

/* END: These will be removed when all lazy kernels use shape function. */

class ostream_flag_guard {
 public:
  [[nodiscard]] static ostream_flag_guard create(std::ostream& stream) {
    return ostream_flag_guard{stream};
  }

  ~ostream_flag_guard() {
    stream_.flags(flags_);
  }

 private:
  explicit ostream_flag_guard(std::ostream& stream)
      : stream_{stream}, flags_{stream.flags()} {}

  std::ostream& stream_;
  std::ios_base::fmtflags flags_;
};

} // namespace synapse_helpers
