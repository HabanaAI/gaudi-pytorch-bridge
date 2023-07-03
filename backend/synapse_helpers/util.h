/*******************************************************************************
 * Copyright (C) 2020-2023 Habana Labs, Ltd. an Intel Company
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
