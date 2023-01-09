/******************************************************************************
 * Copyright (C) 2020 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */
#pragma once

#include <ios>
#include <ostream>
#include <unordered_map>

namespace synapse_helpers {

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
