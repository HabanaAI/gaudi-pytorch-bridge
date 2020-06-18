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

#if __cplusplus > 201703L
#define NODISCARD [[nodiscard]]
#define UNUSED [[maybe_unused]]
#else
#if defined(__clang__) || defined(__GNUC__)
#define NODISCARD __attribute__((warn_unused_result))
#define UNUSED __attribute__((unused))
#endif
#endif

namespace synapse_helpers {

class ostream_flag_guard {
 public:
  static NODISCARD ostream_flag_guard create(std::ostream& stream) {
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
