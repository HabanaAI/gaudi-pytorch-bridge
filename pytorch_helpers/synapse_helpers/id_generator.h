/******************************************************************************
 * Copyright (C) 2021 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 *******************************************************************************/
#pragma once

#include <type_traits>
#include <vector>
#include "habana_helpers/logging.h"

namespace synapse_helpers {

template <typename T>
class id_generator {
  static_assert(std::is_integral<T>::value, "Integrals required!");

 public:
  id_generator() {
    free_.reserve(1024); // Prellocate space for freed ids
  }
  ~id_generator() {
    reset();
  }

  T get() {
    if (!free_.empty()) {
      return get_free();
    } else {
      return generate_new();
    }
  }

  void put(T value) {
    free_.emplace_back(value);
  }

  void reset() {
    counter_ = 0;
    free_.clear();
  }

 private:
  T generate_new() {
    T out = ++counter_;
    if (out == 0) {
      // Overflow occured what means we cannot proceed.
      // Whole id space is in use. Otherwise get_free function
      // would return one.
      PT_SYNHELPER_FATAL("Overflow occured, Run out of ids");
    }
    return out;
  }
  T get_free() {
    T out = std::move(*free_.crbegin());
    free_.pop_back();
    return out;
  }

  T counter_ = 0;
  std::vector<T> free_;
};

} // namespace synapse_helpers
