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

#include <absl/types/optional.h>
#include <absl/types/variant.h>
#include <synapse_api.h>

namespace synapse_helpers {

// TODO: replace with tl::expected or something similar
struct synapse_error {
  std::string error;
  synStatus status; // TODO: uncomment once we have C++14 and remove assert
                    // below: = synStatus::synSuccess
  static_assert(
      synStatus::synSuccess == 0,
      "default-initialized synapse_error shouldn't contain an error code");
};

using synapse_error_o = absl::optional<synapse_error>;
template <typename T>
using synapse_error_v = absl::variant<T, synapse_error>;

template <typename T>
inline T& get_value(synapse_error_v<T>& variant) {
  return absl::get<T>(variant);
}

class tensor;

// Bug-prone case -- prohibit.
// Usually such a variant gets returned from functions creating tensors, so we
// should extract the tensor to somewhere instead of just getting a reference.
// When the variant goes out of scope, the tensor would get destroyed.
template <>
inline tensor& get_value(synapse_error_v<tensor>& variant) = delete;

template <typename T>
inline const T& get_value(const synapse_error_v<T>& variant) {
  return absl::get<T>(variant);
}

template <typename T>
inline T& get_value(synapse_error_v<std::reference_wrapper<T>>& variant) {
  return absl::get<std::reference_wrapper<T>>(variant).get();
}

template <typename T>
inline T get_value(synapse_error_v<T>&& variant) {
  return absl::get<T>(std::move(variant));
}

template <typename T>
inline T& get_value(synapse_error_v<std::reference_wrapper<T>>&& variant) {
  return absl::get<std::reference_wrapper<T>>(std::move(variant)).get();
}

template <typename alternative_t>
inline bool ok(synapse_error_v<alternative_t>& error_variant) {
  return !absl::holds_alternative<synapse_error>(error_variant) ||
      absl::get<synapse_error>(error_variant).status == synSuccess;
}

template <typename alternative_t>
inline synapse_error& get_error(synapse_error_v<alternative_t>& error_variant) {
  return absl::get<synapse_error>(error_variant);
}

inline bool ok(synapse_error_o& error_optional) {
  return !error_optional.has_value() ||
      error_optional.value().status == synSuccess;
}

inline synapse_error& get_error(synapse_error_o& error_optional) {
  return error_optional.value();
}

inline bool ok(const synapse_error& error) {
  return error.status == synSuccess;
}

inline synapse_error& get_error(synapse_error& error) {
  return error;
}
inline bool ok(bool success) {
  return success;
}

inline synapse_error& get_error(bool /*success*/) {
  static synapse_error e{"fail", synFail};
  return e;
}
} // namespace synapse_helpers

#define SYNAPSE_SUCCESS_CHECK(error, status)                   \
  if (ABSL_PREDICT_FALSE(status != synStatus::synSuccess)) {   \
    PT_SYNHELPER_WARN(Logger::formatStatusMsg(status), error); \
    return synapse_helpers::synapse_error{error, status};      \
  }

#define SYNAPSE_SUCCESS_CHECK_WITH_OP(error, status, op)       \
  if (ABSL_PREDICT_FALSE(status != synStatus::synSuccess)) {   \
    PT_SYNHELPER_WARN(Logger::formatStatusMsg(status), error); \
    op;                                                        \
    return synapse_helpers::synapse_error{error, status};      \
  }

#define SYNAPSE_RETURN_IF_ERROR(error_carrier_for_eval) \
  do {                                                  \
    auto&& error_carrier{error_carrier_for_eval};       \
    if (ABSL_PREDICT_FALSE(!ok(error_carrier))) {       \
      return get_error(error_carrier);                  \
    }                                                   \
  } while (false)

#define SYNAPSE_RETURN_IF_ERROR_V(error_variant_for_eval)             \
  do {                                                                \
    auto&& error_variant{error_variant_for_eval};                     \
    if (ABSL_PREDICT_FALSE(                                           \
            absl::holds_alternative<synapse_error>(error_variant))) { \
      return absl::get<synapse_error>(error_variant);                 \
    }                                                                 \
  } while (false)
