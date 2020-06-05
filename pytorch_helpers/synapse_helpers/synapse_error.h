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

#include <absl/types/optional.h>
#include <absl/types/variant.h>
#include <synapse.h>

namespace synapse_helpers {

// TODO: replace with tl::expected or something similar
struct synapse_error {
  std::string error;
  synStatus status;  // TODO: uncomment once we have C++14 and remove assert below: = synStatus::synSuccess
  static_assert(synStatus::synSuccess == 0, "default-initialized synapse_error shouldn't contain an error code");
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
// Usually such a variant gets returned from functions creating tensors, so we should extract the tensor to somewhere
// instead of just getting a reference. When the variant goes out of scope, the tensor would get destroyed.
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
  return !error_optional.has_value() || error_optional.value().status == synSuccess;
}

inline synapse_error& get_error(synapse_error_o& error_optional) { return error_optional.value(); }

inline bool ok(const synapse_error& error) { return error.status == synSuccess; }

inline synapse_error& get_error(synapse_error& error) { return error; }
inline bool ok(bool success) { return success; }

inline synapse_error& get_error(bool /*success*/) {
  static synapse_error e{"fail", synFail};
  return e;
}
}  // namespace synapse_helpers

#define SYNAPSE_SUCCESS_CHECK(error, status)                 \
  if (ABSL_PREDICT_FALSE(status != synStatus::synSuccess)) { \
    LOG_(ERROR) << error << " Err: " << status;               \
    return synapse_helpers::synapse_error{error, status};    \
  }

#define SYNAPSE_SUCCESS_CHECK_WITH_OP(error, status, op)     \
  if (ABSL_PREDICT_FALSE(status != synStatus::synSuccess)) { \
    LOG_(ERROR) << error << " Err: " << status;               \
    op;                                                      \
    return synapse_helpers::synapse_error{error, status};    \
  }

#define SYNAPSE_RETURN_IF_ERROR(error_carrier_for_eval) \
  do {                                                  \
    auto&& error_carrier{error_carrier_for_eval};       \
    if (ABSL_PREDICT_FALSE(!ok(error_carrier))) {       \
      return get_error(error_carrier);                  \
    }                                                   \
  } while (false)

#define SYNAPSE_RETURN_IF_ERROR_V(error_variant_for_eval)                            \
  do {                                                                               \
    auto&& error_variant{error_variant_for_eval};                                    \
    if (ABSL_PREDICT_FALSE(absl::holds_alternative<synapse_error>(error_variant))) { \
      return absl::get<synapse_error>(error_variant);                                \
    }                                                                                \
  } while (false)

#define SYNAPSE_RETURN_IF_ERROR_O(error_optional_for_eval) \
  do {                                                     \
    auto&& error_optional{error_optional_for_eval};        \
    if (ABSL_PREDICT_FALSE(error_optional.has_value())) {  \
      return std::move(error_optional);                    \
    }                                                      \
  } while (false)

#define SYNAPSE_RETURN_IF_ERROR_O_TO_V(error_optional_for_eval) \
  do {                                                          \
    auto&& error_optional{error_optional_for_eval};             \
    if (ABSL_PREDICT_FALSE(error_optional.has_value())) {       \
      return error_optional.value();                            \
    }                                                           \
  } while (false)

#define OP_REQUIRES_SYNAPSE(CTX, error_carrier_for_eval)                                      \
  do {                                                                                        \
    auto&& error_carrier{error_carrier_for_eval};                                             \
    if (TF_PREDICT_FALSE(!ok(error_carrier))) {                                               \
      CheckNotInComputeAsync((CTX), "OP_REQUIRES_ASYNC");                                     \
      auto& error = get_error(error_carrier);                                                 \
      (CTX)->CtxFailure(__FILE__, __LINE__, errors::Aborted(error.error, " ", error.status)); \
      return;                                                                                 \
    }                                                                                         \
  } while (false)

#define OP_REQUIRES_SYNAPSE_ASYNC(CTX, error_variant_for_eval, CALLBACK)                      \
  do {                                                                                        \
    auto&& error_carrier{error_carrier_for_eval};                                             \
    if (TF_PREDICT_FALSE(!ok(error_carrier))) {                                               \
      auto& error = get_error(error_carrier);                                                 \
      (CTX)->CtxFailure(__FILE__, __LINE__, errors::Aborted(error.error, " ", error.status)); \
      CALLBACK();                                                                             \
      return;                                                                                 \
    }                                                                                         \
  }

#define OP_REQUIRES_SYNAPSE_V(CTX, error_variant_for_eval)                                          \
  do {                                                                                              \
    auto&& error_variant{error_variant_for_eval};                                                   \
    if (TF_PREDICT_FALSE(absl::holds_alternative<synapse_helpers::synapse_error>(error_variant))) { \
      CheckNotInComputeAsync((CTX), "OP_REQUIRES_ASYNC");                                           \
      auto& error = absl::get<synapse_helpers::synapse_error>(error_variant);                       \
      (CTX)->CtxFailure(__FILE__, __LINE__, errors::Aborted(error.error, " ", error.status));       \
      return;                                                                                       \
    }                                                                                               \
  } while (false)

#define OP_REQUIRES_SYNAPSE_V_ASYNC(CTX, error_variant_for_eval, CALLBACK)                          \
  do {                                                                                              \
    auto&& error_variant{error_variant_for_eval};                                                   \
    if (TF_PREDICT_FALSE(absl::holds_alternative<synapse_helpers::synapse_error>(error_variant))) { \
      auto& error = absl::get<synapse_helpers::synapse_error>(error_variant);                       \
      (CTX)->CtxFailure(__FILE__, __LINE__, errors::Aborted(error.error, " ", error.status));       \
      CALLBACK();                                                                                   \
      return;                                                                                       \
    }                                                                                               \
  } while (false)

#define TF_RETURN_IF_SYNAPSE_ERROR_V(error_variant_for_eval)                                        \
  do {                                                                                              \
    auto&& error_variant{error_variant_for_eval};                                                   \
    if (TF_PREDICT_FALSE(absl::holds_alternative<synapse_helpers::synapse_error>(error_variant))) { \
      auto& error = absl::get<synapse_helpers::synapse_error>(error_variant);                       \
      return errors::Aborted(error.error, " ", error.status);                                       \
    }                                                                                               \
  } while (false)

#define OP_REQUIRES_SYNAPSE_O(CTX, error_optional_for_eval)                                   \
  do {                                                                                        \
    auto&& error_optional{error_optional_for_eval};                                           \
    if (TF_PREDICT_FALSE(error_optional.has_value())) {                                       \
      CheckNotInComputeAsync((CTX), "OP_REQUIRES_ASYNC");                                     \
      auto& error = error_optional.value();                                                   \
      (CTX)->CtxFailure(__FILE__, __LINE__, errors::Aborted(error.error, " ", error.status)); \
      return;                                                                                 \
    }                                                                                         \
  } while (false)

#define OP_REQUIRES_SYNAPSE_O_ASYNC(CTX, error_optional_for_eval, CALLBACK)                   \
  do {                                                                                        \
    auto&& error_optional{error_optional_for_eval};                                           \
    if (TF_PREDICT_FALSE(error_optional.has_value())) {                                       \
      auto& error = error_optional.value();                                                   \
      (CTX)->CtxFailure(__FILE__, __LINE__, errors::Aborted(error.error, " ", error.status)); \
      CALLBACK();                                                                             \
      return;                                                                                 \
    }                                                                                         \
  } while (false)

#define OP_RETURN_IF_FALSE(v)       \
  do {                              \
    if (ABSL_PREDICT_FALSE(!(v))) { \
      return;                       \
    }                               \
  } while (false)

#define TF_RETURN_IF_SYNAPSE_ERROR_O(...)                     \
  do {                                                        \
    auto&& error_optional{__VA_ARGS__};                       \
    if (ABSL_PREDICT_FALSE(error_optional.has_value())) {     \
      auto& error = error_optional.value();                   \
      return errors::Aborted(error.error, " ", error.status); \
    }                                                         \
  } while (false)

#define OP_REQUIRES_WITH_RETVAL(CTX, RETVAL, EXP, STATUS) \
  do {                                                    \
    if (!TF_PREDICT_TRUE(EXP)) {                          \
      CheckNotInComputeAsync((CTX), "OP_REQUIRES_ASYNC"); \
      (CTX)->CtxFailure(__FILE__, __LINE__, (STATUS));    \
      return RETVAL;                                      \
    }                                                     \
  } while (false)

#define OP_REQUIRES_OK_WITH_RETVAL(CTX, RETVAL, ...)         \
  do {                                                       \
    ::tensorflow::Status _s(__VA_ARGS__);                    \
    if (!TF_PREDICT_TRUE(_s.ok())) {                         \
      CheckNotInComputeAsync((CTX), "OP_REQUIRES_OK_ASYNC"); \
      (CTX)->CtxFailureWithWarning(__FILE__, __LINE__, _s);  \
      return RETVAL;                                         \
    }                                                        \
  } while (false)
