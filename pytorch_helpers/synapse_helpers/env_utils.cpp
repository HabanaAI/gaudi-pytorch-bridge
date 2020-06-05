/******************************************************************************
 * Copyright (C) 2020 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */
#include <absl/strings/match.h>
#include <cstdlib>
#include <cstring>
#include <ostream>
#include "synapse_helpers/logging.h"

namespace synapse_helpers {

bool is_env_var_equal(const char* variable_name, const char* value, bool unset_default) {
  auto* env_value = std::getenv(variable_name);
  if (env_value == nullptr) return unset_default;
  return std::strcmp(env_value, value) == 0;
}

bool get_bool_env_var(const char* variable_name, bool unset_default) {
  auto* env_value = std::getenv(variable_name);
  if (env_value == nullptr) return unset_default;

  bool true_found = absl::EqualsIgnoreCase(env_value, "1") || absl::EqualsIgnoreCase(env_value, "true");
  bool false_found = absl::EqualsIgnoreCase(env_value, "0") || absl::EqualsIgnoreCase(env_value, "false");

  if (true_found)
    return true;
  else if (false_found)
    return false;
  else {
    LOG_(FATAL) << variable_name << " contains unexpected value.";
    return unset_default;
  }
}

}  // namespace synapse_helpers