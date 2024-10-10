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
#include "absl/strings/string_view.h"

#include "backend/synapse_helpers/util.h"

namespace synapse_logger {
enum class data_dump_category : unsigned {};
bool logger_is_enabled([[maybe_unused]] data_dump_category cat) {
  return false;
}

void log(absl::string_view payload) {
  (void)(payload);
}
} // namespace synapse_logger
