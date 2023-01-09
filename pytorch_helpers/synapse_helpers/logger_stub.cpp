/******************************************************************************
 * Copyright (C) 2020 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */
#include "absl/strings/string_view.h"

#include "synapse_helpers/util.h"

namespace synapse_logger {
enum class data_dump_category : unsigned {};
bool logger_is_enabled([[maybe_unused]] data_dump_category cat) {
  return false;
}

void log(absl::string_view payload) {
  (void)(payload);
}
} // namespace synapse_logger
