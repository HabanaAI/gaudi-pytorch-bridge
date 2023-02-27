/******************************************************************************
 * Copyright (C) 2021 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */

#include "generated/backend/vdot.h"

namespace habana {
sizes_vec VdotOutputShape(const at::Stack&) {
  return {{}};
}
} // namespace habana
