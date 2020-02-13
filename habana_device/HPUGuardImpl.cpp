/******************************************************************************
 * Copyright (C) 2020 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */
#include "HPUGuardImpl.h"

namespace at {
namespace detail {

C10_REGISTER_GUARD_IMPL(HABANA, HABANAGuardImpl);

} // namespace detail
} // namespace at
