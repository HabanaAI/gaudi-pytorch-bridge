/******************************************************************************
 * Copyright (C) 2022 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */

#include "collective_kernel_info.h"
#include "tensor_info.h"

namespace habana_helpers {

size_t collective_kernel_info::Size() const {
  size_t size = sizeof(*this);
  size += input_tensor_infos.size() *
      sizeof(decltype(input_tensor_infos)::value_type);
  size += output_tensor_infos.size() *
      sizeof(decltype(output_tensor_infos)::value_type);
  return size;
}

} // namespace habana_helpers
