/******************************************************************************
 * Copyright (C) 2022 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */

#pragma once

#include <memory>
#include <vector>

class PtTensorInfo;
using PtTensorInfoShared = std::shared_ptr<PtTensorInfo>;

namespace habana {
class HabanaOperator;
}

namespace habana_helpers {

struct collective_kernel_info {
  std::vector<PtTensorInfoShared> input_tensor_infos;
  std::vector<PtTensorInfoShared> output_tensor_infos;
  std::shared_ptr<habana::HabanaOperator> kernel;
};

} // namespace habana_helpers
