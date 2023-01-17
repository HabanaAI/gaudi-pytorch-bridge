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
class CollectiveOperator;
}

namespace habana_helpers {

struct collective_kernel_info {
  std::vector<PtTensorInfoShared> input_tensor_infos;
  std::vector<PtTensorInfoShared> output_tensor_infos;
  std::shared_ptr<habana::CollectiveOperator> kernel;

  size_t Size() const;
};

} // namespace habana_helpers
