/**
* Copyright (c) 2021-2024 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/

#pragma once

#include <memory>
#include <vector>

#include "habana_kernels/hccl_kernels.h"

class PtTensorInfo;
using PtTensorInfoShared = std::shared_ptr<PtTensorInfo>;

namespace habana {
class CollectiveOperator;
}

namespace habana_helpers {

class CollectiveKernelInfos {
 public:
  struct Info {
    std::vector<PtTensorInfoShared> input_tensor_infos;
    std::vector<PtTensorInfoShared> output_tensor_infos;
    std::shared_ptr<habana::CollectiveOperator> kernel;

    size_t Size() const;
  };

  CollectiveKernelInfos() = default;

  CollectiveKernelInfos(
      std::istream& is,
      const std::vector<PtTensorInfoShared>& dtensorinfos) {
    Deserialize(is, dtensorinfos);
  }

  CollectiveKernelInfos(const CollectiveKernelInfos&) = default;
  CollectiveKernelInfos& operator=(const CollectiveKernelInfos&) = default;
  CollectiveKernelInfos(CollectiveKernelInfos&&) = default;
  CollectiveKernelInfos& operator=(CollectiveKernelInfos&&) = default;

  void Serialize(
      std::ostream& os,
      const std::vector<PtTensorInfoShared>& dtensorinfos) const;
  void Deserialize(
      std::istream& is,
      const std::vector<PtTensorInfoShared>& dtensorinfos);
  void Launch(bool async, synapse_helpers::event_done_callback cleanup_callback)
      const;
  void ClearAllPtAndSynTensors();

  void AddKernel(Info&& info) {
    infos_.emplace_back(std::move(info));
  }

  bool Empty() const {
    return infos_.empty();
  };

  size_t Size() const {
    size_t size = 0;
    for (const auto& kernel_info : infos_) {
      size += kernel_info.Size();
    }
    return size;
  }
  void Clear() {
    infos_.clear();
  }

 private:
  std::vector<Info> infos_;
};

} // namespace habana_helpers
