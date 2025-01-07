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
#include <torch/torch.h>
#include <unordered_set>

namespace habana {
class SupportedDtypes {
 public:
  SupportedDtypes(std::unordered_map<int, std::unordered_set<at::ScalarType>>
                      per_gen_dtypes);
  bool count(at::ScalarType type) const;
  bool count(const at::Tensor& tensor) const;
  bool count(const at::optional<at::Tensor>& tensor) const;

 private:
  std::unordered_set<at::ScalarType> m_dtypes;
};
} // namespace habana
