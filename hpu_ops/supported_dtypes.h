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
