/*******************************************************************************
 * Copyright (C) 2024 Habana Labs, Ltd. an Intel Company
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
#include "hpu_ops/unique_dim.h"
#include "habana_eager/ops/eager_op.h"
#include "habana_eager/ops/view.h"

namespace habana {
namespace eager {

std::tuple<at::Tensor, at::Tensor, at::Tensor> unique_dim_eager(
    const at::Tensor& self,
    int64_t dim,
    bool sorted,
    bool return_inverse,
    bool return_counts) {
  dim = at::maybe_wrap_dim(dim, self.dim());
  int elements = self.numel();
  at::Tensor inverse_tensor;
  at::Tensor counts_tensor;
  at::Tensor result;
  auto inputShape = self.sizes().vec();
  auto output_shape = self.sizes().vec();

  if (elements == 0) {
    auto feature_map =
        at::empty(output_shape, self.options(), self.suggest_memory_format());
    inverse_tensor = at::empty(
        at::Tensor{}.sizes().vec(),
        self.options().dtype(c10::ScalarType::Long),
        self.suggest_memory_format());
    counts_tensor = at::empty(
        at::Tensor{}.sizes().vec(),
        self.options().dtype(c10::ScalarType::Long),
        self.suggest_memory_format());
    return std::make_tuple(feature_map, inverse_tensor, counts_tensor);
  }
  auto param_shape = std::vector<int64_t>{output_shape.at(dim)};
  std::vector<int64_t> valid_count_shape{1};

  auto dim_in = self.dim() - 1 - dim;
  if (return_inverse && return_counts) {
    auto hpu_op = habana::eager::EagerOp<
        std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>>{
        "hpu::unique_dim_eager",
        {self, dim_in, sorted, return_inverse, return_counts},
        {output_shape, valid_count_shape, param_shape, param_shape},
        0};

    hpu_op.SetOutputMetaFn(UniqueDimMeta);
    auto result_unique = hpu_op.call();
    auto feature_map = std::get<0>(result_unique);
    auto valid_count = std::get<1>(result_unique);
    auto end = valid_count.item<int64_t>();
    result = at::slice(feature_map, dim, 0, end, 1);

    inverse_tensor = std::get<2>(result_unique);

    counts_tensor = std::get<3>(result_unique);
    counts_tensor = at::slice(counts_tensor, 0, 0, end, 1);
  } else if (return_inverse != return_counts) {
    auto hpu_op =
        habana::eager::EagerOp<std::tuple<at::Tensor, at::Tensor, at::Tensor>>{
            "hpu::unique_dim_eager",
            {self, dim_in, sorted, return_inverse, return_counts},
            {output_shape, valid_count_shape, param_shape},
            0};

    hpu_op.SetOutputMetaFn(UniqueDimMeta);
    auto result_unique = hpu_op.call();
    auto feature_map = std::get<0>(result_unique);
    auto valid_count = std::get<1>(result_unique);
    auto end = valid_count.item<int64_t>();
    result = at::slice(feature_map, dim, 0, end, 1);

    if (return_inverse) {
      inverse_tensor = std::get<2>(result_unique);
    } else if (return_counts) {
      counts_tensor = std::get<2>(result_unique);
      counts_tensor = at::slice(counts_tensor, 0, 0, end, 1);
    }
  } else {
    auto hpu_op = habana::eager::EagerOp<std::tuple<at::Tensor, at::Tensor>>{
        "hpu::unique_dim_eager",
        {self, dim_in, sorted, return_inverse, return_counts},
        {output_shape, valid_count_shape},
        0};

    hpu_op.SetOutputMetaFn(UniqueDimMeta);
    auto result_unique = hpu_op.call();
    auto feature_map = std::get<0>(result_unique);
    auto valid_count = std::get<1>(result_unique);
    auto end = valid_count.item<int64_t>();
    result = at::slice(feature_map, dim, 0, end, 1);
  }
  return std::make_tuple(result, inverse_tensor, counts_tensor);
}

TORCH_LIBRARY_FRAGMENT(hpu, m) {
  m.def(
      "unique_dim_eager(Tensor self, int dim, bool sorted, bool return_inverse, bool return_counts) -> (Tensor, Tensor, Tensor)");
}
} // namespace eager
} // namespace habana
