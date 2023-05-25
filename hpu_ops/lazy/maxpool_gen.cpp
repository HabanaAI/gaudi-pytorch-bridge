/******************************************************************************
 * Copyright (C) 2022-2023 Habana Labs, Ltd. an Intel Company
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
#include "generated/eager/max_pool3d_with_indices.h"
#include "generated/lazy/max_pool2d_with_indices.h"
#include "generated/lazy/max_pool2d_with_indices_backward.h"

namespace habana {

int ComputeMaxpool3dOutputDim(
    int input_shape,
    int kernel,
    int stride,
    int padding,
    int dilation,
    bool ceilMode) {
  return (
      ((input_shape + 2 * padding - dilation * (kernel - 1) - 1 +
        (ceilMode ? stride - 1 : 0)) /
       stride) +
      1);
}

sizes_vec MaxPool3DIndicesOutputShape(const at::Stack& stack) {
  std::vector<long int> pad = {0, 0, 0};
  std::vector<long int> dil = {1, 1, 1};
  auto self = stack.at(0).toTensor();
  auto kernel = stack.at(1).toIntVector();
  auto stride = stack.at(2).toIntVector().size() == 0
      ? kernel
      : stack.at(2).toIntVector();
  auto padding =
      stack.at(3).toIntVector().size() == 0 ? pad : stack.at(3).toIntVector();
  auto dilation =
      stack.at(4).toIntVector().size() == 0 ? dil : stack.at(4).toIntVector();
  const bool ceil_mode = stack.at(5).toBool();

  TORCH_CHECK(
      self.dim() == 5 || self.dim() == 4,
      "Maxpool3d expects Input size must be 5 or 4, but got ",
      self.dim());
  TORCH_CHECK(
      padding.size() == 3,
      "Maxpool3d expects padding size is 3 but got ",
      padding.size());
  TORCH_CHECK(
      kernel.size() == 3,
      "Maxpool3d expects kernel size is 3 but got ",
      kernel.size());
  TORCH_CHECK(
      stride.size() == 3,
      "Maxpool3d expects stride size is 3 but got ",
      stride.size());
  TORCH_CHECK(
      dilation.size() == 3,
      "Maxpool3d expects dilation size is 3 but got ",
      dilation.size());

  std::vector<int64_t> input_shape = self.sizes().vec();
  std::vector<int64_t> output_shape = self.sizes().vec();

  int n = kernel.size();
  int output_shape_index = output_shape.size() - n;
  // updating the width, height, & depth dimension
  for (int i = 0; i < n; i++) {
    output_shape.at(output_shape_index) = ComputeMaxpool3dOutputDim(
        input_shape.at(output_shape_index),
        kernel[i],
        stride[i],
        padding[i],
        dilation[i],
        ceil_mode);
    output_shape_index++;
  }

  // ensure that the last pooling starts inside the image
  // needed to avoid problems in ceil mode
  if (ceil_mode) {
    int index_ceil_mode = output_shape.size() - n;
    for (int i = 0; i < n; i++) {
      if ((output_shape.at(index_ceil_mode) - 1) * stride[i] >=
          input_shape.at(index_ceil_mode) + padding[i])
        --output_shape.at(index_ceil_mode);
    }
    index_ceil_mode++;
  }
  return {output_shape, output_shape};
}

OutputMetaDataVector Maxpool3dWithIndicesMeta(const at::Stack& stack) {
  const auto& self = stack.at(0).toTensor();
  const auto& output_shape = MaxPool3DIndicesOutputShape(stack);
  OutputMetaDataVector meta;
  meta.resize(output_shape.size());

  meta[0].shape = output_shape[0];
  meta[0].dtype = self.scalar_type();

  meta[1].shape = output_shape[0];
  meta[1].dtype = c10::ScalarType::Long;

  return meta;
}

template <>
LazyMaxPool<std::tuple<at::Tensor, at::Tensor>>::LazyMaxPool(
    const std::string& qualstring,
    const std::vector<at::IValue>& inputs,
    const std::function<sizes_vec(const at::Stack&)>& out_shapes_fn)
    : habana_lazy::LazyOp<std::tuple<at::Tensor, at::Tensor>>(
          qualstring,
          inputs,
          out_shapes_fn,
          -1) {}

template <>
std::tuple<at::Tensor, at::Tensor> LazyMaxPool<
    std::tuple<at::Tensor, at::Tensor>>::get_result_overrideable() {
  auto inputs = get_inputs();
  auto t = inputs.at(0).toTensor();
  auto out_shape = get_out_shapes()[0];
  at::Tensor maxpool = habana_lazy::empty_hpu_lazy(
      out_shape, t.options(), t.suggest_memory_format(), false);
  // TODO Analyse memory performance for the dtype change
  // (https://jira.habana-labs.com/browse/SW-108396),
  at::Tensor indices = habana_lazy::empty_hpu_lazy(
      out_shape,
      t.options().dtype(c10::ScalarType::Long),
      t.suggest_memory_format(),
      false);
  return {maxpool, indices};
}

} // namespace habana
