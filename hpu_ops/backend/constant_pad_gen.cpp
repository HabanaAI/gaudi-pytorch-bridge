/******************************************************************************
 * Copyright (C) 2021-2023 Habana Labs, Ltd. an Intel Company
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

#include "generated/backend/constant_pad_nd.h"

namespace habana {

std::vector<int64_t> pad_output_shape(
    const at::Tensor& self,
    c10::IntArrayRef pad) {
  auto ndim = self.dim();
  auto lpad = pad.size() / 2;

  TORCH_CHECK(
      pad.size() % 2 == 0,
      "Length of pad must be even but instead it equals ",
      pad.size());

  TORCH_CHECK(
      ndim >= (int64_t)lpad,
      "Length of pad should be no more than twice the number of "
      "dimensions of the input. Pad length is ",
      pad.size(),
      "while the input has ",
      ndim,
      "dimensions.");

  auto shape = self.sizes().vec();

  for (unsigned int i = 0; i < lpad; i++) {
    auto pad_start = pad[2 * i];
    auto pad_end = pad[2 * i + 1];
    shape[ndim - i - 1] += (pad_start + pad_end);
    TORCH_CHECK(
        shape[ndim - i - 1] > 0,
        "The input size ",
        self.sizes()[i],
        ", plus negative padding ",
        pad_start,
        " and ",
        pad_end,
        " resulted in a invalid output size, "
        "Check dimension ",
        i,
        " of your input.");
  }
  return shape;
}

OutputMetaDataVector ConstantPadMeta(const at::Stack& stack) {
  const auto& self = stack.at(0).toTensor();
  OutputMetaData meta;
  if ((stack.size() == 4) && (!stack.at(1).isTensor())) {
    auto pad = stack.at(1).toIntVector();
    auto ndim = self.dim();
    auto lpad = pad.size() / 2;
    auto shape = stack.at(3).toIntVector();

    for (unsigned int i = 0; i < lpad; i++) {
      auto pad_start = pad[2 * i];
      auto pad_end = pad[2 * i + 1];
      shape[ndim - i - 1] += (pad_start + pad_end);
    }
    meta.shape = shape;
  } else if ((stack.size() == 4) && (stack.at(1).isTensor())) {
    meta.shape = stack[2].toTensor().sizes().vec();
  } else {
    auto pad = stack.at(1).toIntVector();
    meta.shape = pad_output_shape(self, pad);
  }
  meta.mem_format = self.suggest_memory_format();
  meta.dtype = self.scalar_type();
  return {meta};
}

std::shared_ptr<void> FillConstantPadParams(
    const at::Stack& stack,
    size_t& size) {
  auto self = stack.at(0).toTensor();
  PARAMS_STUB(ns_PadKernelEx::Params);
  params->mode = PadMode_t::PAD_MODE_CONSTANT;

  memset(params->pads, 0, sizeof(params->pads));

  if ((stack.size() == 4) && (stack.at(1).isTensor())) {
    if (c10::isIntegralType(self.scalar_type(), false)) {
      params->value.i = stack.at(3).toScalar().to<decltype(params->value.i)>();
    } else {
      params->value.f = stack.at(3).toScalar().to<float>();
    }
    return params;
  } else if ((stack.size() == 4) && (!stack.at(1).isTensor())) {
    if (c10::isIntegralType(self.scalar_type(), false)) {
      params->value.i = stack.at(2).toScalar().to<decltype(params->value.i)>();
    } else {
      params->value.f = stack.at(2).toScalar().to<float>();
    }
    return params;
  } else {
    if (c10::isIntegralType(self.scalar_type(), false)) {
      params->value.i = stack.at(2).toScalar().to<decltype(params->value.i)>();
    } else {
      params->value.f = stack.at(2).toScalar().to<float>();
    }
    auto pad = stack.at(1).toIntVector();

    auto ndim = self.dim();
    auto lpad = pad.size() / 2;

    for (unsigned int i = 0; i < lpad; i++) {
      params->pads[i] = pad[2 * i];
      params->pads[i + ndim] = pad[2 * i + 1];
    }

    return params;
  }
}

void ConstantPad::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  auto meta = ConstantPadMeta(stack)[0];
  auto self = stack[0].toTensor();
  size_t size = 0;
  const auto& param = FillConstantPadParams(stack, size);
  if ((stack.size() == 3) ||
      ((stack.size() == 4) && (!stack.at(1).isTensor()))) {
    auto op = BuildOp(
        graph,
        guid_,
        {syn_in(0)},
        {{meta.shape, meta.dtype, 0}},
        param.get(),
        size);
    syn_out(0) = std::move(op.at(0));
  } else {
    at::Tensor host_tensor = stack[1].toTensor();
    auto tmeta{get_tensor_extra_meta(host_tensor)};
    auto output_shape = stack[2].toTensor().sizes().vec();
    auto input_shape = stack[0].toTensor().sizes().vec();
    if (habana::ShapeInference::GetCurrentPass() ==
        habana::ShapeInfo::InferencePass::MIN_SHAPE) {
      auto ndim = self.dim();
      auto in_data = self.sizes().vec();
      std::vector<uint32_t> data(MAX_DIMENSIONS_NUM * 2, 0);
      for (unsigned int i = 0; i < ndim; i++) {
        // order of dims is reversed in H2D tensor
        data[ndim - i - 1] = output_shape[i] - input_shape[i];
      }
      tmeta->set_min<uint32_t>(data);
    } else if (
        habana::ShapeInference::GetCurrentPass() ==
        habana::ShapeInfo::InferencePass::MAX_SHAPE) {
      auto ndim = self.dim();
      auto in_data = self.sizes().vec();
      std::vector<uint32_t> data(MAX_DIMENSIONS_NUM * 2, 0);
      for (unsigned int i = 0; i < ndim; i++) {
        // order of dims is reversed in H2D tensor
        data[ndim - i - 1] = output_shape[i] - input_shape[i];
      }
      tmeta->set_max<uint32_t>(data);
    }
    auto op = BuildOp(
        graph,
        guid_,
        {syn_in(0), syn_in(1)},
        {{meta.shape, meta.dtype, 0}},
        param.get(),
        size);
    syn_out(0) = std::move(op.at(0));
  }
}

struct ConstantPadDS : OpBackend {
  ConstantPadDS(int device_id, c10::ScalarType scalar_type);
  void AddNode(synapse_helpers::graph&, const at::Stack&) override;
};

ConstantPadDS::ConstantPadDS(int device_id, c10::ScalarType scalar_type)
    : OpBackend(device_id, "pad_fwd", scalar_type, {0}, {}, {}, false) {
  SetOutputMetaFn(ConstantPadMeta);
}

void ConstantPadDS::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  auto meta = ConstantPadMeta(stack)[0];
  auto self = stack[0].toTensor();
  size_t size = 0;
  const auto& param = FillConstantPadParams(stack, size);
  if ((stack.size() == 3) ||
      ((stack.size() == 4) && (!stack.at(1).isTensor()))) {
    auto op = BuildOp(
        graph,
        guid_,
        {syn_in(0)},
        {{meta.shape, meta.dtype, 0}},
        param.get(),
        size);
    syn_out(0) = std::move(op.at(0));
  } else {
    at::Tensor host_tensor = stack[1].toTensor();
    auto tmeta{get_tensor_extra_meta(host_tensor)};
    auto output_shape = stack[2].toTensor().sizes().vec();
    auto input_shape = stack[0].toTensor().sizes().vec();
    if (habana::ShapeInference::GetCurrentPass() ==
        habana::ShapeInfo::InferencePass::MIN_SHAPE) {
      auto ndim = self.dim();
      auto in_data = self.sizes().vec();
      std::vector<uint32_t> data(MAX_DIMENSIONS_NUM * 2, 0);
      for (unsigned int i = 0; i < ndim; i++) {
        // order of dims is reversed in H2D tensor
        data[ndim - i - 1] = output_shape[i] - input_shape[i];
      }
      tmeta->set_min<uint32_t>(data);
    } else if (
        habana::ShapeInference::GetCurrentPass() ==
        habana::ShapeInfo::InferencePass::MAX_SHAPE) {
      auto ndim = self.dim();
      auto in_data = self.sizes().vec();
      std::vector<uint32_t> data(MAX_DIMENSIONS_NUM * 2, 0);
      for (unsigned int i = 0; i < ndim; i++) {
        // order of dims is reversed in H2D tensor
        data[ndim - i - 1] = output_shape[i] - input_shape[i];
      }
      tmeta->set_max<uint32_t>(data);
    }
    auto op = BuildOp(
        graph,
        guid_,
        {syn_in(0), syn_in(1)},
        {{meta.shape, meta.dtype, 0}},
        param.get(),
        size);
    syn_out(0) = std::move(op.at(0));
  }
}

} // namespace habana

static const auto& HabanaRandomKernelRegistry =
    habana::KernelRegistry()
        .add("hpu::constant_pad_nd", KERNEL_FN_GLOBAL(habana::ConstantPadDS))
        .add(
            "hpu::constant_pad_nd_ds",
            KERNEL_FN_GLOBAL(habana::ConstantPadDS));