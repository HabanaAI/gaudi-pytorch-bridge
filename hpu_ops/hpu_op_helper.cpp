/******************************************************************************
 * Copyright (C) 2021 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */
#include "hpu_op_helper.h"

namespace habana {

std::string to_string(const at::IValue& ival) {
  std::ostringstream oss;
  auto print_tensor_meta = [](const at::Tensor& t) {
    std::ostringstream oss_;
    if (t.defined()) {
      oss_ << t.toString() << t.sizes();
    } else {
      oss_ << "[UndefinedTensor]";
    }
    return oss_.str();
  };
  if (ival.isTensor()) {
    oss << print_tensor_meta(ival.toTensor());
  } else if (ival.isScalar()) {
    const auto& s = ival.toScalar();
    if (s.isFloatingPoint()) {
      oss << s.toDouble();
    } else if (s.isIntegral(false)) {
      oss << s.toLong();
    } else if (s.isBoolean()) {
      oss << s.toBool();
    }
  } else if (ival.isString()) {
    oss << ival.toStringRef();
  } else if (ival.isList()) {
    if (ival.isTensorList()) {
      bool comma = true;
      for (const auto& t : ival.toTensorList()) {
        if (comma) {
          comma = false;
        } else {
          oss << ", ";
        }
        oss << print_tensor_meta(t);
      }
    } else if (ival.isIntList()) {
      oss << at::IntArrayRef(ival.toIntVector());
    } else if (ival.isDoubleList()) {
      oss << ival.toDoubleVector();
    }
  } else if (ival.isNone()) {
    oss << "None";
  }
  return oss.str();
}

std::vector<at::Tensor> GetMetaTensorList(
    const std::vector<at::Tensor>& tensors) {
  std::vector<at::Tensor> metatensors;
  metatensors.reserve(tensors.size());
  for (const auto& tensor : tensors) {
    metatensors.emplace_back(at::empty(
        tensor.sizes(),
        tensor.options().device(at::kMeta),
        tensor.suggest_memory_format()));
  }
  return metatensors;
}

std::vector<c10::optional<at::Tensor>> GetMetaOptTensorList(
    const std::vector<c10::optional<at::Tensor>>& tensors) {
  std::vector<c10::optional<at::Tensor>> metatensors;
  metatensors.reserve(tensors.size());
  for (const auto& tensor : tensors) {
    if (tensor.has_value()) {
      const auto& tv = tensor.value();
      metatensors.emplace_back(at::empty(
          tv.sizes(),
          tv.options().device(at::kMeta),
          tv.suggest_memory_format()));
    } else {
      metatensors.emplace_back(tensor);
    }
  }
  return metatensors;
}

} // namespace habana
