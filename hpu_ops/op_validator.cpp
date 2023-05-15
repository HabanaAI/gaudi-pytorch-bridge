/*******************************************************************************
 * Copyright (C) 2023 Habana Labs, Ltd. an Intel Company
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
#include "op_validator.h"
#include <syn_sl_api.h>
#include <unistd.h>
#include <sstream>
#include <string>
#include "backend/habana_device/hpu_cached_devices.h"
#include "habana_kernels/index_kernels.h"
#include "habana_kernels/lazy_kernels.h"
#include "habana_kernels/random_gen_kernels.h"
#include "op_backend.h"
namespace habana {

namespace {

struct SharedLayerInitialization {
  SharedLayerInitialization() {
    static auto status = synSharedLayerInit();
    TORCH_CHECK(status == synSuccess, "cannot initialize shared layer");
  }

  ~SharedLayerInitialization() {
    synSharedLayerFinit();
  }
};

SharedLayerInitialization _slu_initializer;

gcapi::DeviceId_t synDeviceTypeToGcApiDeviceType(synDeviceType tp) {
  switch (tp) {
    case synDeviceGaudi:
      return gcapi::DEVICE_ID_GAUDI;
    case synDeviceGaudi2:
      return gcapi::DEVICE_ID_GAUDI2;
    case synDeviceGaudi3:
      return gcapi::DEVICE_ID_GAUDI3;
    default:
      break;
  }

  TORCH_CHECK(false, "unsupported synDeviceType for shared layer");
}

gcapi::DeviceId_t _getDeviceType() {
  auto deviceType = synapse_helpers::HPURegistrar::get_device(0).type();
  auto deviceId = synDeviceTypeToGcApiDeviceType(deviceType);
  return deviceId;
}

gcapi::DeviceId_t getDeviceType() {
  static auto deviceId = _getDeviceType();
  return deviceId;
}

synDeviceId getDeviceId() {
  return synapse_helpers::HPURegistrar::get_device().id();
}

bool fillGcApiTypeFromScalarType(
    gcapi::TensorDataType_t& dst,
    at::ScalarType t) {
  switch (t) {
    case at::ScalarType::Bool:
      dst = gcapi::DATA_I8;
      return true;
    case at::ScalarType::Byte:
      dst = gcapi::DATA_U8;
      return true;
    case at::ScalarType::Char:
      dst = gcapi::DATA_I8;
      return true;
    case at::ScalarType::Int:
    case at::ScalarType::Long:
      dst = gcapi::DATA_I32;
      return true;
    case at::ScalarType::Short:
      dst = gcapi::DATA_I16;
      return true;
    case at::ScalarType::Float:
    case at::ScalarType::Double:
      dst = gcapi::DATA_F32;
      return true;
    case at::ScalarType::BFloat16:
      dst = gcapi::DATA_BF16;
      return true;
    default:
      dst = gcapi::NUM_DATATYPES;
      return false;
  }
}

bool fillGuidParamInfoWithIntList(
    shared_layer::guid_param_info& param,
    const std::vector<int64_t>& xs) {
  param.geometry.dims = xs.size() - 1;
  for (uint64_t dim = 0; dim < xs.size() - 1; ++dim) {
    int64_t syn_dim = xs.size() - dim - 2;
    param.layout[syn_dim] = xs[dim];
  }
  if (param.geometry.dims == 0) {
    param.geometry.dims = 1;
    param.layout[0] = 1;
  }
  if (not fillGcApiTypeFromScalarType(
          param.dataType, (at::ScalarType)xs.back()))
    return false;

  return true;
}

bool fillGuidParamInfoWithTensor(
    shared_layer::guid_param_info& param,
    const at::Tensor& t) {
  param.geometry.dims = t.dim();
  for (int64_t dim = 0; dim < t.dim(); ++dim) {
    int64_t syn_dim = t.dim() - dim - 1;
    param.layout[syn_dim] = t.size(dim);
  }
  if (param.geometry.dims == 0) {
    param.geometry.dims = 1;
    param.layout[0] = 1;
  }
  if (not fillGcApiTypeFromScalarType(param.dataType, t.scalar_type()))
    return false;
  return true;
}

/*
 * This function is a wrapper for shared layer query interface.
 */
bool ValidateGuid(
    const std::string& guid,
    const detail::TensorDescrArray& input_values,
    const detail::TensorDescrArray& output_values,
    void* filledParams = nullptr) {
  synSharedLayerParams_t params{};
  params.apiVersion = 1;
  strncpy(params.nodeName, guid.c_str(), gcapi::MAX_NODE_NAME);
  params.nodeName[gcapi::MAX_NODE_NAME - 1] = 0;
  params.nodeParams = filledParams;

  for (auto i = 0u; i < input_values.size(); ++i) {
    bool result = false;
    if (input_values[i].isTensor()) {
      result = fillGuidParamInfoWithTensor(
          params.inputTensors[i], *input_values[i].m_tensor);
    } else {
      result = fillGuidParamInfoWithIntList(
          params.inputTensors[i], input_values[i].m_dims_and_type);
    }

    if (not result)
      return false;
  }
  params.inputTensorNr = input_values.size();

  for (auto i = 0u; i < output_values.size(); ++i) {
    bool result = false;
    if (output_values[i].isTensor()) {
      result = fillGuidParamInfoWithTensor(
          params.outputTensors[i], *output_values[i].m_tensor);
    } else {
      result = fillGuidParamInfoWithIntList(
          params.outputTensors[i], output_values[i].m_dims_and_type);
    }

    if (not result)
      return false;
  }
  params.outputTensorNr = output_values.size();
  auto device = getDeviceType();
  synStatus status;
  status = synSharedLayerValidateGuid(guid.c_str(), &params, device);
  return status == synSuccess;
}

detail::TensorDescr TryCastTensor(
    const at::IValue& val,
    at::ScalarType targetType) {
  if (val.isTensor()) {
    const auto& t = val.toTensor();
    if (targetType == at::ScalarType::Undefined) {
      return detail::TensorDescr(&t);
    }
    if (targetType == t.scalar_type()) {
      return detail::TensorDescr(&t);
    }

    std::vector<std::int64_t> description;
    description.reserve(t.dim());
    for (std::int64_t dim = 0; dim < t.dim(); ++dim) {
      description.push_back(t.size(dim));
    }
    description.push_back((int64_t)targetType);
    return detail::TensorDescr(std::move(description));
  }

  return detail::TensorDescr();
}

[[maybe_unused]] std::string ToDebugString(const std::vector<int64_t>& xs) {
  std::string r = "[";
  const char* sep = "";
  for (auto x : xs) {
    r += sep;
    r += std::to_string(x);
    sep = ", ";
  }
  r += "]";
  return r;
}

[[maybe_unused]] std::string ToDebugString(const at::IValue& x) {
  if (x.isTensor()) {
    std::string t;
    t += "iTensor(st=";
    t += std::to_string((int64_t)x.toTensor().scalar_type());
    t += ", shape=";
    for (int i = 0; i < x.toTensor().dim(); ++i) {
      t += " ";
      t += std::to_string(x.toTensor().size(i));
    }
    t += ")";
    return t;
  }
  if (x.isIntList()) {
    std::string t;
    t += "iIntList(";
    std::vector<int64_t> xs = x.toIntVector();
    t += ToDebugString(xs);
    t += ")";
    return t;
  }

  std::string t;
  t += "iValue(";
  t += x.tagKind();
  t += ")";
  return t;
}

[[maybe_unused]] std::string ToDebugString(const detail::TensorDescr& x) {
  if (x.isTensor()) {
    std::string t;
    t += "Tensor(st=";
    t += std::to_string((int64_t)x.m_tensor->scalar_type());
    t += ", shape=";
    for (int i = 0; i < x.m_tensor->dim(); ++i) {
      t += " ";
      t += std::to_string(x.m_tensor->size(i));
    }
    t += ")";
    return t;
  } else {
    std::string t;
    t += "dims_and_type(";
    t += ToDebugString(x.m_dims_and_type);
    t += ")";
    return t;
  }
}

[[maybe_unused]] std::string ToDebugString(const std::vector<at::IValue>& xs) {
  std::string r = "[";
  const char* sep = "";
  for (auto x : xs) {
    r += sep;
    r += ToDebugString(x);
    sep = ", ";
  }
  r += "]";
  return r;
}

[[maybe_unused]] std::string ToDebugString(const detail::TensorDescrArray& xs) {
  std::string r = "[";
  const char* sep = "";
  for (auto x : xs) {
    r += sep;
    r += ToDebugString(x);
    sep = ", ";
  }
  r += "]";
  return r;
}

} // namespace

detail::TensorDescrArray CheckNodeWithSharedLayerValidator::
    CreateRegularInputList(const std::vector<at::IValue>& values) {
  detail::TensorDescrArray inputList;
  std::size_t limit = m_isOutFn ? values.size() - 1 : values.size();

  for (std::size_t i = 0; i < limit; ++i) {
    if (values[i].isTensor()) {
      inputList.push_back(detail::TensorDescr(&values[i].toTensor()));
    }
  }
  return inputList;
}

detail::TensorDescrArray CheckNodeWithSharedLayerValidator::
    CreateTypePromotionInputList(
        const std::vector<at::IValue>& values,
        at::ScalarType resultType) {
  detail::TensorDescrArray inputList;
  inputList.emplace_back(TryCastTensor(values[0], resultType));
  inputList.emplace_back(TryCastTensor(values[1], resultType));
  return inputList;
}

detail::TensorDescrArray CheckNodeWithSharedLayerValidator::CreateInputList(
    const std::vector<at::IValue>& values,
    at::ScalarType resultType) {
  if (m_typePromotion or m_promoteIntToFloat) {
    return CreateTypePromotionInputList(values, resultType);
  }

  return CreateRegularInputList(values);
}

detail::TensorDescrArray CheckNodeWithSharedLayerValidator::
    CreateRegularOutputList(const std::vector<at::IValue>& values) {
  detail::TensorDescrArray outputList;
  if (m_isOutFn) {
    outputList.push_back(detail::TensorDescr(&values.back().toTensor()));
  } else {
    outputList.push_back(detail::TensorDescr(&values.front().toTensor()));
  }
  return outputList;
}

detail::TensorDescrArray CheckNodeWithSharedLayerValidator::
    CreateTypePromotionOutputList(
        const std::vector<at::IValue>& values,
        at::ScalarType resultType) {
  detail::TensorDescrArray outputList;
  if (m_isOutFn) {
    outputList.push_back(detail::TensorDescr(&values.back().toTensor()));
  } else if (m_isInplace) {
    outputList.push_back(detail::TensorDescr(&values.front().toTensor()));
  } else {
    outputList.push_back(TryCastTensor(values.front(), resultType));
  }

  return outputList;
}

detail::TensorDescrArray CheckNodeWithSharedLayerValidator::CreateOutputList(
    const std::vector<at::IValue>& values,
    at::ScalarType resultType) {
  if (m_typePromotion or m_promoteIntToFloat) {
    return CreateTypePromotionOutputList(values, resultType);
  }

  return CreateRegularOutputList(values);
}

at::ScalarType CheckNodeWithSharedLayerValidator::ComputePromotedType(
    const std::vector<at::IValue>& values) {
  if (not(m_typePromotion or m_promoteIntToFloat)) {
    return at::ScalarType::Undefined;
  }

  c10::optional<const at::IValue*> output = c10::nullopt;
  if (m_isInplace) {
    output = &values.front();
  } else if (m_isOutFn) {
    output = &values.back();
  }

  auto dtype_helper = habana_helpers::DTypeHelper::
      binary_op_with_optional_int_to_float_promotion(
          values, m_promoteIntToFloat, output, m_safeCastCheck);
  auto common_type = dtype_helper.get_common_dtype();

  return common_type;
}

bool CheckNodeWithSharedLayerValidator::Validate(
    const at::Tensor& input,
    const std::vector<at::IValue>& values) {
  return Validate(input.scalar_type(), values);
}

bool CheckNodeWithSharedLayerValidator::Validate(
    at::ScalarType compute_type,
    const std::vector<at::IValue>& values) {
  bool result;
  if (getDeviceType() == gcapi::DEVICE_ID_GAUDI3) {
    result = ValidateWithDTypes(compute_type, values);
  } else {
    result = ValidateWithSharedLayer(compute_type, values);
  }
  if (not result) {
    PT_OP_INFO("Fallback for op", m_opname);
  }
  return result;
}

bool CheckNodeWithSharedLayerValidator::ValidateWithDTypes(
    at::ScalarType,
    const std::vector<at::IValue>& values) {
  std::size_t limit = values.size();
  if (m_isOutFn) {
    // The `out` tensor was not checked by old mechanism
    limit -= 1;
  }
  for (unsigned int i = 0; i < limit; ++i) {
    const auto& value = values[i];
    if (value.isTensor()) {
      if (not m_supportedDtypes.count(value.toTensor())) {
        return false;
      }
    }
  }

  return true;
}

bool CheckNodeWithSharedLayerValidator::ValidateWithSharedLayer(
    at::ScalarType,
    const std::vector<at::IValue>& values) {
  std::shared_ptr<void> params;

  if (m_fillNodeParamsFunc) {
    // FillNodeParams function can throw exception when some parameters are not
    // supported by HPU
    try {
      std::size_t params_size = 0;
      params = m_fillNodeParamsFunc(values, params_size);
    } catch (...) {
      PT_OP_INFO(
          "Shared layer rejected op, ",
          m_opname,
          ": guid=",
          m_guid,
          " cannot fill node parameters")
      return false;
    }
  }

  auto promoted_type = ComputePromotedType(values);
  auto inputs = CreateInputList(values, promoted_type);
  auto outputs = CreateOutputList(values, promoted_type);
  if (not ValidateGuid(m_guid, inputs, outputs, params.get())) {
    PT_OP_INFO(
        "Shared layer rejected op: ",
        m_opname,
        ":  guid=",
        m_guid,
        "inputlist=",
        ToDebugString(inputs),
        " outputlist=",
        ToDebugString(outputs),
        " values=",
        ToDebugString(values));
    return false;
  }
  return true;
}

} // namespace habana
