/*******************************************************************************
 * Copyright (C) 2023-2024 Habana Labs, Ltd. an Intel Company
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
#include "backend/habana_device/HPUGuardImpl.h"
#include "backend/habana_device/hpu_cached_devices.h"
#include "habana_kernels/index_kernels.h"
#include "habana_kernels/lazy_kernels.h"
#include "habana_kernels/random_gen_kernels.h"
#include "hpu_ops/hpu_op_helper.h"
#include "op_backend.h"
#include "pytorch_helpers/habana_helpers/pt_version_check.h"

namespace habana {

namespace {

struct SharedLayerInitialization {
  SharedLayerInitialization() {
    static auto status = synSharedLayerInit();
    TORCH_CHECK(
        SharedLayer::Return_t::SHARED_LAYER_SUCCESS == status,
        "cannot initialize shared layer");
  }

  ~SharedLayerInitialization() {
    synSharedLayerFinit();
  }
};

SharedLayerInitialization _slu_initializer;

SharedLayer::DeviceId synDeviceTypeToSharedLayerType(synDeviceType tp) {
  switch (tp) {
    case synDeviceGaudi:
      return SharedLayer::DeviceId::DEVICE_ID_GAUDI;
    case synDeviceGaudi2:
      return SharedLayer::DeviceId::DEVICE_ID_GAUDI2;
    case synDeviceGaudi3:
      return SharedLayer::DeviceId::DEVICE_ID_GAUDI3;
    default:
      break;
  }

  TORCH_CHECK(false, "unsupported synDeviceType for shared layer");
}

SharedLayer::DeviceId _getDeviceType() {
  // getDevice should be invoked in case device has not been initialized yet.
  HABANAGuardImpl device_guard;
  device_guard.getDevice();

  auto deviceType = HPURegistrar::get_device(0).type();
  auto deviceId = synDeviceTypeToSharedLayerType(deviceType);
  return deviceId;
}

SharedLayer::DeviceId getDeviceType() {
  static auto deviceId = _getDeviceType();
  return deviceId;
}

bool fillSharedLayerTensorType(SharedLayer::Tensor& tensor, at::ScalarType t) {
  switch (t) {
    case at::ScalarType::Byte:
      tensor.geometry.dataType = SharedLayer::TensorDataType::DATA_U8;
      return true;
    case at::ScalarType::Char:
      tensor.geometry.dataType = SharedLayer::TensorDataType::DATA_I8;
      return true;
    case at::ScalarType::Short:
      tensor.geometry.dataType = SharedLayer::TensorDataType::DATA_I16;
      return true;
    case at::ScalarType::Int:
    case at::ScalarType::Long:
      tensor.geometry.dataType = SharedLayer::TensorDataType::DATA_I32;
      return true;
    case at::ScalarType::Half:
      tensor.geometry.dataType = SharedLayer::TensorDataType::DATA_F16;
      return true;
    case at::ScalarType::Float:
    case at::ScalarType::Double:
      tensor.geometry.dataType = SharedLayer::TensorDataType::DATA_F32;
      return true;
    case at::ScalarType::Bool:
      tensor.geometry.dataType = SharedLayer::TensorDataType::DATA_I8;
      return true;
    case at::ScalarType::BFloat16:
      tensor.geometry.dataType = SharedLayer::TensorDataType::DATA_BF16;
      return true;
    case at::ScalarType::Float8_e5m2:
      tensor.geometry.dataType = SharedLayer::TensorDataType::DATA_F8_152;
      return true;
    case at::ScalarType::Float8_e4m3fn:
      tensor.geometry.dataType = SharedLayer::TensorDataType::DATA_F8_143;
      return true;
    case at::ScalarType::Undefined:
      tensor.geometry.dataType = SharedLayer::TensorDataType::NUM_DATATYPES;
      return true;
    default:
      tensor.geometry.dataType = SharedLayer::TensorDataType::NUM_DATATYPES;
      return false;
  }
}

bool fillGuidParamInfo(
    SharedLayer::Tensor& tensor,
    const detail::TensorDescr& tensor_descr) {
  if (not fillSharedLayerTensorType(tensor, tensor_descr.getType()))
    return false;

  const auto rank = tensor_descr.getRank();
  const bool isOptionalNotPresent =
      tensor_descr.getType() == at::ScalarType::Undefined;
  tensor.geometry.dims = (rank == 0 && !isOptionalNotPresent) ? 1 : rank;
  return true;
}

template <size_t MaxSize>
void safe_string_copy(const std::string& source, char* destination) {
  static const auto limited_length_string_format =
      "%." + std::to_string(MaxSize) + "s";
  sprintf(destination, limited_length_string_format.c_str(), source.c_str());
}

/*
 * This function is a wrapper for shared layer query interface.
 */
SharedLayer::Return_t ValidateGuid(
    const std::string& guid,
    const detail::TensorDescrArray& input_values,
    const detail::TensorDescrArray& output_values,
    bool is_dynamic = false) {
  SharedLayer::ParamsV2_t params{};
  params.apiVersion = 1;
  auto deviceId = getDeviceType();
  params.deviceId = deviceId;

  safe_string_copy<SharedLayer::MAX_NODE_NAME>(guid, params.guid.name);
  // skipping:
  // params.guid.nameHash - not used in lower layer
  // params.guid.kernelProperties - not used in lower layer
  // params.nodeParams.nodeParams - not used in lower layer
  // params.nodeParams.nodeParamsSize - not used in lower layer

  const size_t input_count = input_values.size();
  const size_t output_count = output_values.size();

  HABANA_ASSERT(
      input_count <= SharedLayer::MAX_TENSOR_NR,
      "Input count passed to Shared Layer exceeds limit");

  HABANA_ASSERT(
      output_count <= SharedLayer::MAX_TENSOR_NR,
      "Output count passed to Shared Layer exceeds limit");

  SharedLayer::Tensor input_tensors[input_count];
  SharedLayer::Tensor output_tensors[output_count];

  for (auto i = 0u; i < input_count; ++i) {
    if (not fillGuidParamInfo(input_tensors[i], input_values[i])) {
      return SharedLayer::Return_t::SHARED_LAYER_FAILED;
    }
  }
  params.inputTensorNr = input_count;

  for (auto i = 0u; i < output_count; ++i) {
    if (not fillGuidParamInfo(output_tensors[i], output_values[i])) {
      return SharedLayer::Return_t::SHARED_LAYER_FAILED;
    }
  }
  params.outputTensorNr = output_count;

  params.inputTensors = input_tensors;
  params.outputTensors = output_tensors;
  params.supportsDynamicShapes = is_dynamic;

  return synSharedLayerValidateGuidV2(&params);
}

detail::TensorDescr TryCastTensor(
    const at::Tensor& t,
    at::ScalarType targetType) {
  if (targetType == at::ScalarType::Undefined or
      targetType == t.scalar_type()) {
    return detail::TensorDescr(&t);
  }
  return detail::TensorDescr(t.dim(), targetType);
}

detail::TensorDescr HandleTensor(
    const at::Tensor& tensor,
    at::ScalarType promotionType) {
  if (promotionType == at::ScalarType::Undefined) {
    return detail::TensorDescr(&tensor);
  }
  return TryCastTensor(tensor, promotionType);
}

detail::TensorDescr HandleScalar(
    const at::Scalar& scalar,
    at::ScalarType promotionType) {
  auto dtype = promotionType == at::ScalarType::Undefined ? scalar.type()
                                                          : promotionType;
  return detail::TensorDescr(1, dtype);
}

bool VectorContains(const std::vector<int>& vec, const int value) {
  return std::find(vec.begin(), vec.end(), value) != vec.end();
}

at::ScalarType MaybePromotionType(
    const std::vector<int>& promotion_ids,
    const int id,
    at::ScalarType promotionType) {
  return VectorContains(promotion_ids, id) ? promotionType
                                           : at::ScalarType::Undefined;
}

template <class T>
detail::TensorDescrArray CreateTensorList(const T& meta) {
  detail::TensorDescrArray tensorList;
  for (const auto& out_meta : meta) {
    tensorList.emplace_back(out_meta);
  }
  return tensorList;
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

[[maybe_unused]] std::string ToDebugString(const bool x) {
  std::string t;
  t += "Bool(";
  t += x ? "true" : "false";
  t += ")";
  return t;
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
  std::string t;
  t += "Tensor(st=";
  t += std::to_string((int64_t)x.getType());
  t += ", rank=";
  t += std::to_string(x.getRank());
  t += ")";
  return t;
}

[[maybe_unused]] std::string ToDebugString(const at::Stack& xs) {
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

std::string ToDebugString(const SharedLayer::Return_t errcode) {
  switch (errcode) {
    case SharedLayer::Return_t::SHARED_LAYER_SUCCESS:
      return "SUCCESS";
    case SharedLayer::Return_t::SHARED_LAYER_GUID_NOT_FOUND:
      return "GUID_NOT_FOUND";
    case SharedLayer::Return_t::SHARED_LAYER_INCOMPATIBLE_INPUT_COUNT:
      return "INCOMPATIBLE_INPUT_COUNT";
    case SharedLayer::Return_t::SHARED_LAYER_INCOMPATIBLE_INPUT_DIMENSION:
      return "INCOMPATIBLE_INPUT_DIMENSION";
    case SharedLayer::Return_t::SHARED_LAYER_INCOMPATIBLE_INPUT_SIZE:
      return "INCOMPATIBLE_INPUT_SIZE";
    case SharedLayer::Return_t::SHARED_LAYER_INCOMPATIBLE_OUTPUT_COUNT:
      return "INCOMPATIBLE_OUTPUT_COUNT";
    case SharedLayer::Return_t::SHARED_LAYER_INCOMPATIBLE_OUTPUT_DIMENSION:
      return "INCOMPATIBLE_OUTPUT_DIMENSION";
    case SharedLayer::Return_t::SHARED_LAYER_INCOMPATIBLE_OUTPUT_SIZE:
      return "INCOMPATIBLE_OUTPUT_SIZE";
    case SharedLayer::Return_t::SHARED_LAYER_INCOMPATIBLE_DATA_TYPE:
      return "INCOMPATIBLE_DATA_TYPE";
    case SharedLayer::Return_t::SHARED_LAYER_UNSUPPORTED_LAYER_CONFIGURATION:
      return "UNSUPPORTED_LAYER_CONFIGURATION";
    case SharedLayer::Return_t::SHARED_LAYER_UNSUPPORTED_QUANT_PARAMS:
      return "UNSUPPORTED_QUANT_PARAMS";
    case SharedLayer::Return_t::SHARED_LAYER_UNSUPPORTED_BROADCAST_MODE:
      return "UNSUPPORTED_BROADCAST_MODE";
    case SharedLayer::Return_t::SHARED_LAYER_KERNEL_INVALID_SCALAR_ARGUMENT:
      return "INVALID_KERNEL_SCALAR_ARGUMENT";
    case SharedLayer::Return_t::SHARED_LAYER_MISSING_PRIVATE_STRUCTURE:
      return "MISSING_PRIVATE_STRUCTURE";
    case SharedLayer::Return_t::SHARED_LAYER_GUID_MISSING_DYNAMIC_SUPPORT:
      return "MISSING_DYNAMIC_SUPPORT";
    case SharedLayer::Return_t::SHARED_LAYER_FAILED:
    default:
      return "UNKNOWN_FAILURE";
  }
}
} // namespace

detail::TensorDescrArray CheckNodeWithSharedLayerValidator::CreateInputList(
    const at::Stack& values,
    at::ScalarType promotionType,
    const size_t outs_num) {
  detail::TensorDescrArray inputList;
  size_t limit = m_isOutFn ? values.size() - outs_num : values.size();

  for (std::size_t i = 0; i < limit; ++i) {
    const auto& val = values[i];
    if (val.isTensor() and val.toTensor().defined()) {
      inputList.push_back(HandleTensor(
          val.toTensor(),
          MaybePromotionType(m_typePromotionIds, i, promotionType)));
    } else if (val.isScalar() and VectorContains(m_scalarIds, i)) {
      inputList.push_back(HandleScalar(
          val.toScalar(),
          MaybePromotionType(m_typePromotionIds, i, promotionType)));
    }
  }
  return inputList;
}

at::ScalarType CheckNodeWithSharedLayerValidator::ComputePromotedType(
    const at::Stack& values) {
  if (m_typePromotionIds.empty()) {
    return at::ScalarType::Undefined;
  }

  c10::optional<const at::IValue*> output = c10::nullopt;
  if (m_isInplace) {
    output = &values.front();
  } else if (m_isOutFn) {
    output = &values.back();
  }

  at::Stack stack;
  for (const auto id : m_typePromotionIds) {
    stack.push_back(values[id]);
  }

  const auto& dtype_helper =
      habana_helpers::DTypeHelper::op_with_optional_dtype_promotion(
          stack, m_promoteIntToFloat, output, m_safeCastCheck);

  return dtype_helper.get_common_dtype();
}

std::unordered_set<std::string> load_static_guids(
    const std::string_view list_name,
    const std::unordered_set<std::string>& default_list) {
  auto static_guids_path = std::getenv(list_name.data());
  if (static_guids_path) {
    std::ifstream file(static_guids_path);
    if (!file.is_open()) {
      PT_BRIDGE_WARN(
          "Failed to open file with static guids: ",
          static_guids_path,
          ". Use built in list instead.");
      return default_list;
    } else {
      std::unordered_set<std::string> static_guids_list;
      std::string line;
      std::string ops;
      while (getline(file, line)) {
        static_guids_list.insert(line);
        ops += line + ", ";
      }
      PT_BRIDGE_DEBUG("Static guids loaded: ", ops);
      return static_guids_list;
    }
  } else {
    return default_list;
  }
}

bool is_guid_support_dynamic_shape(const std::string& guid) {
  using namespace std::literals;
  // placeholder list containing guids only support static shape in tpc_kernels
  // and CGUID
  static const std::unordered_set<std::string> tpc_static_guids = {};

  static const std::unordered_set<std::string> static_guids_list =
      load_static_guids("PT_HPU_STATIC_GUIDS", tpc_static_guids);

  return !static_guids_list.count(guid);
}

bool CheckNodeWithSharedLayerValidator::Validate(
    const at::Stack& values,
    bool is_dynamic,
    const SharedMetaVector& meta) {
  auto promoted_type = ComputePromotedType(values);

  detail::TensorDescrArray outputs;
  if (not meta.empty()) {
    outputs = CreateTensorList(meta);
  } else if (m_outputMetaFunc) {
    outputs = CreateTensorList(m_outputMetaFunc(values));
  } else if (not m_resIds.empty()) {
    const auto is_promoted = promoted_type != at::ScalarType::Undefined;
    const auto values_size = values.size();
    for (auto id : m_resIds) {
      if (id < 0) {
        id += values_size;
      }
      const auto& tensor = values[id].toTensor();
      auto dtype = is_promoted ? promoted_type : tensor.scalar_type();
      outputs.emplace_back(tensor.dim(), dtype);
    }
  } else {
    TORCH_CHECK(
        false,
        "Op should be either _out or have defined one of [output_meta, res_ids, inplace_ids]");
  }

  auto inputs = CreateInputList(values, promoted_type, outputs.size());

  auto validation_result = ValidateGuid(m_guid, inputs, outputs, is_dynamic);

  if (SharedLayer::Return_t::SHARED_LAYER_SUCCESS == validation_result &&
      is_dynamic && !is_guid_support_dynamic_shape(m_guid)) {
    validation_result =
        SharedLayer::Return_t::SHARED_LAYER_GUID_MISSING_DYNAMIC_SUPPORT;
  }

  if (SharedLayer::Return_t::SHARED_LAYER_SUCCESS != validation_result) {
    // This log line is used by the logging analysis tool. Please be cautious
    // when changing.
    PT_OP_INFO(
        "Shared layer rejected op: ",
        m_opname,
        ":  guid=",
        m_guid,
        " inputlist=",
        ToDebugString(inputs),
        " outputlist=",
        ToDebugString(outputs),
        " values=",
        ToDebugString(values),
        " is_dynamic=",
        ToDebugString(is_dynamic),
        " reason=",
        ToDebugString(validation_result));
    PT_OP_INFO("Fallback for op: ", m_opname);
    return false;
  }

  // (TODO)query if h2d and st are needed
  m_require_h2d = false;
  m_require_st = false;
  return true;
}

bool CheckNodeWithSharedLayerValidator::ValidateCustom(
    const at::Stack& values,
    bool is_dynamic) {
  for (const auto& meta : m_sharedMetaFunc(values)) {
    auto inputs = CreateTensorList(meta.inputs_data);
    auto outputs = CreateTensorList(meta.outputs_data);

    auto validation_result =
        ValidateGuid(meta.guid, inputs, outputs, is_dynamic);

    if (SharedLayer::Return_t::SHARED_LAYER_SUCCESS != validation_result) {
      // This log line is used by the logging analysis tool. Please be cautious
      // when changing.
      PT_OP_INFO(
          "Shared layer rejected complex op: ",
          m_opname,
          ":  guid=",
          meta.guid,
          " inputlist=",
          ToDebugString(inputs),
          " outputlist=",
          ToDebugString(outputs),
          " is_dynamic=",
          ToDebugString(is_dynamic),
          " reason=",
          ToDebugString(validation_result));
      PT_OP_INFO("Fallback for op: ", m_opname);
      return false;
    }
  }

  return true;
}

} // namespace habana
