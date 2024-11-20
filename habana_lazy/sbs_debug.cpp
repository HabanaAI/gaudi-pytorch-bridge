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
#include "sbs_debug.h"
#include "aten_lazy_bridge.h"
#include "debug_utils.h"
#include "sbs_runner.h"
#include "tensor_comparator.hpp"

class float16;
class bfloat16;
namespace habana_lazy {

#define TENSOR_COMPARE_TYPE(primitive_type)                      \
  PT_LAZY_DEBUG("SBS: comparing tensors with type: ", cpu_type); \
  success = mp_tc->compare(                                      \
      tensor_name,                                               \
      (primitive_type*)hpu_data,                                 \
      (primitive_type*)cpu_data,                                 \
      cpu_res.numel(),                                           \
      compare_method,                                            \
      true);                                                     \
  break;

#define CASE_TENSOR_COMPARE_TYPE_WITH_CPP(scalar_type, primitive_type) \
  case scalar_type:                                                    \
    TENSOR_COMPARE_TYPE(primitive_type)
#define CASE_TENSOR_COMPARE_TYPE_SCALAR_TYPE(scalar_type) \
  CASE_TENSOR_COMPARE_TYPE_WITH_CPP(                      \
      scalar_type, decltype(c10::impl::ScalarTypeToCPPType<scalar_type>::t))

// handling duplicate names (adding a counter suffix)
static std::string handle_name_duplicates(const std::string& op_type) {
  std::string tensor_name = op_type;
  static std::map<std::string, int> checked_tensors_occurences;
  auto it = checked_tensors_occurences.find(tensor_name);
  if (it != checked_tensors_occurences.end()) {
    PT_LAZY_DEBUG("Tensor name: ", tensor_name, " check number: ", it->second);
    tensor_name += "_" + std::to_string(it->second++);
  }
  checked_tensors_occurences[tensor_name] = 1;

  return tensor_name;
}

bool SBSDebug::NeedToCompare(const HbLazyTensor& hb_tensor, bool update) {
  static std::map<int64_t, int> checked_tensors_versions;
  auto id = hb_tensor.getTensorUniqueId();
  if (!hb_tensor.GetSBSCompareIndication()) {
    PT_LAZY_DEBUG(
        "SBS: Tensor is disabled for comparison. Name: ",
        hb_tensor.CurrentIrValue().ToString(),
        " sbs name: ",
        hb_tensor.FetchSBSTensorName());
    return false;
  }
  auto version = hb_tensor.GetSBSTensorVersion();
  auto it = checked_tensors_versions.find(id);
  if (it != checked_tensors_versions.end()) {
    auto prev_compared_version = it->second;
    PT_LAZY_DEBUG(
        "SBS: Tensor name: ",
        hb_tensor.CurrentIrValue().ToString(),
        " sbs name: ",
        hb_tensor.FetchSBSTensorName(),
        " id: ",
        id,
        " previously compared version: ",
        prev_compared_version);
    if (version == prev_compared_version) {
      return false;
    }
  }
  if (update) {
    PT_LAZY_DEBUG("SBS: Updating compared tensor version to ", version);
    checked_tensors_versions[id] = version;
  }
  return true;
}

void SBSDebug::report(const std::string& log_message, size_t& log_counter) {
  mp_tc->makeReport(m_report_file_name, TensorComparison_pt::ExportType::CSV);
  ++log_counter;
  PT_LAZY_DEBUG(
      "SBS: Current number of ",
      log_message,
      " reported: ",
      log_counter,
      " total lines: ",
      GetNumberOfReportLines());
}

bool castToHigherType(
    at::Tensor& hpu_res_on_host_compare,
    at::Tensor& cpu_res_compare,
    std::string& original_types_str) {
  auto hpu_type = hpu_res_on_host_compare.scalar_type();
  auto cpu_type = cpu_res_compare.scalar_type();
  PT_LAZY_DEBUG(
      "SBS: Comparing different types, casting to the higher type. Assuming tensor types should be related. CPU: ",
      cpu_type,
      " HPU: ",
      hpu_type);
  bool cast_cpu_tensor;
  c10::ScalarType from_type;
  c10::ScalarType to_type;
  if (c10::isFloatingType(hpu_type) &&
      c10::isIntegralType(cpu_type, /*includeBool*/ true)) {
    PT_LAZY_DEBUG(
        "SBS: Trying to cast to the more precise type.",
        " HPU is floating, CPU is integral.");
    from_type = cpu_type;
    to_type = hpu_type;
    cast_cpu_tensor = true;
  } else if (
      c10::isIntegralType(hpu_type, /*includeBool*/ true) &&
      c10::isFloatingType(cpu_type)) {
    PT_LAZY_DEBUG(
        "SBS: Trying to cast to the more precise type.",
        " HPU is integral, CPU is floating.");
    from_type = hpu_type;
    to_type = cpu_type;
    cast_cpu_tensor = false;
  } else if (c10::elementSize(hpu_type) > c10::elementSize(cpu_type)) {
    PT_LAZY_DEBUG(
        "SBS: Trying to cast to the bigger sized type: HPU.",
        " (None or both are floating types)");
    from_type = cpu_type;
    to_type = hpu_type;
    cast_cpu_tensor = true;
  } else {
    PT_LAZY_DEBUG(
        "SBS: Trying to cast to the bigger sized type: CPU.",
        " (None or both are floating types)");
    from_type = hpu_type;
    to_type = cpu_type;
    cast_cpu_tensor = false;
  }
  if (!c10::canCast(from_type, to_type)) {
    PT_LAZY_DEBUG(
        "SBS: Cannot cast from type: ",
        c10::toString(from_type),
        " to type: ",
        c10::toString(to_type));
    return false;
  }
  original_types_str =
      std::string("Performed type casting. Original types: HPU: ") +
      c10::toString(hpu_type) + " CPU: " + c10::toString(cpu_type) +
      ". Compare type: " + c10::toString(to_type);
  PT_LAZY_DEBUG(
      "SBS: Casting ",
      cast_cpu_tensor ? "CPU" : "HPU",
      " tensor from ",
      from_type,
      " to ",
      to_type);
  if (cast_cpu_tensor) {
    cpu_res_compare.detach().to(to_type);
  } else {
    hpu_res_on_host_compare.detach().to(to_type);
  }

  return true;
}

void SBSDebug::compare_tensors_cos(
    const at::Tensor& hpu_res,
    const at::Tensor& cpu_res,
    const std::string& op_type) {
  std::string tensor_name = handle_name_duplicates(op_type);

  PT_LAZY_DEBUG(
      __FUNCTION__,
      " flushing tensor name=",
      GetHbLazyTensor(hpu_res).CurrentIrValue().ToString(),
      " sbs name: ",
      GetHbLazyTensor(hpu_res).FetchSBSTensorName(),
      " id=",
      GetHbLazyTensor(hpu_res).getTensorUniqueId(),
      " version: ",
      GetHbLazyTensor(hpu_res).GetSBSTensorVersion());
  const auto& hpu_res_on_host = hpu_res.to(c10::kCPU).detach();
  PT_LAZY_DEBUG(
      __FUNCTION__,
      " after flush, comparing tensor name=",
      GetHbLazyTensor(hpu_res).CurrentIrValue().ToString(),
      " sbs name: ",
      GetHbLazyTensor(hpu_res).FetchSBSTensorName(),
      " id=",
      GetHbLazyTensor(hpu_res).getTensorUniqueId(),
      " version: ",
      GetHbLazyTensor(hpu_res).GetSBSTensorVersion());
  auto cpu_res_compare = cpu_res.contiguous();
  auto hpu_res_on_host_compare = hpu_res_on_host.contiguous();

  auto hpu_type = hpu_res_on_host_compare.scalar_type();
  auto cpu_type = cpu_res_compare.scalar_type();
  std::string original_types_str = "";
  if ((hpu_type != cpu_type) &&
      !castToHigherType(
          hpu_res_on_host_compare, cpu_res_compare, original_types_str)) {
    LogError(
        tensor_name,
        std::string(
            "Could not compare this tensor: Different types. HPU type: ") +
            c10::toString(hpu_type) + " CPU type: " + c10::toString(cpu_type));
    return;
  }

  std::stringstream ss;
  ss << "HPU Shape: " << hpu_res.sizes();
  ss << " Strides: " << hpu_res.strides();
  ss << " Channel last: "
     << hpu_res.is_contiguous(c10::MemoryFormat::ChannelsLast) ||
      hpu_res.is_contiguous(c10::MemoryFormat::ChannelsLast3d);
  ss << " HPU comparison Shape: " << hpu_res_on_host_compare.sizes();
  ss << " Strides: " << hpu_res_on_host_compare.strides();
  ss << " Channel last: "
     << hpu_res_on_host_compare.is_contiguous(
            c10::MemoryFormat::ChannelsLast) ||
      hpu_res_on_host_compare.is_contiguous(c10::MemoryFormat::ChannelsLast3d);
  ss << " CPU Shape: " << cpu_res_compare.sizes();
  ss << " Strides: " << cpu_res_compare.strides();
  ss << " Channel last: "
     << cpu_res_compare.is_contiguous(c10::MemoryFormat::ChannelsLast) ||
      cpu_res_compare.is_contiguous(c10::MemoryFormat::ChannelsLast3d);
  std::string shapes_string(ss.str());
  std::string comment_string(shapes_string);
  if (!original_types_str.empty()) {
    comment_string += " " + original_types_str;
  }
  if ((!hpu_res_on_host_compare.strides().empty()) &&
      (!cpu_res_compare.strides().empty()) &&
      (hpu_res_on_host_compare.strides() != cpu_res_compare.strides())) {
    std::stringstream ss1;
    ss1 << "Could not compare this tensor: Different strides structure. "
        << comment_string;
    LogError(tensor_name, ss1.str());
    return;
  }
  if ((!hpu_res_on_host_compare.sizes().empty()) &&
      (!cpu_res_compare.sizes().empty()) &&
      (hpu_res_on_host_compare.sizes() != cpu_res_compare.sizes())) {
    std::stringstream ss1;
    ss1 << "Could not compare this tensor: Different shape structure. "
        << comment_string;
    LogError(tensor_name, ss1.str());
    return;
  }
  auto scalarType = cpu_res_compare.scalar_type();
  TensorComparison_pt::ComparisonMethods compare_method;
  compare_method.set(); // all test methods

  void* hpu_data = hpu_res_on_host_compare.data_ptr();
  void* cpu_data = cpu_res_compare.data_ptr();
  bool success = true;
  switch (scalarType) {
    // TODO: Fix when this is resolved: SW-78371
    // CASE_TENSOR_COMPARE_TYPE_WITH_CPP(c10::ScalarType::Byte, unsigned char);
    // TODO: Fix when this is resolved: SW-78371
    // CASE_TENSOR_COMPARE_TYPE_WITH_CPP(c10::ScalarType::Char, signed char);
    // TODO: Fix when this is resolved: SW-78371
    // CASE_TENSOR_COMPARE_TYPE_WITH_CPP(c10::ScalarType::Short, short);
    CASE_TENSOR_COMPARE_TYPE_SCALAR_TYPE(c10::ScalarType::Int);
    CASE_TENSOR_COMPARE_TYPE_SCALAR_TYPE(c10::ScalarType::Long);
    // TODO: [SW-73217] change to CASE_TENSOR_COMPARE_TYPE_SCALAR_TYPE
    CASE_TENSOR_COMPARE_TYPE_WITH_CPP(c10::ScalarType::Half, float16);
    CASE_TENSOR_COMPARE_TYPE_SCALAR_TYPE(c10::ScalarType::Float);
    // TODO: [SW-73217] change to CASE_TENSOR_COMPARE_TYPE_SCALAR_TYPE
    CASE_TENSOR_COMPARE_TYPE_WITH_CPP(c10::ScalarType::BFloat16, bfloat16);
    default:
      LogError(
          tensor_name,
          std::string("Could not compare this tensor: Unsupported type: ") +
              c10::toString(hpu_res_on_host_compare.scalar_type()));
      return;
  }

  if (!success) {
    PT_LAZY_WARN("Tensor Comparator failed to execute");
    return;
  }

  PT_LAZY_DEBUG("SBS: Adding comment: ", comment_string);
  mp_tc->addComment(tensor_name, comment_string);
  PT_LAZY_DEBUG("SBS: printing compare result for tensor: ", tensor_name);
  report("successful compares", m_number_of_successful_compares);
}

at::Tensor FetchAtenFromHbLazyTensor(HbLazyTensor hb_tensor) {
  auto attached_tensor = hb_tensor.CurrentTensorAttached();
  // getting real memory format from backend tensor
  c10::optional<c10::MemoryFormat> memory_format = c10::nullopt;
  if (attached_tensor.has_value()) {
    memory_format = attached_tensor.value().suggest_memory_format();
  }
  at::Tensor at_tensor = AtenFromHbLazyTensor(
      hb_tensor,
      /*tensor_type*/ c10::nullopt,
      /*size*/ c10::nullopt,
      /*stride*/ c10::nullopt,
      memory_format);
  return at_tensor;
}

void SBSDebug::CompareTensors(std::vector<HbLazyTensor>& tensors) {
  if (GET_ENV_FLAG_NEW(PT_SBS) == SBSModes::SBS_MODE_DISABLED) {
    return;
  }
  for (auto& hb_tensor : tensors) {
    c10::optional<at::Tensor> cpu_ref = hb_tensor.GetCPUTensorData();
    if (cpu_ref == c10::nullopt) {
      PT_LAZY_DEBUG(
          "SBS: Tensor is live (comparison point), but has no CPU (SBS is not supported). Name: ",
          hb_tensor.CurrentIrValue().ToString(),
          " sbs name: ",
          hb_tensor.FetchSBSTensorName());
    } else if (NeedToCompare(hb_tensor, /*update*/ true)) {
      std::string name = hb_tensor.FetchSBSTensorName();
      if (name.empty()) {
        name = std::string("Op Name N/A. ID ") +
            std::to_string(hb_tensor.getTensorUniqueId());
      }
      PT_LAZY_DEBUG(
          "Comparing tensor. Name: ",
          name,
          ", ID: ",
          hb_tensor.getTensorUniqueId());

      hb_tensor.SetSBSLiveTensorIndication(true); // Used by SBS modes 2 & 3
      at::Tensor at_tensor = FetchAtenFromHbLazyTensor(hb_tensor);
      PT_LAZY_DEBUG(
          "SBS: calling compare_tensors_cos. Name: ",
          hb_tensor.CurrentIrValue().ToString(),
          " sbs name: ",
          hb_tensor.FetchSBSTensorName());
      compare_tensors_cos(at_tensor, cpu_ref.value(), name);
    }
  }
  PT_LAZY_DEBUG(__FUNCTION__, " Done.");
}

bool SBSDebug::LogError(
    const std::string& op_name,
    const std::string& message_short,
    const std::string& message_detailed) {
  auto& message = (message_detailed.empty() ? message_short : message_detailed);
  PT_LAZY_DEBUG("SBS: Op tensor ", op_name, ": ", message);

  mp_tc->addComment(op_name, message_short);
  report("errors", m_number_of_errors);
  if (!m_error_file.is_open()) {
    PT_LAZY_DEBUG("SBS: Error file is not opened, can't log error");
    return false;
  }

  m_error_file << op_name << "," << message_short << std::endl;
  return true;
}

size_t SBSDebug::GetNumberOfReportLines() {
  return m_number_of_successful_compares + m_number_of_errors;
}
size_t SBSDebug::GetNumberOfErrorLines() {
  return m_number_of_errors;
}
size_t SBSDebug::GetNumberOfCompareLines() {
  return m_number_of_successful_compares;
}

void SBSDebug::init() {
  if (GET_ENV_FLAG_NEW(PT_SBS) != SBSModes::SBS_MODE_DISABLED &&
      !m_is_initialized) {
    PT_LAZY_DEBUG(
        "SBS: Tensor compare report will be saved to: ", m_report_file_name);
    m_error_file.open(m_error_file_name, std::ios::out);
    if (m_error_file.is_open()) {
      PT_LAZY_DEBUG("SBS: Error report will be saved to: ", m_error_file_name);
      m_error_file << "tensor name,comment" << std::endl;
      m_is_initialized = true;
    }
  }
}

void SBSDebug::reset() {
  PT_LAZY_DEBUG("SBS: Resetting SBSDebug");
  m_number_of_successful_compares = 0;
  m_number_of_errors = 0;
  m_number_of_accumulated_ops = 0;
  m_number_of_accumulated_op_output_tensors = 0;
}

SBSDebug::SBSDebug()
    : mp_tc(std::make_shared<TensorComparison_pt::TensorValidator>()),
      m_number_of_successful_compares(0),
      m_number_of_errors(0),
      m_number_of_accumulated_ops(0),
      m_number_of_accumulated_op_output_tensors(0),
      m_is_initialized(false) {}

} // namespace habana_lazy