/******************************************************************************
 * Copyright (C) 2022 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */
#include "sbs_debug.h"
#include "aten_lazy_bridge.h"
#include "debug_utils.h"

class float16;
class bfloat16;
namespace habana_lazy {

#define TENSOR_COMPARE_TYPE(primitive_type) \
  success = tc.compare(                     \
      tensor_name,                          \
      (primitive_type*)hpu_data,            \
      (primitive_type*)cpu_data,            \
      cpu_res.numel(),                      \
      compare_method,                       \
      true);                                \
  break;

#define CASE_TENSOR_COMPARE_TYPE(scalar_type, primitive_type) \
  case scalar_type:                                           \
    TENSOR_COMPARE_TYPE(primitive_type)

// handling duplicate names (adding a counter suffix)
static std::string handle_duplicates(const std::string& op_type) {
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

void SBSDebug::compare_tensors_cos(
    at::Tensor hpu_res,
    at::Tensor cpu_res,
    const std::string& op_type) {
  std::string tensor_name = handle_duplicates(op_type);

  static TensorComparison::TensorValidator tc;
  auto hpu_res_on_host = hpu_res.to("cpu");
  if (hpu_res_on_host.dtype() == cpu_res.dtype()) {
    auto scalarType = cpu_res.scalar_type();
    TensorComparison::ComparisonMethods compare_method;
    compare_method.set(); // all test methods

    void* hpu_data = hpu_res_on_host.data_ptr();
    void* cpu_data = cpu_res.data_ptr();
    bool success = true;
    switch (scalarType) {
      CASE_TENSOR_COMPARE_TYPE(c10::ScalarType::Byte, unsigned char)
      CASE_TENSOR_COMPARE_TYPE(c10::ScalarType::Char, signed char)
      CASE_TENSOR_COMPARE_TYPE(c10::ScalarType::Short, short)
      CASE_TENSOR_COMPARE_TYPE(c10::ScalarType::Long, long)
      CASE_TENSOR_COMPARE_TYPE(c10::ScalarType::Half, float16)
      CASE_TENSOR_COMPARE_TYPE(c10::ScalarType::Float, float)
      CASE_TENSOR_COMPARE_TYPE(c10::ScalarType::BFloat16, bfloat16)
      default:
        PT_LAZY_WARN(
            "sbs could not run on this op due to unsupported dtype. dtype: ",
            hpu_res_on_host.dtype());
        return;
    }

    if (!success) {
      PT_LAZY_WARN("Tensor Comparator failed to execute");
      return;
    }
    tc.makeReport(m_report_file_name, TensorComparison::ExportType::CSV);
  } else {
    PT_LAZY_WARN(
        "sbs could not run on this op due to different dtype. hpu dtype: ",
        hpu_res_on_host.dtype(),
        " cpu dtype:",
        cpu_res.dtype());
  }
}

void SBSDebug::CompareTensors(std::vector<HbLazyTensor>& tensors) {
  if (GET_ENV_FLAG_NEW(PT_SBS) == SBSModes::SBS_MODE_DISABLED) {
    return;
  }
  for (auto& hb_tensor : tensors) {
    at::Tensor at_tensor = AtenFromHbLazyTensor(
        hb_tensor, c10::nullopt, c10::nullopt, c10::nullopt, c10::nullopt);
    c10::optional<at::Tensor> cpu_ref = hb_tensor.GetCPUTensorData();
    if (cpu_ref == c10::nullopt) {
      PT_LAZY_DEBUG(
          "SBS: Tensor is live (comparison point), but has no CPU (SBS is not supported). Name: ",
          hb_tensor.CurrentIrValue().ToString())
    } else if (!hb_tensor
                    .GetSBSLiveTensorIndication()) // We'll avoid tensors that
                                                   // we've already checked
    {
      std::string name = hb_tensor.CurrentIrValue().ToString();
      if (name.empty()) {
        name = std::string("Op Name N/A. ID ") +
            std::to_string(hb_tensor.getTensorUniqueId());
      }
      PT_LAZY_DEBUG(
          "Comparing tensor. Name: ",
          name,
          ", ID: ",
          hb_tensor.getTensorUniqueId());

      compare_tensors_cos(at_tensor, cpu_ref.value(), name);
      hb_tensor.SetSBSLiveTensorIndication(); // Used by SBS modes 2 & 3
    }
  }
}

bool SBSDebug::LogError(
    const std::string& op_name,
    const std::string& message_short,
    const std::string& message_detailed) {
  auto& message = (message_detailed.empty() ? message_short : message_detailed);
  PT_LAZY_DEBUG("SBS: Op ", op_name, ": ", message);

  m_tc.addComment(op_name, message_short);
  m_tc.makeReport(m_report_file_name, TensorComparison::ExportType::CSV);
  if (!m_error_file.is_open()) {
    PT_LAZY_DEBUG("SBS: Error file is not opened, can't log error");
    return false;
  }

  m_error_file << op_name << "," << message_short << std::endl;
  return true;
}

SBSDebug::SBSDebug() {
  PT_LAZY_DEBUG(
      "SBS: Tensor compare report will be saved to: ", m_report_file_name);
  m_error_file.open(m_error_file_name, std::ios::out);
  if (m_error_file.is_open()) {
    PT_LAZY_DEBUG("SBS: Error report will be saved to: ", m_error_file_name);
    m_error_file << "tensor name,comment" << std::endl;
  }
}

} // namespace habana_lazy