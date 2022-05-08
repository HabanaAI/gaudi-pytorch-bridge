/******************************************************************************
 * Copyright (C) 2022 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */

#include "habana_lazy/permute_tensors.h"
#include <c10/core/Storage.h>
#include "habana_device/HPUAllocator.h"
#include "habana_helpers/logging.h"
#include "habana_kernels/lazy_kernels_declarations.h"
#include "habana_lazy/aten_lazy_bridge.h"
#include "habana_lazy/hpu_lazy_tensors.h"
#include "synapse_helpers/layout_utils.h"

using namespace synapse_helpers::layouts;

namespace habana_lazy {

unsigned PermuteTensors::m_permute_counter = 0;

void PermuteTensors::permuteWeight(torch::Tensor& weight) {
  PT_LAZY_TRACE;
  TORCH_CHECK(
      GET_ENV_FLAG_NEW(PT_HPU_ENABLE_SYNAPSE_LAYOUT_HANDLING),
      "PermuteWeight only for Synapse layout handling mode");
  TORCH_CHECK(
      weight.device().type() == c10::DeviceType::HPU,
      "permuteWeight only for HPU tensors");
  if (shouldPermuteWeight(weight)) {
    permuteWeightByDim(weight);
  } else if (shouldPermutePreCastedWeight(weight)) {
    auto pre_caster_weight = getPreCastedWeight(weight);
    permuteWeightByDim(pre_caster_weight);
  }
}

void PermuteTensors::permuteWeightByDim(torch::Tensor& weight) {
  auto dim = weight.dim();
  if (dim == 4) {
    habana_lazy::PermuteTensors::permuteWeightToRSCKInMemory(weight);
  } else if (dim == 5) {
    habana_lazy::PermuteTensors::permuteWeightToQRSCKInMemory(weight);
  } else {
    HABANA_ASSERT(false && "Permute weight support only 4/5D tensors");
  }
}

void PermuteTensors::permuteWeightToRSCKInMemory(torch::Tensor& weight) {
  PT_LAYOUTS_DEBUG("Permuting weight to RSCK, count: ", m_permute_counter++);
  torch::Tensor weight_cpu = weight.to(c10::kCPU);
  if (weight.scalar_type() == c10::ScalarType::BFloat16) {
    permuteWeightTensorDataToRSCK<c10::BFloat16>(weight_cpu);
  } else {
    permuteWeightTensorDataToRSCK<float>(weight_cpu);
  }
  copy_hpu_lazy_(weight, weight_cpu, false);

  // Update lazy & impl status
  HbLazyTensor weight_hb_tensor = GetHbLazyTensor(weight);
  auto hb_weight_data = weight_hb_tensor.GetHbLazyTensorData().value();
  auto hb_weight_impl = habana_lazy::GetHbInternalTensorImpl(hb_weight_data);
  hb_weight_impl->SetMemoryPermutation(
      synapse_helpers::layouts::weight_rsck_in_memory);
}

void PermuteTensors::permuteWeightToQRSCKInMemory(torch::Tensor& weight) {
  PT_LAYOUTS_DEBUG("Permuting weight to QRSCK, count: ", m_permute_counter++);
  torch::Tensor weight_cpu = weight.to(c10::kCPU);
  if (weight.scalar_type() == c10::ScalarType::BFloat16) {
    restrideWeightTensorDataToQRSCK<c10::BFloat16>(weight_cpu);
  } else {
    restrideWeightTensorDataToQRSCK<float>(weight_cpu);
  }
  copy_hpu_lazy_(weight, weight_cpu, false);

  // Update lazy & impl status
  // Currently updating both as between iteration impl info is vanished
  // While lazy tensor is kept & in lowering to Synapse stage we don't
  // have access to lazy tensors.
  HbLazyTensor weight_hb_tensor = GetHbLazyTensor(weight);
  auto hb_weight_data = weight_hb_tensor.GetHbLazyTensorData().value();
  auto hb_weight_impl = habana_lazy::GetHbInternalTensorImpl(hb_weight_data);
  PT_LAYOUTS_DEBUG(
      "PermuteTensors setting permutation on tensor: ",
      weight_hb_tensor.getDataPtr()->ir_value.mp_node->get_id(),
      " hbinternal address: ",
      reinterpret_cast<void*>(hb_weight_impl));
  hb_weight_impl->SetMemoryPermutation(
      synapse_helpers::layouts::weight_qrsck_in_memory);
}

bool PermuteTensors::shouldPermuteWeight(const torch::Tensor& weight) {
  HbLazyTensor weight_hb_tensor = GetHbLazyTensor(weight);
  PT_LAYOUTS_DEBUG(
      "shouldPermuteWeight tensor: ", weight_hb_tensor.getTensorUniqueId())
  bool is_input = weight_hb_tensor.CurrentIrValue().IsHpuInputNode();
  if (!is_input) {
    PT_LAYOUTS_DEBUG("shouldPermuteWeight is not input to the graph");
    return false;
  }
  // Don't access tensor impl if not graph input
  auto hb_weight_data = weight_hb_tensor.GetHbLazyTensorData().value();
  auto hb_weight_impl = habana_lazy::GetHbInternalTensorImpl(hb_weight_data);
  auto required_permute = weight.dim() == 4
      ? synapse_helpers::layouts::weight_rsck_in_memory
      : synapse_helpers::layouts::weight_qrsck_in_memory;
  bool is_permuted = hb_weight_impl->GetMemoryPermutation() == required_permute;
  if (is_permuted) {
    PT_LAYOUTS_DEBUG(
        "shouldPermuteWeight already permuted to: ",
        VecToString(required_permute));
  }
  return is_input && !is_permuted;
}

bool PermuteTensors::shouldPermutePreCastedWeight(const torch::Tensor& weight) {
  HbLazyTensor weight_hb_tensor = GetHbLazyTensor(weight);
  PT_LAYOUTS_DEBUG(
      "shouldPermutePreCastedWeight tensor: ",
      weight_hb_tensor.getTensorUniqueId())
  bool is_input = weight_hb_tensor.CurrentIrValue().IsHpuInputNode();
  if (!is_input) {
    PT_LAYOUTS_DEBUG(
        "shouldPermutePreCastedWeight tensor isn't an input to the graph.")
    const auto& ir_value = weight_hb_tensor.GetIrValue();
    const auto& ir_node = ir_value.mp_node;
    const auto& ir_op = ir_node->op();
    // Permuting only if weight is is ouptut of cast (fp->bf16) and originaly
    // input to graph.
    if (strcmp(ir_op.toQualString(), "hpu::cast") == 0) {
      const auto& ir_inputs = ir_node->GetInputs();
      const auto& ir_weight_value = ir_inputs[0];
      std::shared_ptr<Data> d1 = ir_weight_value.m_data_ptr.lock();
      if (ir_weight_value.IsHpuInputNode()) {
        // Checking if orginal weight already permuted
        auto tensor_data = d1->tensor_data.value();
        auto hb_weight_impl = habana_lazy::GetHbInternalTensorImpl(tensor_data);
        auto required_permute = weight.dim() == 4
            ? synapse_helpers::layouts::weight_rsck_in_memory
            : synapse_helpers::layouts::weight_qrsck_in_memory;
        bool is_permuted =
            hb_weight_impl->GetMemoryPermutation() == required_permute;
        if (!is_permuted) {
          PT_LAYOUTS_DEBUG(
              "Permuting pre-casted weight. IR Value id: ", d1->unique_id);
          return true;
        } else {
          PT_LAYOUTS_DEBUG(
              "Already permuted pre-casted weight. IR Value id: ",
              d1->unique_id);
          return false;
        }
      } else {
        PT_LAYOUTS_DEBUG(
            "Pre-casted weight isn't input to graph, not permuting. IR Value id: ",
            d1->unique_id);
        return false;
      }
    }
    std::shared_ptr<Data> d1 = ir_value.m_data_ptr.lock();
    PT_LAYOUTS_DEBUG("Weight isn't an output of a cast: ", d1->unique_id);
    return false;
  }
  PT_LAYOUTS_DEBUG(
      "shouldPermutePreCastedWeight tensor is an input to the graph.")
  return false;
}

const torch::Tensor PermuteTensors::getPreCastedWeight(
    const torch::Tensor& weight) {
  HbLazyTensor weight_hb_tensor = GetHbLazyTensor(weight);
  const auto& ir_value = weight_hb_tensor.GetIrValue();
  const auto& ir_node = ir_value.mp_node;
  const auto& ir_inputs = ir_node->GetInputs();
  const auto& ir_weight_value = ir_inputs[0];
  std::shared_ptr<Data> d1 = ir_weight_value.m_data_ptr.lock();
  auto tensor_data = d1->tensor_data.value();
  auto copy_to_cpu = empty_hpu_lazy(
      weight.sizes(),
      weight.options().dtype(c10::ScalarType::Float),
      c10::MemoryFormat::Contiguous,
      false);
  auto hl_copy_to_cpu = GetHbLazyTensor(copy_to_cpu);
  hl_copy_to_cpu.AssignIrValue(ir_weight_value);
  hl_copy_to_cpu.SetTensorData(tensor_data);
  return copy_to_cpu;
}

template <typename T>
void PermuteTensors::permuteWeightTensorDataToRSCK(
    const torch::Tensor& weight) {
  T* ptr = (T*)weight.data_ptr();
  auto strides = weight.strides();
  auto sizes = weight.sizes();

  // Creating temp buffer the size of the tensor
  T* tempBuff = new T[weight.numel()]();
  int buffer_counter = 0;

  // Copy data to tmp buffer in RSCK memory format
  for (int r = 0; r < sizes[WEIGHT_KERNEL_R_IDX]; ++r) {
    for (int s = 0; s < sizes[WEIGHT_KERNEL_S_IDX]; ++s) {
      for (int c = 0; c < sizes[WEIGHT_KERNEL_C_IDX]; ++c) {
        for (int k = 0; k < sizes[WEIGHT_KERNEL_K_IDX]; ++k) {
          tempBuff[buffer_counter] =
              ptr[k * strides[WEIGHT_KERNEL_K_IDX] +
                  c * strides[WEIGHT_KERNEL_C_IDX] +
                  r * strides[WEIGHT_KERNEL_R_IDX] +
                  s * strides[WEIGHT_KERNEL_S_IDX]];
          buffer_counter++;
        }
      }
    }
  }

  // Copy tmp buffer to original tensor memory
  std::memcpy(ptr, tempBuff, weight.numel() * sizeof(T));
  delete[] tempBuff;
}

template <typename T>
void PermuteTensors::restrideWeightTensorDataToQRSCK(
    const torch::Tensor& weight) {
  T* ptr = (T*)weight.data_ptr();
  auto strides = weight.strides();
  auto sizes = weight.sizes();

  // Creating temp buffer the size of the tensor
  T* tempBuff = new T[weight.numel()]();
  int buffer_counter = 0;

  // Copy data to tmp buffer in QRSCK memory format
  for (int q = 0; q < sizes[WEIGHT_KERNEL_3D_Q_IDX]; ++q) {
    for (int r = 0; r < sizes[WEIGHT_KERNEL_3D_R_IDX]; ++r) {
      for (int s = 0; s < sizes[WEIGHT_KERNEL_3D_S_IDX]; ++s) {
        for (int c = 0; c < sizes[WEIGHT_KERNEL_3D_C_IDX]; ++c) {
          for (int k = 0; k < sizes[WEIGHT_KERNEL_3D_K_IDX]; ++k) {
            tempBuff[buffer_counter] =
                ptr[k * strides[WEIGHT_KERNEL_3D_K_IDX] +
                    c * strides[WEIGHT_KERNEL_3D_C_IDX] +
                    r * strides[WEIGHT_KERNEL_3D_R_IDX] +
                    s * strides[WEIGHT_KERNEL_3D_S_IDX] +
                    q * strides[WEIGHT_KERNEL_3D_Q_IDX]];
            buffer_counter++;
          }
        }
      }
    }
  }

  // Copy tmp buffer to original tensor memory
  std::memcpy(ptr, tempBuff, weight.numel() * sizeof(T));
  delete[] tempBuff;
}

} // namespace habana_lazy