/*******************************************************************************
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

#include "habana_lazy/permute_tensors.h"
#include <c10/core/Storage.h>
#include <cstddef>
#include "backend/synapse_helpers/layout_utils.h"
#include "habana_helpers/logging.h"
#include "habana_kernels/lazy_kernels.h"
#include "habana_kernels/lazy_kernels_declarations.h"
#include "habana_kernels/tensor_shape_kernels.h"
#include "habana_lazy/aten_lazy_bridge.h"
#include "habana_lazy/hpu_lazy_tensors.h"
#include "habana_lazy/lazy_executor.h"

using namespace synapse_helpers::layouts;
using namespace habana_lazy;

namespace habana_lazy {

unsigned PermuteTensors::m_permute_counter = 0;

void increasePermuteCount(torch::Tensor& weight) {
  HbLazyTensor hb_tensor = GetHbLazyTensor(weight);
  auto hb_data{hb_tensor.EvaluateTensorData()};
  auto tmeta{habana::get_tensor_extra_meta(hb_data)};
  tmeta->increase_permuted_counter();
}

void PermuteTensors::permuteWeight(torch::Tensor& weight) {
  PT_LAZY_TRACE;
  TORCH_CHECK(
      weight.device().type() == c10::DeviceType::HPU,
      "permuteWeight only for HPU tensors");

  habana_helpers::print_tensor_debug(weight);

  if (shouldPermuteWeight(weight)) {
    permuteWeightByDim(weight);
  } else if (shouldPermutePreCastedWeight(weight)) {
    auto pre_caster_weight = getPreCastedWeight(weight);
    permuteWeightByDim(pre_caster_weight);
  }
  habana_helpers::print_tensor_debug(weight);
}

void PermuteTensors::permuteWeightByDim(torch::Tensor& weight) {
  PT_LAZY_TRACE;
  HbLazyTensor hb_tensor = GetHbLazyTensor(weight);
  auto hb_data{hb_tensor.EvaluateTensorData()};
  auto tmeta{habana::get_tensor_extra_meta(hb_data)};
  if (tmeta->get_permuted_counter() >=
      GET_ENV_FLAG_NEW(PT_HPU_MAX_PERMUTE_THRESHOLD)) {
    PT_LAYOUTS_DEBUG(
        "Reached threshold permutations of ",
        GET_ENV_FLAG_NEW(PT_HPU_MAX_PERMUTE_THRESHOLD));
    return;
  }

  auto dim = weight.dim();
  if (GET_ENV_FLAG_NEW(PT_HPU_ENABLE_WEIGHT_HPU_PERMUTE)) {
    if (dim == 4 || dim == 5) {
      handleWeightTensorLayout(weight);
    } else {
      HABANA_ASSERT(false && "Permute weight support only 4/5D tensors");
    }
  }

  else if (GET_ENV_FLAG_NEW(PT_HPU_ENABLE_WEIGHT_CPU_PERMUTE)) {
    if (dim == 4) {
      habana_lazy::PermuteTensors::permuteWeightToRSCKInMemory(weight);
    } else if (dim == 5) {
      habana_lazy::PermuteTensors::permuteWeightToQRSCKInMemory(weight);
    } else {
      HABANA_ASSERT(false && "Permute weight support only 4/5D tensors");
    }
  }
}

namespace {
std::vector<int64_t> translateSynapsePermuteToPt(
    const std::vector<uint8_t>& synapse_permuate) {
  // first reverse vector and then change idx to mirror
  // NHWC 2013 -> 3102 -> 0231
  // RSCK 3201 -> 1023 -> 2310
  std::vector<int64_t> pt_permute(
      synapse_permuate.rbegin(), synapse_permuate.rend());
  for (size_t i = 0; i < synapse_permuate.size(); i++) {
    pt_permute[i] = pt_permute.size() - pt_permute[i] - 1;
  }
  return pt_permute;
}

std::vector<int64_t> calcNewStrides(
    const torch::Tensor& permutedTensor,
    const std::vector<int64_t>& pt_permute) {
  // compute strides based on new sizes after permute
  // permtue the strides
  std::vector<int64_t> new_sizes, new_strides;
  std::tie(new_sizes, new_strides) =
      PermuteOperator::compute_output_shape(permutedTensor, pt_permute);

  std::vector<int64_t> new_strides_perm(new_strides.size());
  for (size_t i = 0; i < new_strides.size(); i++) {
    new_strides_perm[pt_permute[i]] = new_strides[i];
  }
  return new_strides_perm;
}
} // namespace

void PermuteTensors::handlePermutedTensor(
    const torch::Tensor& permutedTensor,
    torch::Tensor& cpuTensor,
    bool non_blocking) {
  PT_LAZY_TRACE;
  habana_helpers::print_tensor_debug(permutedTensor);
  habana_helpers::print_tensor_debug(cpuTensor);
  TORCH_CHECK(
      permutedTensor.device().type() == c10::DeviceType::HPU,
      "handlePermutedTensor permutedTensor should be HPU");
  TORCH_CHECK(
      cpuTensor.device().type() == c10::DeviceType::CPU,
      "handlePermutedTensor cpuTensor should be CPU");

  auto synapse_permute = getMemoryPermutation(permutedTensor);
  if (synapse_permute.size() != 0) {
    if (non_blocking) {
      TORCH_CHECK(
          false, "handlePermutedTensor we only support non_blocking = false");
    }
    // translate synapse permtue to pt permute
    auto pt_permute = translateSynapsePermuteToPt(synapse_permute);
    // calculate new strides according to permutation
    auto strides = calcNewStrides(permutedTensor, pt_permute);
    auto old_sizes = cpuTensor.sizes();
    // set cpu tensor with old sizes + new strides
    cpuTensor.unsafeGetTensorImpl()->set_sizes_and_strides(old_sizes, strides);
    // permute tensor back to host
    cpuTensor = cpuTensor.contiguous();
  }
}

MemoryPermutation PermuteTensors::getMemoryPermutation(
    const torch::Tensor& tensor) {
  PT_LAZY_TRACE;
  TORCH_CHECK(
      tensor.device().type() == c10::DeviceType::HPU,
      "getMemoryPermutation tensor should be HPU");

  HbLazyTensor hb_tensor = GetHbLazyTensor(tensor);
  auto hb_data = hb_tensor.EvaluateTensorData();
  auto hb_impl = habana_lazy::GetHbInternalTensorImpl(hb_data);
  return hb_impl->GetMemoryPermutation();
}

void PermuteTensors::setMemoryPermutation(
    const torch::Tensor& tensor,
    MemoryPermutation permutation) {
  PT_LAZY_TRACE;
  TORCH_CHECK(
      tensor.device().type() == c10::DeviceType::HPU,
      "setMemoryPermutation tensor should be HPU");

  HbLazyTensor hb_tensor = GetHbLazyTensor(tensor);
  auto hb_data = hb_tensor.EvaluateTensorData();
  auto hb_impl = habana_lazy::GetHbInternalTensorImpl(hb_data);
  PT_LAZY_EAGER_DEBUG(
      "[LAZY EAGER SHAPE AGNOSTIC] Setting permute on HbInternal address : ",
      hb_impl,
      " storage address : ",
      hb_impl->data());
  hb_impl->SetMemoryPermutation(permutation);
}

void PermuteTensors::clearPermuteInformation(
    const torch::Tensor& permutedTensor) {
  PT_LAZY_TRACE;
  TORCH_CHECK(
      permutedTensor.device().type() == c10::DeviceType::HPU,
      "clearPermuteInformation permutedTensor should be HPU");
  setMemoryPermutation(permutedTensor, {});
}

void PermuteTensors::permuteWeightToRSCKInMemory(torch::Tensor& weight) {
  PT_LAZY_TRACE;
  PT_LAYOUTS_DEBUG("Permuting weight to RSCK, count: ", m_permute_counter++);
  torch::Tensor weight_cpu = weight.to(c10::kCPU);
  if (weight.scalar_type() == c10::ScalarType::BFloat16) {
    permuteWeightTensorDataToRSCK<c10::BFloat16>(weight_cpu);
  } else if (weight.scalar_type() == c10::ScalarType::Half) {
    permuteWeightTensorDataToRSCK<c10::Half>(weight_cpu);
  } else if (weight.scalar_type() == c10::ScalarType::Float) {
    permuteWeightTensorDataToRSCK<float>(weight_cpu);
  } else {
    HABANA_ASSERT(
        false && "PermuteWeightTensorDataToRSCK doesn't support used dtype");
  }
  copy_hpu_lazy_(weight, weight_cpu, false);

  habana_helpers::print_tensor_debug(weight);
  // Update Permutation
  increasePermuteCount(weight);
  setMemoryPermutation(weight, weight_rsck_in_memory);
}

void PermuteTensors::permuteWeightToQRSCKInMemory(torch::Tensor& weight) {
  PT_LAZY_TRACE;
  PT_LAYOUTS_DEBUG("Permuting weight to QRSCK, count: ", m_permute_counter++);
  torch::Tensor weight_cpu = weight.to(c10::kCPU);
  if (weight.scalar_type() == c10::ScalarType::BFloat16) {
    restrideWeightTensorDataToQRSCK<c10::BFloat16>(weight_cpu);
  } else if (weight.scalar_type() == c10::ScalarType::Half) {
    restrideWeightTensorDataToQRSCK<c10::Half>(weight_cpu);
  } else if (weight.scalar_type() == c10::ScalarType::Float) {
    restrideWeightTensorDataToQRSCK<float>(weight_cpu);
  } else {
    HABANA_ASSERT(
        false && "PermuteWeightTensorDataToQRSCK doesn't support used dtype");
  }

  copy_hpu_lazy_(weight, weight_cpu, false);

  habana_helpers::print_tensor_debug(weight);
  // Update lazy & impl status
  increasePermuteCount(weight);
  setMemoryPermutation(weight, weight_qrsck_in_memory);
}

bool PermuteTensors::shouldPermuteWeight(const torch::Tensor& weight) {
  PT_LAZY_TRACE;
  HbLazyTensor weight_hb_tensor = GetHbLazyTensor(weight);
  habana_lazy::AccThread::Get().SyncAccThreadPool();

  PT_LAYOUTS_DEBUG(
      "shouldPermuteWeight tensor: ", weight_hb_tensor.getTensorUniqueId())
  bool is_input = weight_hb_tensor.CurrentIrValue().IsHpuInputNode();
  if (!is_input) {
    PT_LAYOUTS_DEBUG("shouldPermuteWeight is not input to the graph");
    return false;
  }
  // Don't access tensor impl if not graph input
  auto current_permutaion = getMemoryPermutation(weight);
  auto required_permute = weight.dim() == 4
      ? synapse_helpers::layouts::weight_rsck_in_memory
      : synapse_helpers::layouts::weight_qrsck_in_memory;
  bool is_permuted = current_permutaion == required_permute;
  if (is_permuted) {
    PT_LAYOUTS_DEBUG(
        "shouldPermuteWeight already permuted to: ",
        VecToString(required_permute));
  }
  PT_LAZY_EAGER_DEBUG(
      "[LAZY EAGER SHAPE AGNOSTIC] shouldPermuteWeight tensor: ",
      weight_hb_tensor.getTensorUniqueId(),
      " is_permuted : ",
      is_permuted,
      " curr perm : ",
      VecToString(current_permutaion),
      " req perm : ",
      VecToString(required_permute));
  return is_input && !is_permuted;
}

bool PermuteTensors::shouldPermutePreCastedWeight(const torch::Tensor& weight) {
  PT_LAZY_TRACE;
  HbLazyTensor weight_hb_tensor = GetHbLazyTensor(weight);
  habana_lazy::AccThread::Get().SyncAccThreadPool();

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
  PT_LAZY_TRACE;
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
  auto context =
      get_device_lazy_execution_context(copy_to_cpu.device().index());
  context->MarkTensorStatus(
      hl_copy_to_cpu.getDataPtr(), LazyTensorExecutionStatus::kINPUT);
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
