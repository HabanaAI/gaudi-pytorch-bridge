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

#include "habana_eager/graph_weight_permute.h"
#include "backend/helpers/eager_pipeline.h"
#include "habana_eager/eager_context.h"
#include "habana_eager/eager_pipeline_utils.h"
#include "habana_eager/graph_exec.h"
#include "habana_eager/ops/copy_from.h"
#include "habana_helpers/logging.h"

using namespace synapse_helpers::layouts;

namespace habana {
namespace graph {

PermuteWeightTensor::PermuteWeightTensor(const torch::Tensor& weight)
    : m_weight(weight),
      m_tensor_dim(weight.dim()),
      m_storage_meta(habana::get_storage_extra_meta(m_weight)) {}

void PermuteWeightTensor::PermuteIfNeeded() {
  if (ShouldPermuteWeight()) {
    PT_EAGER_TRACE;
    MemoryPermutation new_permutation;

    torch::Tensor weight_cpu{m_weight.to(c10::kCPU)};
    c10::ScalarType dtype{weight_cpu.scalar_type()};
    HABANA_ASSERT(
        dtype == c10::ScalarType::BFloat16 || dtype == c10::ScalarType::Float ||
            dtype == c10::ScalarType::Half,
        "Unsupported dtype for weight permutation.");

    if (m_tensor_dim == 4) {
      PT_EAGER_DEBUG("Performing weight permutation to RSCK");
      if (dtype == c10::ScalarType::BFloat16) {
        PermuteDataToRSCK<c10::BFloat16>(weight_cpu);
      }
      if (dtype == c10::ScalarType::Float) {
        PermuteDataToRSCK<float>(weight_cpu);
      }
      if (dtype == c10::ScalarType::Half) {
        PermuteDataToRSCK<c10::Half>(weight_cpu);
      }

      new_permutation = weight_rsck_in_memory;
    }

    if (m_tensor_dim == 5) {
      PT_EAGER_DEBUG("Performing weight permutation to QRSCK");
      if (weight_cpu.scalar_type() == c10::ScalarType::BFloat16) {
        PermuteDataToQRSCK<c10::BFloat16>(weight_cpu);
      }
      if (dtype == c10::ScalarType::Float) {
        PermuteDataToQRSCK<float>(weight_cpu);
      }
      if (dtype == c10::ScalarType::Half) {
        PermuteDataToQRSCK<float>(weight_cpu);
      }
      new_permutation = weight_qrsck_in_memory;
    }

    habana::eager::_copy_from(weight_cpu, m_weight, false);
    auto pipeline_or_direct_set_memory_permutation =
        [](MemoryPermutation new_permutation,
           habana::StorageExtraMeta* m_storage_meta) {
          m_storage_meta->set_memory_permutation(new_permutation);
        };
    habana::eager::pipeline_or_direct_generic(
        pipeline_or_direct_set_memory_permutation,
        std::move(new_permutation),
        m_storage_meta);
  }
}

bool PermuteWeightTensor::ShouldPermuteWeight() {
  // For conv1d (tensor_dim == 3) permutation is not needed
  if (m_tensor_dim <= 3) {
    return false;
  }
  HABANA_ASSERT(
      m_tensor_dim == 4 || m_tensor_dim == 5,
      "Unexpected tensor dimensions: ",
      m_tensor_dim);

  auto tmeta{habana::get_tensor_extra_meta(m_weight)};
  if (tmeta == nullptr)
    return false;
  if (tmeta->is_view_tensor())
    return false;

  MemoryPermutation current_perm{m_storage_meta->get_memory_permutation()};
  MemoryPermutation required_perm{
      (m_tensor_dim == 4) ? weight_rsck_in_memory : weight_qrsck_in_memory};
  if (0 == current_perm.size()) {
    return true;
  }
  HABANA_ASSERT(
      current_perm == required_perm, "Unexpected permutation of weight tensor");
  return false;
}

template <typename T>
void PermuteWeightTensor::PermuteDataToRSCK(const torch::Tensor& weight_cpu) {
  PT_EAGER_TRACE;
  T* ptr = (T*)weight_cpu.data_ptr();
  auto strides = weight_cpu.strides();
  auto sizes = weight_cpu.sizes();

  // Creating temp buffer the size of the tensor
  std::vector<T> tempBuff;
  tempBuff.resize(weight_cpu.numel());
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
  std::memcpy(ptr, tempBuff.data(), weight_cpu.numel() * sizeof(T));
}

template <typename T>
void PermuteWeightTensor::PermuteDataToQRSCK(const torch::Tensor& weight_cpu) {
  PT_EAGER_TRACE;
  T* ptr = (T*)weight_cpu.data_ptr();
  auto strides = weight_cpu.strides();
  auto sizes = weight_cpu.sizes();

  // Creating temp buffer the size of the tensor
  std::vector<T> tempBuff;
  tempBuff.resize(weight_cpu.numel());
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
  std::memcpy(ptr, tempBuff.data(), weight_cpu.numel() * sizeof(T));
}

} // namespace graph
} // namespace habana