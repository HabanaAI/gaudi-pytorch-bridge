/******************************************************************************
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
#pragma once
#include <torch/torch.h>

template <class T>
const at::Tensor* convert_to_type_supported_on_cpu(
    const at::Tensor& tin,
    at::Tensor& storage) {
  if (std::is_same_v<T, float>) {
    switch (tin.scalar_type()) {
      case torch::kBFloat16:
      case torch::kFloat16:
        storage = tin.to(torch::kFloat32);
        return &storage;
      default:
        break;
    }
  }
  return &tin;
}

template <class T>
void dump_tensor(
    const std::string& label,
    const at::Tensor& tin,
    bool verbose) {
  if (verbose) {
    at::Tensor storage;
    const auto* t = convert_to_type_supported_on_cpu<T>(tin, storage);
    auto ptr = (T*)t->data_ptr();
    std::cout << label << " shape = " << tin.sizes() << std::endl;
    for (size_t i = 0; i < t->numel(); ++i) {
      std::cout << i << " : " << ptr[i] << std::endl;
    }
  }
}

template <class T>
void dump_tensors(
    const std::string& label,
    const at::Tensor& t1in,
    const at::Tensor& t2in,
    bool verbose) {
  if (verbose) {
    std::array<at::Tensor, 2> storage;
    const auto* t1 = convert_to_type_supported_on_cpu<T>(t1in, storage[0]);
    const auto* t2 = convert_to_type_supported_on_cpu<T>(t2in, storage[1]);

    auto ptr1 = (T*)t1->data_ptr();
    auto ptr2 = (T*)t2->data_ptr();
    std::cout << label << " shapes = " << t1in.sizes() << ", " << t2in.sizes()
              << std::endl;
    for (size_t i = 0; (i < t1->numel()) && (i < t2->numel()); ++i) {
      auto d = fabs(ptr1[i] - ptr2[i]);
      auto r = ptr1[i] ? d / abs(ptr1[i]) : INFINITY;
      std::cout << i << " : " << ptr1[i] << " vs " << ptr2[i] << " D = " << d
                << " R = " << r << std::endl;
    }
  }
}

struct TensorAndView {
  torch::Tensor t;
  torch::Tensor view;
};

std::vector<torch::Tensor> TensorAndViewVecToViewVec(
    const std::vector<TensorAndView>&);

template <class T>
bool CompareTensors(
    const std::string& label,
    int id,
    const TensorAndView& hpu,
    const TensorAndView& cpu,
    bool verbose,
    float atol,
    float rtol) {
  auto hpu_on_cpu = hpu.t.to(torch::kCPU);
  dump_tensors<T>(
      label + "[" + std::to_string(id) + "]", hpu_on_cpu, cpu.t, verbose);
  return cpu.t.allclose(hpu_on_cpu, atol, rtol);
};

template <class U>
const TensorAndView& AccessTensorForCompareFewTensors(
    const U& v,
    std::vector<TensorAndView> U::*pmTensorAndViewVec,
    int idInVec) {
  return (v.*pmTensorAndViewVec)[idInVec];
}

template <class U, class V>
const TensorAndView& AccessTensorForCompareFewTensors(
    const U& v,
    const std::pair<V, int>& pair,
    int) {
  return AccessTensorForCompareFewTensors(v, pair.first, pair.second);
}

template <class T, class U, class V, class... Args>
bool CompareRecursiveForCompareFewTensors(
    bool equalSoFar,
    int id,
    const U& hpu,
    const U& cpu,
    bool verbose,
    float atol,
    float rtol,
    const std::string& label,
    V pmTensorVecOrPairWithIndex,
    Args&&... args) {
  auto equal = CompareTensors<T>(
      label,
      id,
      AccessTensorForCompareFewTensors(hpu, pmTensorVecOrPairWithIndex, id),
      AccessTensorForCompareFewTensors(cpu, pmTensorVecOrPairWithIndex, id),
      verbose,
      atol,
      rtol);
  // Don't shorten to equal = equalSoFar && CompareTensors(...) as we want
  // CompareTensors() is executed even if equalSoFar is false beforehand, for
  // logging purpose.
  return CompareRecursiveForCompareFewTensors<T>(
      equalSoFar && equal,
      id,
      hpu,
      cpu,
      verbose,
      atol,
      rtol,
      std::forward<Args>(args)...);
}

template <class T, class U>
bool CompareRecursiveForCompareFewTensors(
    bool equal,
    int id,
    const U& hpu,
    const U& cpu,
    bool verbose,
    float atol,
    float rtol) {
  return equal;
}

template <class T, class... Args>
bool CompareFewTensors(Args&&... args) {
  return CompareRecursiveForCompareFewTensors<T>(
      true, std::forward<Args>(args)...);
}

template <class T>
void PushBackHpuAndCpuTensors(
    torch::Tensor src,
    T& hpu,
    T& cpu,
    std::vector<TensorAndView> T::*pmTensorAndViewVec,
    bool onHpuMakeView) {
  if (onHpuMakeView) {
    auto src_flat = src.flatten();
    (cpu.*pmTensorAndViewVec).push_back({src_flat, src_flat});
    auto src_flat_on_hpu = src_flat.to(torch::kHPU);
    auto src_on_hpu = src_flat_on_hpu.view(src.sizes());
    (hpu.*pmTensorAndViewVec).push_back({src_flat_on_hpu, src_on_hpu});
  } else {
    (cpu.*pmTensorAndViewVec).push_back({src, src});
    auto src_on_hpu = src.to(torch::kHPU);
    (hpu.*pmTensorAndViewVec).push_back({src_on_hpu, src_on_hpu});
  }
}
