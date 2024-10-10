/*******************************************************************************
 * Copyright (C) 2024 Habana Labs, Ltd. an Intel Company
 * All Rights Reserved.
 *
 * Unauthorized copying of this file or any element(s) within it, via any medium
 * is strictly prohibited.
 * This file contains Habana Labs, Ltd. proprietary and confidential information
 * and is subject to the confidentiality and license agreements under which it
 * was provided.
 *
 *******************************************************************************/

#include "random.h"
#include <ATen/CPUGeneratorImpl.h>
#include <ATen/core/Generator.h>
#include "habana_kernels/lazy_kernels.h"
#include "habana_lazy/lazy_executor.h"
namespace habana {
namespace detail {
// Getting the HPU worker generator instance
at::Generator& getDefaultHPUGenerator() {
  static auto default_gen_hpu = createHPUGenerator();
  return default_gen_hpu;
}

// Utility to create a CPUGeneratorImpl. Returns a shared_ptr
at::Generator createHPUGenerator() {
  auto default_cpu_gen = at::detail::getDefaultCPUGenerator();
  auto gen =
      at::make_generator<at::CPUGeneratorImpl>(default_cpu_gen.current_seed());
  return gen;
}
} // namespace detail

uint32_t get_seed_hpu(const c10::optional<at::Generator>& gen) {
  at::CPUGeneratorImpl* generator =
      at::get_generator_or_default<at::CPUGeneratorImpl>(
          gen, detail::getDefaultHPUGenerator());

  auto context = habana_lazy::get_device_lazy_execution_context();
  if (context->getDryRun()) {
    return 0;
  }
  // Acquire lock when using random generators
  std::lock_guard<std::mutex> lock(generator->mutex_);
  return generator->random();
}

at::Tensor get_seed_tensor_hpu(const c10::optional<at::Generator>& gen) {
  int seed = get_seed_hpu(gen);
  at::Tensor seed_tensor = at::tensor(seed);
  auto t = habana_lazy::append_to_batch_h2d_list(seed_tensor);
  auto context = habana_lazy::get_device_lazy_execution_context();
  if (context->getCapturing()) {
    habana_lazy::HbLazyTensor hb_tensor = habana_lazy::GetHbLazyTensor(t);
    hb_tensor.getDataPtr()->is_random_seed_tensor = true;
    context->getSeedTensorMap()[hb_tensor.getDataPtr()->unique_id] = gen;
  }
  return t;
}

} // namespace habana
