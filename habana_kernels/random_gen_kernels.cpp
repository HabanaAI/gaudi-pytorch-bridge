/******************************************************************************
 * Copyright (C) 2020 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */

#include <ATen/ExpandUtils.h>
#include <perf_lib_layer_params.h>
#include <torch/script.h>
#include <memory>

#include "habana_device/HPUCheck.h"
#include "habana_device/hpu_cached_devices.h"
#include "habana_helpers/tensor_utils.h"
#include "habana_helpers/unused_macro.h"
#include "habana_kernels/kernel_utils.h"
#include "habana_kernels/simple_generic_kernel.h"

using namespace torch;

/*******************************************************************
*@brief Implements uniform distribution generation kernel
*INPUTS
@param[in, out] self - output tensor with uniform distributed values, 2D/3D/4D,
bf16/FP32
@param[in] from - lower bound
@param[in] to - upper bound
@param[in] gen - Generator class for seed (optional)
*******************************************************************/
void uniform_hpu(
    const Tensor& self,
    double from = 0,
    double to = 1,
    Generator* gen = nullptr) {
  LOG_FUNC_BEGIN;
  uint64_t seed;
  uint32_t seed_u32;

  if (gen == nullptr) {
    seed = at::detail::getNonDeterministicRandom(false);
  } else {
    // Acquire lock when using random generators
    std::lock_guard<std::mutex> lock(gen->mutex_);
    seed = gen->current_seed();
  }

  // Convert to 32 bit as TPC kernel supports only 32 bit seed
  seed_u32 = (uint32_t)(seed & 0xFFFFFFFF);

  std::vector<const at::Tensor*> pt_inputs{};
  std::vector<const at::Tensor*> pt_outputs{&self};

  ns_RandomUniform::Params params;
  params.low = static_cast<float>(from);
  params.high = static_cast<float>(to);
  params.seed = seed_u32;

  synapse_simple_generic_kernel(
      pt_outputs,
      pt_inputs,
      "random_uniform",
      &params,
      sizeof(params),
      SynapsePassType::FORWARD_PASS);

  LOG_FUNC_END;
}

/*******************************************************************
*@brief Implements normal distribution generation kernel
*INPUTS
@param[in, out] self - output tensor with normal distributed values, 2D/3D/4D,
bf16/FP32
@param[in] mean, default = 0
@param[in] std, default = 1
@param[in] gen - Generator class for seed (optional)
*******************************************************************/
void normal_hpu(
    const Tensor& self,
    double mean = 0,
    double std = 1,
    Generator* gen = nullptr) {
  LOG_FUNC_BEGIN;
  uint64_t seed;
  uint32_t seed_u32;

  if (gen == nullptr) {
    seed = at::detail::getNonDeterministicRandom(false);
  } else {
    // Acquire lock when using random generators
    std::lock_guard<std::mutex> lock(gen->mutex_);
    seed = gen->current_seed();
  }

  // Convert to 32 bit as TPC kernel supports only 32 bit seed
  seed_u32 = (uint32_t)(seed & 0xFFFFFFFF);

  std::vector<const at::Tensor*> pt_inputs{};
  std::vector<const at::Tensor*> pt_outputs{&self};

  ns_RandomNormal::Params params;
  params.mean = static_cast<float>(mean);
  params.stddev = static_cast<float>(std);
  params.seed = seed_u32;

  synapse_simple_generic_kernel(
      pt_outputs,
      pt_inputs,
      "random_normal",
      &params,
      sizeof(params),
      SynapsePassType::FORWARD_PASS);

  LOG_FUNC_END;
}

static auto registry =
    torch::RegisterOperators()
        .op(torch::RegisterOperators::options()
                .schema(
                    "aten::uniform_(Tensor(a!) self, float from=0, float to=1, *, Generator? generator=None) -> Tensor(a!)")
                .impl_unboxedOnlyKernel<decltype(uniform_hpu), &uniform_hpu>(
                    DispatchKey::HABANATensorId)
                .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
        .op(torch::RegisterOperators::options()
                .schema(
                    "aten::normal_(Tensor(a!) self, float mean=0, float std=1, *, Generator? generator=None) -> Tensor(a!)")
                .impl_unboxedOnlyKernel<decltype(normal_hpu), &normal_hpu>(
                    DispatchKey::HABANATensorId)
                .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA));
