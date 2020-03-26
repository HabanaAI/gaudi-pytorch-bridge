/******************************************************************************
 * Copyright (C) 2020 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */
#include <torch/script.h>

#include "habana_device/HPUCheck.h"
#include "habana_device/HPUContext.h"
#include "habana_device/hpu_cached_devices.h"
#include "habana_helpers/tensor_utils.h"
#include "habana_kernels/kernel_utils.h"
#include "habana_kernels/simple_generic_kernel.h"

using namespace torch;

Tensor relu_hpu(const Tensor& input) {
  LOG_FUNC_BEGIN;
  auto output = at::empty(input.sizes(), input.options());
  std::vector<const at::Tensor*> pt_outputs{&output};
  std::vector<const at::Tensor*> pt_inputs{&input};

  synapse_simple_generic_kernel(
      pt_outputs, pt_inputs, "relu", nullptr, 0, true);

  LOG_FUNC_END;
  return output;
}

Tensor& relu_hpu_(Tensor& self) {
  LOG_FUNC_BEGIN;
  std::vector<const at::Tensor*> pt_inputs{&self};

  synapse_simple_generic_inplace_kernel(pt_inputs, "relu", nullptr, 0, true);

  LOG_FUNC_END;
  return self;
}

static auto registry =
    torch::RegisterOperators()
        .op(torch::RegisterOperators::options()
                .schema("aten::relu_(Tensor(a!) self) -> Tensor(a!)")
                .impl_unboxedOnlyKernel<decltype(relu_hpu_), &relu_hpu_>(
                    TensorTypeId::HABANATensorId)
                .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
        .op(torch::RegisterOperators::options()
                .schema("aten::relu(Tensor self) -> Tensor")
                .impl_unboxedOnlyKernel<decltype(relu_hpu), &relu_hpu>(
                    TensorTypeId::HABANATensorId)
                .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA));