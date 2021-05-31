/******************************************************************************
 * Copyright (C) 2021 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */
#include "habana_kernels/special_function_kernels.h"
#include <perf_lib_layer_params.h>
#include <torch/script.h>
#define STRINGIFY(op_code) #op_code

#define SPECIAL_FUNCTION_KERNEL_WRAP(op_code, inplace)       \
  STRINGIFY(aten::op_code),                                  \
      [](const int device_id, c10::ScalarType node_type) {   \
        return std::make_shared<SpecialFunctionFwdOperator>( \
            device_id, node_type, #op_code, inplace);        \
      }
#define SPECIAL_FUNCTION_INPLACE_KERNEL(op_code) \
  SPECIAL_FUNCTION_KERNEL_WRAP(op_code, true)
#define SPECIAL_FUNCTION_KERNEL(op_code) \
  SPECIAL_FUNCTION_KERNEL_WRAP(op_code, false)

using namespace torch;
using namespace torch::jit;
using namespace habana;

void SpecialFunctionFwdOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    Stack& inputs,
    bool is_output_persistent) {
  TORCH_CHECK(
      inputs.size() == 1, "Incorrect size of inputs expected for operator");
  TORCH_CHECK(inputs[0].isTensor(), "Input type expected to be tensor");

  at::Tensor input = inputs[0].toTensor();

  if (!m_inplace) {
    auto output = habana_helpers::createPTTensor(input, is_output_persistent);
    AllocateSynapseOutput(graph, output, is_output_persistent);
  } else {
    if (p_context_->pt_inputs_.size() == 0)
      p_context_->pt_inputs_.emplace_back(inputs[0].toTensor());
    AllocateSynapseInplaceOutput(graph);
  }
  AddNodeToSynapseGraph(graph, nullptr, 0);
}

static auto& KernelRegistry =
    habana::KernelRegistry()
        .add(
            "aten::asin",
            [](const int device_id, c10::ScalarType node_type) {
              return std::make_shared<SpecialFunctionFwdOperator>(
                  device_id, node_type, "asin", false);
            })
        .add(SPECIAL_FUNCTION_KERNEL(acos))
        .add(SPECIAL_FUNCTION_KERNEL(acosh))
        .add(SPECIAL_FUNCTION_KERNEL(asinh))
        .add(SPECIAL_FUNCTION_KERNEL(atan))
        .add(SPECIAL_FUNCTION_KERNEL(atanh))
        .add(SPECIAL_FUNCTION_KERNEL(cosh))
        .add(SPECIAL_FUNCTION_INPLACE_KERNEL(acos_))
        .add(SPECIAL_FUNCTION_INPLACE_KERNEL(acosh_))
        .add(SPECIAL_FUNCTION_INPLACE_KERNEL(asinh_))
        .add(SPECIAL_FUNCTION_INPLACE_KERNEL(atan_))
        .add(SPECIAL_FUNCTION_INPLACE_KERNEL(atanh_))
        .add(SPECIAL_FUNCTION_INPLACE_KERNEL(cos_))
        .add(SPECIAL_FUNCTION_INPLACE_KERNEL(cosh_))
        .add(SPECIAL_FUNCTION_INPLACE_KERNEL(tanh_));
