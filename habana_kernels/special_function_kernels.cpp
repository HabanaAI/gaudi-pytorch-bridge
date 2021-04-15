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

#define SPECIAL_FUNCTION_KERNEL(op_code)                     \
  STRINGIFY(aten::op_code),                                  \
      [](const int device_id, c10::ScalarType node_type) {   \
        return std::make_shared<SpecialFunctionFwdOperator>( \
            device_id, node_type, #op_code);                 \
      }

using namespace torch;
using namespace torch::jit;

void SpecialFunctionOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    Stack& inputs,
    bool is_output_persistent) {
  TORCH_CHECK(
      inputs.size() == 1, "Incorrect size of inputs expected for operator");
  TORCH_CHECK(inputs[0].isTensor(), "Input type expected to be tensor");

  at::Tensor input = inputs[0].toTensor();
  auto output = habana_helpers::createPTTensor(input, is_output_persistent);
  AllocateSynapseOutput(graph, output, is_output_persistent);
  AddNodeToSynapseGraph(graph, nullptr, 0);
}

static auto& KernelRegistry =
    ::habana::KernelRegistry()
        .add(
            "aten::asin",
            [](const int device_id, c10::ScalarType node_type) {
              return std::make_shared<SpecialFunctionFwdOperator>(
                  device_id, node_type, "asin");
            })
        .add(SPECIAL_FUNCTION_KERNEL(acos));
