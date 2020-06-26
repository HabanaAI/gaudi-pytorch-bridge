/******************************************************************************
 * Copyright (C) 2020 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */
#pragma once

#include <c10/util/ArrayRef.h>
#include <synapse_api.h>
#include <synapse_api_types.h>
#include <synapse_helpers/habana_tensor.h>
#include <torch/script.h>
#include <string>
#include <unordered_map>

#include <perf_lib_layer_params.h>
#include <synapse_helpers/graph.h>
#include <synapse_helpers/recipe.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/runtime/interpreter.h>
#include "habana_helpers/logging.h"
#include "habana_kernels/habana_operator.h"

namespace habana_helpers {
std::string unique_recipe_name_generator(std::string recipe_name);

[[deprecated]] void compile_and_run(
    const std::string& recipe_prefix,
    const synGraphHandle graph_handle,
    const std::vector<std::string>& input_names,
    const std::vector<std::string>& output_names,
    const std::vector<void*>& input_buffers,
    const std::vector<void*>& output_buffers,
    const uint32_t device_id);

void compile_and_run(
    synapse_helpers::graph&& graph,
    const std::vector<std::string>& input_names,
    const std::vector<std::string>& output_names,
    const std::vector<void*>& input_buffers,
    const std::vector<void*>& output_buffers,
    const uint32_t device_id,
    size_t key = 0);

std::vector<synLaunchTensorInfo> generate_syn_launch_tensor_info(
    const std::vector<std::string>& in_names,
    const std::vector<void*>& in_buffers,
    const std::vector<std::string>& out_names,
    const std::vector<void*>& out_buffers);

void execute_recipe(
    const std::vector<void*>& input_buffers,
    const std::vector<void*>& output_buffers,
    const uint32_t device_id,
    size_t key);

size_t getRecipeKey(
    std::string node,
    std::vector<c10::IValue> stack,
    bool inPlaceOp = false);
} // namespace habana_helpers


namespace habana{
//
// Cast Operator
class CastOperator : public HabanaOperator {
 public:
  CastOperator(int device_id, const std::string& guid)
      : HabanaOperator(guid) {
    this->CreateSynContext(device_id);
    kernel_meta_data_.input_layout.assign({LayoutFormat::ANY, LayoutFormat::ANY});
    kernel_meta_data_.output_layout.assign({LayoutFormat::ANY});

  }

  virtual void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      bool is_output_persistent = false) override;

 private:
  ns_CastKernel::Params synapse_cast_params_builder();
};
} // namespace habana