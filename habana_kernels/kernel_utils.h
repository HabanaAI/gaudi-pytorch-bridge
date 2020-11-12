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

void compile_and_run(
    synapse_helpers::graph&& graph,
    const std::vector<std::string>& input_names,
    const std::vector<std::string>& output_names,
    const std::vector<void*>& input_buffers,
    const std::vector<void*>& output_buffers,
    std::vector<at::Tensor>& pt_inputs,
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
    std::vector<at::Tensor>& pt_inputs,
    const uint32_t device_id,
    size_t key);

size_t getRecipeKey(
    std::string node,
    std::vector<c10::IValue> stack,
    bool inPlaceOp = false,
    bool outOp = false);
} // namespace habana_helpers

// CastOut Operator
class CastOutOperator : public habana::HabanaOperator {
 public:
  CastOutOperator(int device_id, const std::string& guid)
      : HabanaOperator(guid) {
    this->CreateSynContext(device_id);
    kernel_meta_data_.input_layout.assign(
        {habana::LayoutFormat::ANY, habana::LayoutFormat::ANY});
    kernel_meta_data_.output_layout.assign({habana::LayoutFormat::ANY});
  }

  virtual void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      bool is_output_persistent = false) override;

 private:
  ns_CastKernel::Params synapse_cast_params_builder();
};

// Cast Operator
class CastOperator : public CastOutOperator {
 public:
  CastOperator(int device_id, const std::string& guid)
      : CastOutOperator(device_id, guid) {
    kernel_meta_data_.input_layout.assign({habana::LayoutFormat::ANY});
  }

  virtual void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      bool is_output_persistent = false) override;
};

// Constant Operator
class ConstantOperator : public habana::HabanaOperator {
 public:
  ConstantOperator(int device_id, c10::ScalarType scalarType)
      : habana::HabanaOperator(
            "constant_" + habana_helpers::name_suffix_from_type(scalarType)) {
    this->CreateSynContext(device_id);
    kernel_meta_data_.output_layout.assign({habana::LayoutFormat::ANY});
  }

  virtual void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      bool is_output_persistent = false) override;
};


//
// For suporting ones_like operation
class OnesLikeOperator : public ConstantOperator {
 public:
  OnesLikeOperator(int device_id, c10::ScalarType scalarType)
      : ConstantOperator(device_id, scalarType) {}
  virtual void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      bool is_output_persistent) {
    TORCH_CHECK(
        inputs.size() == 6, "OnesLikeOperator Operation expects 6 arguments as input")
    inputs.erase(inputs.begin() + 1, inputs.end());
    inputs.emplace_back(1);
    ConstantOperator::AllocateAndAddSynapseNode(
        graph, inputs, is_output_persistent);
  }
};

// ConstantOut Operator
class ConstantOutOperator : public habana::HabanaOperator {
 public:
  ConstantOutOperator(int device_id, c10::ScalarType scalarType)
      : habana::HabanaOperator(
            "constant_" + habana_helpers::name_suffix_from_type(scalarType)) {
    this->CreateSynContext(device_id);
    // Temporary WA for MNIST Lazy Execution only.
    // Assume ConstantOut will be called from fill_ only, which is only
    // being used to Zero out weight gradients, therefore we can return
    // output_layout as "HWCK" instead of "ANY"
    kernel_meta_data_.output_layout.assign({habana::LayoutFormat::HWCK});
    // special case, adding -1 to the tpc order, will not add any inputs
    kernel_meta_data_.tpc_input_order = {habana::NO_INPUTS};
  }

  virtual void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      bool is_output_persistent = false) override;
};
