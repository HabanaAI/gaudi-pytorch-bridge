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

// Incase of F32 to BF16 use round to nearest even. Use truncation for all F32
// to any int type conversion. Rounding mode is not relevant for other cases
// like integral  to integral or integral to floating type conversion.
#define SET_CAST_ROUNDING_MODE(guid)                            \
  params.round_mode = (guid == std::string("cast_f32_to_bf16")) \
      ? CAST_ROUND_HALF_NE                                      \
      : CAST_ROUND_ZERO;
namespace habana_helpers {
extern const std::
    map<std::pair<c10::ScalarType, c10::ScalarType>, c10::ScalarType>
        promote_dtype;
extern std::map<std::pair<c10::ScalarType, c10::ScalarType>, std::string>
    cast_map;

void type_promotion_for_two_tensor_inputs(
    std::vector<at::IValue>& inputs,
    int& pos,
    c10::ScalarType& dst_dtype);

std::vector<int64_t> compute_broadcast_shape(
    const at::Tensor& arg1,
    const at::Tensor& arg2);

std::string unique_recipe_name_generator(std::string recipe_name);

void compile_and_run(
    synapse_helpers::graph&& graph,
    const std::vector<std::string>& input_names,
    const std::vector<std::string>& output_names,
    const std::vector<void*>& input_buffers,
    const std::vector<void*>& output_buffers,
    std::vector<synapse_helpers::device_ptr> in_event_addr,
    std::vector<synapse_helpers::device_ptr> out_event_addr,
    std::vector<at::Tensor>& pt_inputs,
    const uint32_t device_id,
    size_t key = 0);

void execute_recipe(
    const std::vector<void*>& input_buffers,
    const std::vector<void*>& output_buffers,
    std::vector<synapse_helpers::device_ptr> in_event_addr,
    std::vector<synapse_helpers::device_ptr> out_event_addr,
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

 protected:
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
        inputs.size() == 6,
        "OnesLikeOperator Operation expects 6 arguments as input")
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
    kernel_meta_data_.output_layout.assign({habana::LayoutFormat::ANY});
    // special case, adding -1 to the tpc order, will not add any inputs
    kernel_meta_data_.tpc_input_order = {habana::NO_INPUTS};
  }

  virtual void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      bool is_output_persistent = false) override;
};
