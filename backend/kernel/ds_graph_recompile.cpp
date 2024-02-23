/*******************************************************************************
 * Copyright (C) 2021-2023 Habana Labs, Ltd. an Intel Company
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

#include "backend/kernel/ds_graph_recompile.h"
#include "backend/backend_meta.h"
#include "backend/kernel/hpu_habana_cache.h"

#include "habana_lazy/aten_lazy_bridge.h"
#include "habana_lazy/hlexec.h"
#include "habana_lazy/hpu_lazy_tensors.h"
#include "habana_lazy/lazy_executor.h"
#include "pytorch_helpers/visualize/visualize.h"

#include "habana_kernels/lazy_kernels.h"

#include "backend/helpers/tensor_info.h"
#include "habana_helpers/logging.h"

std::mutex habana::DynamicBucketInfoMap::mutex_;

at::Tensor habana::CreateEmptyTensor(
    const PtTensorInfo& ti,
    habana::ShapeTensorStruct& tensor_data,
    const std::vector<int64_t>& tshape) {
  if (ti.tensor_type() == SHAPE_TENSOR) {
    auto pt_tensor = habana_lazy::empty_hpu_lazy(
        tshape, ti.get_topts(), ti.get_mf(), false, SHAPE_TENSOR);
    if (tensor_data.has_shape_tensor_data()) {
      auto new_tmeta{get_tensor_extra_meta(pt_tensor)};
      HABANA_ASSERT(new_tmeta);
      new_tmeta->get_shape_struct() = tensor_data;
    }
    return pt_tensor;
  }

  auto pt_tensor = at::empty(tshape, ti.get_topts(), ti.get_mf());
  return pt_tensor;
}

torch::jit::Stack habana::CreateInputStack(
    std::shared_ptr<habana::RecipeValueSpec> rvpsh,
    std::unordered_map<uint64_t, habana::ShapeTensorStruct>& input_metadata,
    habana_helpers::TensorShapes& input_shapes) {
  PT_BRIDGE_BEGIN;
  torch::jit::Stack new_input_stack;

  PT_DYNAMIC_SHAPE_DEBUG(
      "Number of graph inputs ",
      rvpsh->num_inputs,
      ", number of inputs with impl data:",
      input_metadata.size());
  for (size_t tidx = 0; tidx < rvpsh->num_inputs; tidx++) {
    auto& ti = rvpsh->dtensorinfos.at(tidx);
    TORCH_CHECK(
        input_shapes.count(tidx),
        "Tensor index ",
        tidx,
        "is missing from ",
        input_shapes);
    habana::ShapeTensorStruct tensor_data;
    if (input_metadata.count(tidx)) {
      tensor_data = input_metadata[tidx];
    }

    auto pt_input = habana::CreateEmptyTensor(
        *ti, tensor_data, input_shapes.at(tidx).get_dims());
    new_input_stack.push_back(torch::jit::IValue(pt_input));
  }
  PT_BRIDGE_END;
  return new_input_stack;
}

void habana::PrintStack(torch::jit::Stack& st) {
  PT_BRIDGE_DEBUG("aten_inputs #", st.size(), "::");
  for (size_t idx = 0; idx < st.size(); idx++) {
    PT_BRIDGE_DEBUG(habana_helpers::DebugString(st.at(idx)));
  }
}

bool habana::RefineBucketDS(size_t graph_key) {
  bool is_refined{true};
  DynamicBucketInfoMap::get_instance().refine_graph(graph_key);
  return is_refined;
}

bool habana::CompileGraphWithRange(
    std::shared_ptr<habana::RecipeValueSpec> rvpsh,
    std::unordered_map<uint64_t, habana::ShapeTensorStruct>& input_metadata,
    habana_helpers::ResultShapes& input_ranges,
    habana_helpers::Bucket& new_bucket,
    size_t& new_recipe_key,
    std::shared_ptr<habana_helpers::CompilationStatistics> statpsh,
    std::shared_ptr<habana_helpers::DynamicBucketInfo> dbipsh) {
  bool ret{true};

  PT_DYNAMIC_SHAPE_DEBUG(
      "BucketRefinement: Will use the following recipe for compilation",
      rvpsh->header_str());

  torch::jit::Stack input_stack =
      habana::CreateInputStack(rvpsh, input_metadata, input_ranges.min_shapes);
  PrintStack(input_stack);

  auto mp_g_ = rvpsh->jit_graph_;

  PT_DYNAMIC_SHAPE_DEBUG("Triggering compilation of the following graph");
  if (mp_g_) {
    PT_DYNAMIC_SHAPE_DEBUG(
        "JIT_IR_Graph_BEGIN\n", mp_g_->toString(), "JIT_IR_Graph_END");

    size_t graphKey{rvpsh->get_graph_key()};
    size_t graphIndex{habana_lazy::exec::HlExec::GetGraphIndex(
        graphKey, torch::jit::last(input_stack, input_stack.size()))};
    std::string graphName{rvpsh->get_graph_name()};
    std::string opStr{rvpsh->get_op_strs()};
    std::shared_ptr<habana::OptimizedJITGraphAndMetaData>
        jit_ir_graph_and_mdata =
            std::make_shared<habana::OptimizedJITGraphAndMetaData>();
    jit_ir_graph_and_mdata->set_cached_graph(mp_g_);
    jit_ir_graph_and_mdata->set_cached_graph_key(graphKey);
    jit_ir_graph_and_mdata->SetGraphIndex(graphIndex);
    jit_ir_graph_and_mdata->SetOpName(graphName);
    jit_ir_graph_and_mdata->set_cached_opstrs(opStr);
    jit_ir_graph_and_mdata->SetDynamicGraph(true);

    habana::HabanaLaunchOpPT habanaFusedOp{jit_ir_graph_and_mdata};
    try {
      habanaFusedOp.CompileGraphWithRange(
          input_stack,
          input_ranges,
          new_bucket,
          new_recipe_key,
          statpsh,
          dbipsh);
    } catch (std::exception& e) {
      PT_DYNAMIC_SHAPE_DEBUG(
          "HabanaLaunchOpPT::Compile returned exception '", e.what(), "'");

      throw;
    }
    PT_DYNAMIC_SHAPE_DEBUG("Completed compilation ...");
  } else {
    PT_DYNAMIC_SHAPE_DEBUG("Empty JIT graph");
  }

  return ret;
}
