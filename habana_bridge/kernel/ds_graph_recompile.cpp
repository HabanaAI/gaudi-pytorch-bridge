/******************************************************************************
 * Copyright (C) 2021 Habana Labs, Ltd. an Intel Company
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

#include "habana_bridge/kernel/ds_graph_recompile.h"

#include "habana_lazy/aten_lazy_bridge.h"
#include "habana_lazy/hpu_lazy_tensors.h"
#include "habana_lazy/lazy_executor.h"

#include "habana_kernels/lazy_kernels.h"

#include "pytorch_helpers/habana_helpers/logging.h"
#include "pytorch_helpers/habana_helpers/tensor_info.h"

at::Tensor habana::CreateEmptyTensor(
    const PtTensorInfo& ti,
    const std::vector<int64_t>& tshape) {
  if (ti.tensor_type() == SHAPE_TENSOR) {
    auto pt_tensor = habana_lazy::empty_hpu_lazy(
        tshape, ti.get_topts(), ti.get_mf(), false, SHAPE_TENSOR);
    return pt_tensor;
  }
  auto pt_tensor = at::empty(tshape, ti.get_topts(), ti.get_mf());
  return pt_tensor;
}

torch::jit::Stack habana::CreateInputStack(
    std::shared_ptr<habana::RecipeValueSpec> rvpsh,
    habana_helpers::DynamicBucketInfo::TensorShapes& input_shapes) {
  PT_BRIDGE_BEGIN;
  torch::jit::Stack new_input_stack;

  for (size_t tidx = 0; tidx < rvpsh->num_inputs; tidx++) {
    auto& ti = rvpsh->dtensorinfos->at(tidx);
    TORCH_CHECK(
        input_shapes.count(tidx),
        "Tensor index ",
        tidx,
        "is missing from ",
        input_shapes);
    auto pt_input =
        habana::CreateEmptyTensor(ti, input_shapes.at(tidx).get_dims());
    new_input_stack.push_back(torch::jit::IValue(pt_input));
  }
  PT_BRIDGE_END;
  return new_input_stack;
}

void habana::PrintStack(torch::jit::Stack& st) {
  std::ostream& O = std::cout;

  O << "aten_inputs #" << st.size() << "::" << '\n';
  for (size_t idx = 0; idx < st.size(); idx++) {
    PrintATenTensor(st.at(idx));
  }
}

bool habana::RefineBucketDS(double time_improve_factor) {
  bool is_refined{true};
  PT_BRIDGE_DEBUG(
      "Current improvement factor for refinement is ", time_improve_factor);
  DynamicBucketInfoMap::get_instance().refine();
  return is_refined;
}

bool habana::CompileGraphWithRange(
    std::shared_ptr<habana::RecipeValueSpec> rvpsh,
    habana_helpers::DynamicBucketInfo::ResultShapes& input_ranges,
    habana_helpers::Bucket& new_bucket) {
  bool ret{true};

  // wait till the execution complete
  bool use_flag{false};
  do {
    use_flag = rvpsh->get_use_flag();
    if (use_flag) {
      PT_BRIDGE_DEBUG("waiting for the completion of recipe, key ", rvpsh->key);
      std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }
  } while (use_flag);
  rvpsh->set_use_flag(true);
  PT_DYNAMIC_SHAPE_DEBUG(
      "BucketRefinement: Will use the following recipe for compilation\n",
      rvpsh->header_str());

  auto& device = synapse_helpers::HPURegistrar::get_device();
  auto hl_context =
      habana_lazy::habana_lazy_executor.getDeviceExecutionContext(device.id());
  SET_ENV_FLAG_NEW(PT_HPU_LAZY_LOWERING, 1, 1);
  hl_context->setExecutionMode(kLOWERING);

  torch::jit::Stack input_stack =
      habana::CreateInputStack(rvpsh, input_ranges.min_shapes);
  PrintStack(input_stack);

  auto mp_g_ = rvpsh->jit_graph_;
  rvpsh->set_use_flag(false);

  PT_DYNAMIC_SHAPE_DEBUG("Triggering compilation of the following graph");
  if (mp_g_) {
    PT_DYNAMIC_SHAPE_DEBUG(
        "JIT_IR_Graph_BEGIN\n", mp_g_->toString(), "JIT_IR_Graph_END");

    size_t graphIndex{0xABCDEF};
    habana::HabanaLaunchOpPT habanaFusedOp{mp_g_, false, graphIndex};
    try {
      habanaFusedOp.CompileGraphWithRange(
          input_stack, input_ranges, new_bucket);
    } catch (std::exception& e) {
      PT_DYNAMIC_SHAPE_DEBUG(
          "HabanaLaunchOpPT::Compile returned exception '", e.what(), "'");

      hl_context->setExecutionMode(kLAZY);
      hl_context->MarkTensorsExecuted();
      UNSET_ENV_FLAG_NEW(PT_HPU_LAZY_LOWERING);
      throw;
    }
    PT_DYNAMIC_SHAPE_DEBUG("Completed compilation ...");
    hl_context->setExecutionMode(kLAZY);
    hl_context->MarkTensorsExecuted();
    UNSET_ENV_FLAG_NEW(PT_HPU_LAZY_LOWERING);
  } else {
    PT_DYNAMIC_SHAPE_DEBUG("Empty JIT graph");
  }

  return ret;
}
