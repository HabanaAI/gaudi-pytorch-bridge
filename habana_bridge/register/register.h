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
#include <functional>

#include <torch/csrc/autograd/record_function.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/passes/pass_manager.h>
#include <torch/csrc/jit/runtime/custom_operator.h>
#include <torch/csrc/jit/runtime/operator_options.h>
#include "habana_bridge/passes/habana_fuser.h"
#include "habana_bridge/passes/remove_inplace_ops.h"

#include "habana_helpers/logging.h"

#include "habana_bridge/kernel/hpu_habana_launch_op_pt.h"

namespace habana {
namespace {

struct hb_torch_opts {
  bool fusion_enabled = false;
  bool remove_inplace_ops = false;
};

void registerHabanaLaunchOp() {
  PT_BRIDGE_BEGIN;
  torch::jit::RegisterOperators op({torch::jit::Operator(
      // TODO: Change this to HabanaFusionOp
      torch::jit::Symbol::fromQualString("prim::HabanaFusedOp"),
      [](const torch::jit::Node* node) -> torch::jit::Operation {
        const auto cc = std::make_shared<HabanaLaunchOpPT>(node, false);
        return [cc](torch::jit::Stack& stack) {
          RECORD_FUNCTION("HabanaFusedOp", std::vector<c10::IValue>());
          cc->run(stack);
          return 0;
        };
      },
      c10::AliasAnalysisKind::INTERNAL_SPECIAL_CASE)});
  PT_BRIDGE_END;
}

void torch_habana_register_pre_diff_pass(
    std::function<hb_torch_opts()> get_options) {
  PT_BRIDGE_BEGIN;
  torch::jit::RegisterPreDiffPass preDiffPass(
      [getOptions =
           std::move(get_options)](std::shared_ptr<torch::jit::Graph>& g) {
        auto opts = getOptions();
        if (opts.remove_inplace_ops) {
          PT_BRIDGE_BEGIN;
          ::habana::RemoveInplaceOps(g);
          PT_BRIDGE_END;
          PT_BRIDGE_DEBUG("Habana Post Remove Inplace Pass Graph: ");
          PT_BRIDGE_DEBUG(g->toString());
        }
      });
  PT_BRIDGE_END;
}

void torch_habana_register_fusion_pass(
    std::function<hb_torch_opts()> get_options) {
  PT_BRIDGE_BEGIN;
  registerHabanaLaunchOp();
  torch::jit::RegisterPass pass([getOptions = std::move(get_options)](
                                    std::shared_ptr<torch::jit::Graph>& g) {
    auto opts = getOptions();
    if (opts.fusion_enabled) {
      PT_BRIDGE_BEGIN;
      torch::jit::HabanaFuseGraph(g);
      PT_BRIDGE_END;
      PT_BRIDGE_DEBUG("Habana Post Fusion Graph: ");
      PT_BRIDGE_DEBUG(g->toString());
    }
  });
  PT_BRIDGE_END;
}

} // namespace
} // namespace habana

// This function is resolved to pre-loaded synapse_logger.
// Otherwise, this function doesn't do anything.
void print_live_allocations(const char* msg = "") __attribute__((weak));
