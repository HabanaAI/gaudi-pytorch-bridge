/******************************************************************************
 * Copyright (C) 2020 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */
#include <ATen/record_function.h>
#include <habana_lazy/hlexec.h>
#include <pybind11/pybind11.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/passes/pass_manager.h>
#include <torch/csrc/jit/python/pybind_utils.h>
#include <torch/csrc/jit/runtime/custom_operator.h>
#include <torch/csrc/jit/runtime/operator_options.h>
#include <functional>
#include "habana_bridge/kernel/hpu_habana_launch_op_pt.h"
#include "habana_bridge/passes/habana_fuser.h"
#include "habana_bridge/passes/remove_inplace_ops.h"
#include "habana_helpers/logging.h"
#include "habana_lazy/hpu_lazy_tensors.h"
#include "synapse_helpers/devmem_logger.h"

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
        return [cc](torch::jit::Stack* stack) {
          RECORD_FUNCTION("HabanaFusedOp", std::vector<c10::IValue>());
          cc->run(*stack);
          return 0;
        };
      },
      c10::AliasAnalysisKind::INTERNAL_SPECIAL_CASE)});
  PT_BRIDGE_END;
}

void torch_habana_register_pre_diff_pass(
    std::function<hb_torch_opts()> get_options) {
  PT_BRIDGE_BEGIN;
  torch::jit::registerPreDiffPass([getOptions = std::move(get_options)](
                                      std::shared_ptr<torch::jit::Graph>& g) {
    auto opts = getOptions();
    if (opts.remove_inplace_ops) {
      PT_BRIDGE_BEGIN;
      habana::RemoveInplaceOps(g);
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

static habana::hb_torch_opts opts;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  std::function<::habana::hb_torch_opts()> get_options = []() { return opts; };
  habana::torch_habana_register_fusion_pass(get_options);
  habana::torch_habana_register_pre_diff_pass(get_options);
  // python API to enable and disable tvm fusion
  m.def("enable", []() { opts.fusion_enabled = true; });
  m.def("disable", []() { opts.fusion_enabled = false; });
  m.def("remove_inplace_ops", []() { opts.remove_inplace_ops = true; });

  // python API to report device memory live allocation details
  m.def("memstat_livealloc", [](const char* msg = "") {
    synapse_helpers::print_live_allocations(msg);
  });

  // Lazy apis
  m.def(
      "mark_step",
      [](const std::string& device_str) {
        habana_lazy::HbLazyTensor::StepMarker(device_str);
      },
      py::arg("device_str") = "");
  m.def(
      "run_saved_model",
      [](const std::string& device_str) {
        habana_lazy::HbLazyTensor::RunSavedGraph(device_str);
      },
      py::arg("device_str") = "");
  m.def(
      "enable_eliminate_common_subexpression",
      [](const bool flag) {
        habana_lazy::exec::OptPassCfg::GetInstance()
            ->enable_eliminate_common_subexpression = flag;
      },
      py::arg("flag"));
  m.def(
      "enable_eliminate_dead_code",
      [](const bool flag) {
        habana_lazy::exec::OptPassCfg::GetInstance()
            ->enable_eliminate_dead_code = flag;
      },
      py::arg("flag"));
  m.def(
      "enable_constant_pooling",
      [](const bool flag) {
        habana_lazy::exec::OptPassCfg::GetInstance()->enable_constant_pooling =
            flag;
      },
      py::arg("flag"));
  m.def(
      "enable_peephole_optimization",
      [](const bool flag) {
        habana_lazy::exec::OptPassCfg::GetInstance()
            ->enable_peephole_optimization = flag;
      },
      py::arg("flag"));
  m.def(
      "enable_fuse_t_mm_optimization",
      [](const bool flag) {
        habana_lazy::exec::OptPassCfg::GetInstance()
            ->enable_fuse_t_mm_optimization = flag;
      },
      py::arg("flag"));
  m.def(
      "enable_fuse_bn_relu_optimization",
      [](const bool flag) {
        habana_lazy::exec::OptPassCfg::GetInstance()
            ->enable_fuse_bn_relu_optimization = flag;
      },
      py::arg("flag"));
  m.def(
      "enable_permute_pass",
      [](const bool flag) {
        habana_lazy::exec::OptPassCfg::GetInstance()->enable_permute_pass =
            flag;
      },
      py::arg("flag"));
  m.def(
      "enable_replace_inplace_ops",
      [](const bool flag) {
        habana_lazy::exec::OptPassCfg::GetInstance()
            ->enable_replace_inplace_ops = flag;
      },
      py::arg("flag"));
  m.doc() = "This module registers habana backend.";
}
