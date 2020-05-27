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
#include <torch/csrc/jit/runtime/custom_operator.h>
#include <torch/csrc/jit/runtime/operator_options.h>
#include <torch/csrc/jit/passes/pass_manager.h>
#include <torch/csrc/jit/ir/ir.h>
#include "habana_bridge/passes/habana_fuser.h"

#include "habana_helpers/logging.h"

#include "habana_bridge/kernel/hpu_habana_launch_op_pt.h"

namespace habana {
namespace {
    void registerHabanaLaunchOp() {
        LOG_FUNC_BEGIN;
        torch::jit::RegisterOperators op({torch::jit::Operator(
            // TODO: Change this to HabanaFusionOp
            torch::jit::Symbol::fromQualString("prim::HabanaFusedOp"),
            [](const torch::jit::Node* node) -> torch::jit::Operation {
                const auto cc = std::make_shared<HabanaLaunchOpPT>(node, false);
                return [cc](torch::jit::Stack &stack) {
                    RECORD_FUNCTION("HabanaFusedOp", std::vector<c10::IValue>());
                    cc->run(stack);
                    return 0;
                };
            },
            c10::AliasAnalysisKind::INTERNAL_SPECIAL_CASE)});
        LOG_FUNC_END;
    }

    void torch_habana_enable(std::function<bool()> enableHabanaCompile) {
		LOG_FUNC_BEGIN;
		registerHabanaLaunchOp();
        torch::jit::RegisterPass pass(
            [enableHabanaCompile =
                std::move(enableHabanaCompile)](std::shared_ptr<torch::jit::Graph>& g) {
                    if (enableHabanaCompile()) {
                        LOG_FUNC_BEGIN;
						torch::jit::HabanaFuseGraph(g);
                        LOG_FUNC_END;
                        std::cout << "Habana Post Fusion Graph: " << std::endl;
                        std::cout << g->toString() << std::endl;
                    }
        });
		LOG_FUNC_END;
}

}  // namespace
}  // namespace habana
