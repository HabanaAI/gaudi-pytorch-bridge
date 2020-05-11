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

#include <torch/csrc/jit/argument_spec.h>
#include <torch/csrc/jit/interpreter.h>
#include <torch/csrc/jit/ir.h>
#include <ATen/Tensor.h>

class HbCompiler {
    public:
        explicit HbCompiler(
            const torch::jit::Node* node,
            bool debug
        );
        void run(torch::jit::Stack& stack);
    private:
          std::shared_ptr<torch::jit::Graph> subgraph_;
          bool debug_;
};