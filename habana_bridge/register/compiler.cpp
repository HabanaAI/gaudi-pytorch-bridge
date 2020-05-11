/******************************************************************************
 * Copyright (C) 2020 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */
#include "compiler.h"

#include <torch/csrc/autograd/record_function.h>
#include <torch/csrc/jit/constants.h>
#include <torch/csrc/jit/interpreter.h>
#include "habana_helpers/logging.h"

#include <algorithm>

using namespace torch::jit;

static std::atomic<size_t> next_kernel_id{0};

HbCompiler::HbCompiler(const Node* node, bool debug) {
    subgraph_ = node->g(attr::Subgraph);
    debug_    = debug;
}

void HbCompiler::run(Stack& stack) {
    LOG_FUNC_BEGIN;

    int num_inputs = subgraph_->inputs().size();
    at::ArrayRef<IValue> inputs = last(stack, num_inputs);

    // TODO:: Add code to validate if all input tensors are on 
    // Habana device, who moves the tensors to habana device??
    // if not on habana device, we should run a fall back kernel
    bool is_all_hpu = true;
    std::for_each(inputs.begin(), inputs.end(), [&is_all_hpu](const IValue input) 
                                                {
                                                    if (input.isTensor()) {
                                                        is_all_hpu &= true; //input.toTensor().device().is_hpu();
                                                    }
                                                });
    // Now if we have do not have the tensor mapped to hpu, run fall back kernel
    /*if (is_all_hpu) {
       // call fall back kernel 
    }*/

    if (debug_ && is_all_hpu) {
        // TODO:: method to print the subgraph for debug
        subgraph_->print(std::cout);    
    }
    

    std::vector<at::Tensor> outputs;
    //std::string name = "habana_kernel_" + c10::to_string(next_kernel_id++);
    //CreateHabanaFusedOpKernel(name, graph, inputs, outputs);
    //launchKernel();

    // Update the stack
    drop(stack, num_inputs);
    stack.insert(
        stack.end(),
        std::make_move_iterator(outputs.begin()),
        std::make_move_iterator(outputs.end())); 

    LOG_FUNC_END;      
}