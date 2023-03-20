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

#include <c10/core/TensorImpl.h>
#include <torch/csrc/jit/ir/ir.h>
#include "habana_lazy/hpu_lazy_tensors.h"
#include "habana_lazy/tensor_impl.h"

namespace habana_lazy {
using Graph = torch::jit::Graph;
using Node = torch::jit::Node;

HbInternalTensorImpl* GetBackEndTensorImpl(
    std::shared_ptr<Graph>& graph,
    torch::jit::Stack& stack,
    Node* node,
    const int idx);

void* GetDataInHostBuffer(
    std::shared_ptr<Graph>& graph,
    torch::jit::Stack& stack,
    Node* node,
    const int idx);

void UpdateDataInDeviceMem(
    std::shared_ptr<Graph>& graph,
    torch::jit::Stack& stack,
    Node* node,
    const int idx,
    void* host_ptr);

void RecalculateBatchnormParams(
    std::shared_ptr<Graph>& graph,
    torch::jit::Stack& stack);
}; // namespace habana_lazy
