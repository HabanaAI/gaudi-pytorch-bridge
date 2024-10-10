/******************************************************************************
 * Copyright (C) 2024 Habana Labs, Ltd. an Intel Company
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
#pragma once

namespace habana {

size_t computeInputsNumber(const at::Stack& stack);

typedef std::function<synapse_helpers::tensor(
    OpBackend*,
    synapse_helpers::graph&,
    std::string&,
    const std::vector<synTensor>&,
    const std::vector<at::IValue>&,
    int out_index)>
    NodeCreateFunction;
typedef std::function<SharedMetaDataVector(const at::Stack&)>
    SharedMetaCreateFunction;

std::vector<synapse_helpers::tensor> CommonForeachBinary(
    OpBackend* op,
    std::string& guid_,
    const std::vector<synTensor>& inputs,
    synapse_helpers::graph& graph,
    const at::Stack& stack,
    NodeCreateFunction node_creator);

SharedMetaDataVector CommonForeachBinarySharedMeta(
    const at::Stack& stack,
    SharedMetaCreateFunction sharedMetaCreator);
} // namespace habana