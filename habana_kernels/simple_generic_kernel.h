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
#include <torch/script.h>
#include <string>
#include <vector>

#include "habana_helpers/tensor_utils.h"
#include "kernel_utils.h"

// synapse kernel with single op
void synapse_simple_generic_kernel(
    std::vector<const at::Tensor*> pt_outputs, // NHWC
    std::vector<const at::Tensor*> pt_inputs, // NHWC
    const std::string& node_guid,
    const void* syn_param,
    size_t syn_param_size,
    bool forward_pass);