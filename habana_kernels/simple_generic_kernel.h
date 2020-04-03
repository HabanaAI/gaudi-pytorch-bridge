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

enum class SynapsePassType {
    NO_PASS = 0,
    FORWARD_PASS,
    BACKWARD_PASS
};
// synapse kernel with single op
void synapse_simple_generic_kernel(
    std::vector<const at::Tensor*> pt_outputs, // NHWC
    std::vector<const at::Tensor*> pt_inputs, // NHWC
    const std::string& node_guid,
    const void* syn_param,
    size_t syn_param_size,
    SynapsePassType pass_type);

/*********************************************************************************
@brief generic function to support inplace kernels

@param pt_inputs - Vector containing input tensor pointers
@param node_guid - Global Unique Identifier used to uniquely map the operator to
device
@param syn_param - additional parameters needed for the op. nullptr if
parameters are not needed
@param syn_param_size - number of additional params. 0 if not needed
@param forward_pass - boolean flag to specify forward or backward operator
**********************************************************************************/
void synapse_simple_generic_inplace_kernel(
    std::vector<const at::Tensor*> pt_inputs, // NHWC
    const std::string& node_guid,
    const void* syn_param,
    size_t syn_param_size,
    SynapsePassType pass_type);