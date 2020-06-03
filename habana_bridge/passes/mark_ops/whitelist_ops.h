/*****************************************************************************
 * Copyright (C) 2020 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */
#pragma once

#include <iostream>
#include <string>
#include <unordered_set>
#include "habana_helpers/logging.h"

class HabanaWhiteList {
    private:
    const static std::unordered_set<std::string> HabanaWhiteListOps;
    public:
    static bool is_op_habana_whitelisted(std::string opName);
};

const std::unordered_set<std::string> HabanaWhiteList::HabanaWhiteListOps = {
        "aten::copy_",
        "aten::as_strided",
        "aten::set_",
        "aten::view",
        "aten::cat",
        "aten::_cat",
        "aten::transpose",
        "aten::transpose_",
        "aten::t",
        "aten::t_",
        "aten::add_",
        "aten::add",
        "aten::sub_",
        "aten::sub.Scalar",
        "aten::mul_",
        "aten::mul",
        "aten::eq",
        "aten::eq.Tensor_out",
        "aten::div",
        "aten::div.out",
        "aten::div_",
        "aten::div.Scalar",
        "aten::div_.Scalar",
        "aten::convolution_overrideable",
        "aten::convolution_backward_overrideable",
        "aten::fill_",
        "aten::mm",
        "aten::matmul",
        "aten::addmm",
        "aten::native_batch_norm",
        "aten::native_batch_norm_backward",
        "aten::max_pool2d_with_indices",
        "aten::max_pool2d_with_indices_backward",
        "aten::avg_pool2d",
        "aten::avg_pool2d_backward",
        "aten::uniform_",
        "aten::normal_",
        "aten::sum",
        "aten::mean",
        "aten::_log_softmax",
        "aten::_log_softmax_backward_data",
        "aten::clone",
        "aten::empty",
        "aten::empty_strided",
        "aten::threshold_backward",
        "aten::topk",
        "aten::relu_",
        "aten::relu",
        "prim::Constant"
};

bool HabanaWhiteList::is_op_habana_whitelisted(std::string opName) {
        if(HabanaWhiteListOps.find(opName) != HabanaWhiteListOps.end())
            return true;
        return false;
}
