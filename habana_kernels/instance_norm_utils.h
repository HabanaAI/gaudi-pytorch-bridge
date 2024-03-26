/*******************************************************************************
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
/*******************************************************************************
 * Core content of this file comes from:
 * pytorch-fork/torch/csrc/autograd/FunctionsManual.h
 * It was modified by adding at:: namespace to all Tensor classes
 * License available: python_packages/LICENSE.txt
 *******************************************************************************
 */
#pragma once

std::tuple<at::Tensor, at::Tensor, at::Tensor> batchnorm_double_backward(
    const at::Tensor& input,
    const c10::optional<at::Tensor>& gamma,
    const at::Tensor& ggI,
    const at::Tensor& ggG,
    const at::Tensor& ggB,
    const at::Tensor& gO,
    const c10::optional<at::Tensor>& running_mean,
    const c10::optional<at::Tensor>& running_var,
    bool training,
    double eps,
    const c10::optional<at::Tensor>& save_mean,
    const c10::optional<at::Tensor>& save_invstd,
    std::array<bool, 3> output_mask);