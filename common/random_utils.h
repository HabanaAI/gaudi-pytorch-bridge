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
#pragma once

// Helpers for defining habana random op and habana random checkpoint op.
// They are needed to handle determinism in torch.compile
// and activation checkpointing.
// Checkpoint variant returns seed tensor as a additional output.

#define HABANA_RANDOM_DEF(op_name, inputs)                 \
  m.def("hpu::habana_" #op_name "(" inputs ") -> Tensor"); \
  m.def("hpu::habana_" #op_name "_checkpoint(" inputs ") -> (Tensor, Tensor)");

#define HABANA_RANDOM_DEF_VARIANT(op_name, variant, inputs)             \
  m.def("hpu::habana_" #op_name "." #variant "(" inputs ") -> Tensor"); \
  m.def("hpu::habana_" #op_name "_checkpoint." #variant "(" inputs      \
        ") -> (Tensor, Tensor)");

#define HABANA_RANDOM_DEF_2_OUTS(op_name, inputs)                    \
  m.def("hpu::habana_" #op_name "(" inputs ") -> (Tensor, Tensor)"); \
  m.def("hpu::habana_" #op_name "_checkpoint(" inputs                \
        ") -> (Tensor, Tensor, Tensor)");
