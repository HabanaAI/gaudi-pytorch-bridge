/**
* Copyright (c) 2021-2024 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
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
