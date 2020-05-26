/******************************************************************************
 * Copyright (C) 2020 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */
#include <pybind11/pybind11.h>
#include <torch/csrc/jit/runtime/custom_operator.h>
#include <torch/csrc/jit/runtime/operator_options.h>
#include <torch/csrc/jit/passes/pass_manager.h>
#include <torch/csrc/jit/python/pybind_utils.h>


#include "register.h"

namespace py = pybind11;

// control if we enable Habana device
static bool fusion_enabled = false;

PYBIND11_MODULE(hb_torch, m) {
    std::function<bool()> is_enabled = []() { return fusion_enabled; };
    habana::torch_habana_enable(is_enabled);
    // python API to enable and disable tvm fusion
    m.def("enable", []() { fusion_enabled = true; });
    m.def("disable", []() { fusion_enabled = false; });

    m.doc() = "This module registers habana backend.";
}
