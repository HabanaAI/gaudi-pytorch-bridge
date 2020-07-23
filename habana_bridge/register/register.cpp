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

static struct ::habana::hb_torch_opts opts;

PYBIND11_MODULE(hb_torch, m) {
    std::function<::habana::hb_torch_opts()> get_options = []() { return opts; };
    habana::torch_habana_register_fusion_pass(get_options);
    habana::torch_habana_register_pre_diff_pass(get_options);
    // python API to enable and disable tvm fusion
    m.def("enable", []() { opts.fusion_enabled = true; });
    m.def("disable", []() { opts.fusion_enabled = false; });
    m.def("remove_inplace_ops", []() {opts.remove_inplace_ops = true; });

    // python API to report device memory live allocation details
    m.def("memstat_livealloc", []() { print_live_allocations(); });

    m.doc() = "This module registers habana backend.";
}
