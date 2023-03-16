###############################################################################
# Copyright (C) 2023 Habana Labs, Ltd. an Intel Company
# All Rights Reserved.
#
# Unauthorized copying of this file or any element(s) within it, via any medium
# is strictly prohibited.
# This file contains Habana Labs, Ltd. proprietary and confidential information
# and is subject to the confidentiality and license agreements under which it
# was provided.
#
###############################################################################

import contextlib
from os import environ

import torch
from habana_frameworks.torch import hpu
from habana_frameworks.torch.utils import _experimental_C
from torch.fx import symbolic_trace

from .torch_overwrites import _names_hook_already_registered


@contextlib.contextmanager
def _e_handler():
    try:
        yield
    except Exception as e:
        pass

def _record_quant_param(name, min, max) -> None:
    if hpu.is_available():
        #handle the naming mismatch issue with a temp fix, till we
        #find a way to get unique module names from the calibration tool
        if "add" not in  name:
           name = name.replace("_", ".")
        _experimental_C.record_quant_param(name, min, max)

def _read_min_max_overwrite():
    range_path = environ.get('PT_INFERENCE_RANGE_FILE')
    if range_path:
        with open(range_path) as file:
            for line in file:
                line = line[line.find('/')+1:len(line)]
                line = line.split()
                _record_quant_param(line[0], float(line[1]), float(line[2]))

def _handle_quant_stats(model=None):
    if model is not None:
        min_calibration_data = dict()
        max_calibration_data = dict()
        placeholder_dict = dict()
        gm : torch.fx.GraphModule = symbolic_trace(model)
        for node in gm.graph.nodes:
            for i in range(len(node.all_input_nodes)):
                if node.all_input_nodes[i].op == "placeholder":
                    name = '.'.join([node.target, "placeholder",str(i)])
                    placeholder_dict[node.all_input_nodes[i].target] = name
                    short_name = '.'.join([node.all_input_nodes[i].target, str(i)])
                    placeholder_dict[short_name] = name
        for name, param in model.state_dict().items():
            if name.endswith('.min_val'):
                min_calibration_data[name.replace(".min_val","")] = param.item()
            if name.endswith('.max_val'):
                max_calibration_data[name.replace(".max_val","")] = param.item()
        for name, param in placeholder_dict.items():
            if name in min_calibration_data.keys():
                min_calibration_data[param] = min_calibration_data[name]
                min_calibration_data.pop(name)
            if name in max_calibration_data.keys():
                max_calibration_data[param] = max_calibration_data[name]
                max_calibration_data.pop(name)
        for name, param in min_calibration_data.items():
            try:
                _record_quant_param(name, min_calibration_data[name], max_calibration_data[name])
            except:
                pass
        for submodule_name, submodule in model.named_modules():
            #if set_hook is "x", module.custom_name will be x
            #What will be the Quantization record will be? layer1/x/conv1
            #Without set_hook, submodule_name is layer1.0, Scope should be layer1/0/conv1
            if isinstance(submodule, torch.nn.Module) and not _names_hook_already_registered(submodule):
                try:
                    dot_index = submodule_name.rfind(".")
                    submodule_name = dot_index == -1 and submodule_name or submodule_name[dot_index+1:]
                    submodule.custom_name = submodule_name
                    submodule.register_forward_pre_hook(pre_fwd_hook)
                    submodule.register_forward_hook(post_fwd_hook)
                    submodule.names_hook = True
                except (RuntimeError):
                    pass

def hpu_initialize(model=None, optimizer=None, args=None):
    if "PT_HPU_INFERENCE_MODE" not in environ:
        environ["PT_HPU_INFERENCE_MODE"] = "1"
    if "PT_HPU_MATMUL3D_2D_RESHAPE" not in environ:
        environ["PT_HPU_MATMUL3D_2D_RESHAPE"] = "1"
    # media WA to not convert imagenet label tensor to int64
    if "GRECO_INFERENCE" not in environ:
        environ["GRECO_INFERENCE"] = "1"
    if model is not None:
        _read_min_max_overwrite()
        with _e_handler():
            _handle_quant_stats(model)
