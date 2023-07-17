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
from habana_frameworks.torch import _core_C
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
        for name, param in model._buffers['ranges']['outputs'].items():
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

_const_id = -1
def _mark_params_as_const(model=None) -> None:
    if model is None:
        return
    for param, param_t in model.state_dict().items():
        try:
            param_t_meta = _core_C.get_new_tensor_extra_meta(param_t)
        except (RuntimeError):
            param_t_meta = _core_C.get_tensor_extra_meta(param_t)
        global _const_id
        _const_id = _const_id + 1
        param_t_meta.is_const_tensor = True
        param_t_meta.const_id = _const_id
        param_t_meta_copy = _core_C.get_tensor_extra_meta(param_t)
        is_const = param_t_meta_copy.is_const_tensor
        id = param_t_meta_copy.const_id
        # print("Tensor '{}' is_const '{}' id '{}'".format(param, is_const, id))

def _get_marked_const_count() -> int:
    global _const_id
    count = (_const_id + 1)
    # print("Total number of marked const tensors: '{}'".format(count))
    return count

def _check_params_as_const(model=None) -> None:
    if model is None:
        return
    for param, param_t in model.state_dict().items():
        param_t_meta_copy = _core_C.get_tensor_extra_meta(param_t)
        is_const = param_t_meta_copy.is_const_tensor

def hpu_initialize(model=None, optimizer=None, args=None):
    if "PT_HPU_INFERENCE_MODE" not in environ:
        environ["PT_HPU_INFERENCE_MODE"] = "1"
    if "PT_HPU_MATMUL3D_2D_RESHAPE" not in environ:
        environ["PT_HPU_MATMUL3D_2D_RESHAPE"] = "1"
    # media WA to not convert imagenet label tensor to int64
    if model is not None:
        #_mark_params_as_const(model=model)
        _read_min_max_overwrite()
        with _e_handler():
            _handle_quant_stats(model)
