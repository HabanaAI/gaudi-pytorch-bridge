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
from os import environ, getenv

import torch
from habana_frameworks.torch import _core_C, hpu
from habana_frameworks.torch.internal import fuse_conv_bn
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
    range_path = environ.get("PT_INFERENCE_RANGE_FILE")
    if range_path:
        with open(range_path) as file:
            for line in file:
                line = line[line.find("/") + 1 : len(line)]
                line = line.split()
                _record_quant_param(line[0], float(line[1]), float(line[2]))


def adjust_name(name):
    # name = name.replace(".bmm.",".baddbmm.")
    # name = name.replace(".bmm2.",".bmm.")
    name = name.replace(".min_val", "")
    name = name.replace(".max_val", "")
    name = name.replace("layernorm.norm", "layernorm")
    # print(f"[name after adjustment] := {name}", flush=True)
    return name


def _handle_quant_stats(model=None):
    if model is not None:
        min_calibration_data = dict()
        max_calibration_data = dict()
        placeholder_dict = dict()
        for name, param in model._buffers["ranges"]["outputs"].items():
            if name.endswith(".min_val"):
                name = adjust_name(name)
                min_calibration_data[name] = param.item()
            if name.endswith(".max_val"):
                name = adjust_name(name)
                max_calibration_data[name] = param.item()
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


def _mark_params_as_const(model=None, only_scales=False, console_prints=False) -> None:
    if model is None:
        return
    for param, param_t in model.state_dict().items():
        if only_scales and "scale" not in param:
            continue
        try:
            param_t_meta = _core_C.get_new_tensor_extra_meta(param_t)
        except RuntimeError:
            param_t_meta = _core_C.get_tensor_extra_meta(param_t)
            if param_t_meta.const_id != -1:
                param_t_meta.is_const_tensor = True
                if console_prints:
                    print("Metadata already exists, const_id '{}'".format(param_t_meta.const_id))
                continue
        global _const_id
        _const_id = _const_id + 1
        param_t_meta.is_const_tensor = True
        param_t_meta.const_id = _const_id
        param_t_meta_copy = _core_C.get_tensor_extra_meta(param_t)
        is_const = param_t_meta_copy.is_const_tensor
        id = param_t_meta_copy.const_id
        if console_prints:
            print("Tensor '{}' is_const '{}' id '{}'".format(param, is_const, id))


def _get_marked_const_count() -> int:
    global _const_id
    count = _const_id + 1
    # print("Total number of marked const tensors: '{}'".format(count))
    return count


def _check_params_as_const(model=None, only_scales=False) -> None:
    if model is None:
        return
    for param, param_t in model.state_dict().items():
        if only_scales and "scale" not in param:
            continue
        param_t_meta_copy = _core_C.get_tensor_extra_meta(param_t)
        is_const = param_t_meta_copy.is_const_tensor


def check_env_flag(name, default=""):
    return getenv(name, default).upper() in ["ON", "1", "YES", "TRUE", "Y"]


def _set_quantization_attributes(model):
    if (
        "HB_QUANTIZATION" in model._buffers
        and "quantization" in model._buffers["HB_QUANTIZATION"]
        and model._buffers["HB_QUANTIZATION"]["quantization"] == True
    ):
        hpu.enable_quantization()


_set_env = 1


def hpu_set_env(model=None):
    """
    [TO BE DEPRECATED] Please use hpu_set_inference_env instead
    Enables inference mode
    If model is given, fuses conv+bn nodes
    To be called before moving tensors/model to hpu
    """
    global _set_env
    hpu.enable_inference_mode()
    hpu.enable_matmul3d_2d_reshape()
    _set_env = 0
    if check_env_flag("PT_HPU_WEIGHT_SHARING", "1") and check_env_flag("EXPERIMENTAL_WEIGHT_SHARING", "1"):
        print(
            """\033[31mWARNING: The experimental weight sharing feature is enabled and may cause larger device memory
              consumption in quantized models. Please disable it by setting PT_HPU_WEIGHT_SHARING=0\033[0m"""
        )
    if model is not None:
        modified_model = fuse_conv_bn.fuse(model)
        return modified_model


def hpu_set_inference_env(model=None):
    """
    Enables inference mode
    If model is given, fuses conv+bn nodes
    To be called before moving tensors/model to hpu
    """

    global _set_env
    hpu.enable_inference_mode()
    hpu.enable_matmul3d_2d_reshape()
    _set_env = 0
    if check_env_flag("PT_HPU_WEIGHT_SHARING", "1") and check_env_flag("EXPERIMENTAL_WEIGHT_SHARING", "1"):
        print(
            """\033[31mWARNING: The experimental weight sharing feature is enabled and may cause larger device memory
              consumption in quantized models. Please disable it by setting PT_HPU_WEIGHT_SHARING=0\033[0m"""
        )
    if model is not None:
        modified_model = fuse_conv_bn.fuse(model)
        return modified_model


def hpu_initialize(model=None, mark_only_scales_as_const=False, optimizer=None, args=None):
    """
    [TO BE DEPRECATED] Please use hpu_inference_initialize instead
    Mark params of the model on HPU as const
    To be called after moving tensors/model to hpu
    To be called after model.to(hpu)
    """
    global _set_env
    if _set_env == 1:
        hpu.enable_inference_mode()
        hpu.enable_matmul3d_2d_reshape()
    if model is not None:
        if getenv("PT_HPU_LAZY_MODE", "1") != "0":
            _mark_params_as_const(model=model, only_scales=mark_only_scales_as_const)
            _check_params_as_const(model=model, only_scales=mark_only_scales_as_const)
        _read_min_max_overwrite()
        _set_quantization_attributes(model)
        with _e_handler():
            _handle_quant_stats(model)


def hpu_inference_initialize(model=None, mark_only_scales_as_const=False, optimizer=None, args=None):
    """
    Mark params of the model on HPU as const
    To be called after moving tensors/model to hpu
    To be called after model.to(hpu)
    """
    global _set_env
    if _set_env == 1:
        hpu.enable_inference_mode()
        hpu.enable_matmul3d_2d_reshape()
    if model is not None:
        if getenv("PT_HPU_LAZY_MODE", "1") != "0":
            _mark_params_as_const(model=model, only_scales=mark_only_scales_as_const)
            _check_params_as_const(model=model, only_scales=mark_only_scales_as_const)
        _read_min_max_overwrite()
        _set_quantization_attributes(model)
        with _e_handler():
            _handle_quant_stats(model)


def hpu_reset_env():
    """
    [TO BE DEPRECATED] Please use hpu_teardown_inference_env instead
    Disables inference mode
    To be called after model execution is done on HPU
    """
    hpu.disable_inference_mode()
    hpu.disable_matmul3d_2d_reshape()


def hpu_teardown_inference_env():
    """
    Disables inference mode
    To be called after model execution is done on HPU
    """
    hpu.disable_inference_mode()
    hpu.disable_matmul3d_2d_reshape()
