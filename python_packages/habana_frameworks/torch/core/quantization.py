###############################################################################
#
#  Copyright (c) 2021-2025 Intel Corporation
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
###############################################################################

import contextlib
from os import environ, getenv

from habana_frameworks.torch import _core_C, hpu
from habana_frameworks.torch.internal import fuse_conv_bn
from habana_frameworks.torch.utils import _experimental_C


@contextlib.contextmanager
def _e_handler():
    try:
        yield
    except Exception:
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
        min_calibration_data = {}
        max_calibration_data = {}
        placeholder_dict = {}
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
        for name, _ in min_calibration_data.items():
            try:
                _record_quant_param(name, min_calibration_data[name], max_calibration_data[name])
            except:
                pass


_const_id = -1


def _mark_params_as_const(model=None, mark_scales=False, mark_non_scales=False, console_prints=False) -> None:
    if model is None:
        return

    def perform_const_marking(param, param_t):
        try:
            param_t_meta = _core_C.get_new_tensor_extra_meta(param_t)
        except RuntimeError:
            param_t_meta = _core_C.get_tensor_extra_meta(param_t)
            if param_t_meta.const_id != -1:
                param_t_meta.is_const_tensor = True
                if console_prints:
                    print(f"Metadata already exists, const_id '{param_t_meta.const_id}'")
                return
        global _const_id
        _const_id = _const_id + 1
        param_t_meta.is_const_tensor = True
        param_t_meta.const_id = _const_id
        param_t_meta_copy = _core_C.get_tensor_extra_meta(param_t)
        is_const = param_t_meta_copy.is_const_tensor
        id = param_t_meta_copy.const_id
        if console_prints:
            print(f"Tensor '{param}' is_const '{is_const}' id '{id}'")

    for param, param_t in model.state_dict().items():
        if (
            (mark_scales and mark_non_scales)
            or (mark_scales and "scale" in param)
            or (mark_non_scales and "scale" not in param)
        ):
            perform_const_marking(param, param_t)


def _get_marked_const_count() -> int:
    global _const_id
    count = _const_id + 1
    # print("Total number of marked const tensors: '{}'".format(count))
    return count


def _check_params_as_const(model=None, mark_scales=False, mark_non_scales=False) -> None:
    if model is None:
        return

    def check_constant_mark(param, param_t):
        param_t_meta_copy = _core_C.get_tensor_extra_meta(param_t)
        is_const = param_t_meta_copy.is_const_tensor

    for param, param_t in model.state_dict().items():
        if (
            (mark_scales and mark_non_scales)
            or (mark_scales and "scale" in param)
            or (mark_non_scales and "scale" not in param)
        ):
            check_constant_mark(param, param_t)


def check_env_flag(name, default=""):
    return getenv(name, default).upper() in ["ON", "1", "YES", "TRUE", "Y"]


def _set_quantization_attributes(model):
    if (
        "HB_QUANTIZATION" in model._buffers
        and "quantization" in model._buffers["HB_QUANTIZATION"]
        and model._buffers["HB_QUANTIZATION"]["quantization"] is True
    ):
        hpu.enable_quantization()


_set_env = 1


def hpu_set_env(model=None):
    print(
        """
        [TO BE DEPRECATED] Please use hpu_inference_set_env instead
        """
    )
    return hpu_inference_set_env(model)


def hpu_set_inference_env(model=None):
    print(
        """
        [TO BE DEPRECATED] Please use hpu_inference_set_env instead
        """
    )
    return hpu_inference_set_env(model)


def hpu_inference_set_env(model=None):
    """
    Enables inference mode
    If model is given, fuses conv+bn nodes
    To be called before moving tensors/model to hpu
    """

    if not hpu.is_inference_mode_enabled():
        hpu.enable_inference_mode()
    if not hpu.is_matmul3d_2d_reshape_enabled():
        hpu.enable_matmul3d_2d_reshape()

    if check_env_flag("PT_HPU_WEIGHT_SHARING", "1") and check_env_flag("EXPERIMENTAL_WEIGHT_SHARING", "1"):
        print(
            """\033[31mWARNING: The experimental weight sharing feature is enabled and may cause larger device memory
              consumption in quantized models. Please disable it by setting PT_HPU_WEIGHT_SHARING=0\033[0m"""
        )
    if model is not None:
        modified_model = fuse_conv_bn.fuse(model)
        return modified_model


def hpu_initialize(
    model=None, mark_only_scales_as_const=False, mark_scales=True, mark_non_scales=True, optimizer=None, args=None
):
    print(
        """
        [TO BE DEPRECATED] Please use hpu_inference_initialize instead
        """
    )
    hpu_inference_initialize(model, mark_only_scales_as_const, mark_scales, mark_non_scales, optimizer, args)


def hpu_inference_initialize(
    model=None, mark_only_scales_as_const=False, mark_scales=True, mark_non_scales=True, optimizer=None, args=None
):
    """
    Mark params of the model on HPU as const
    To be called after moving tensors/model to hpu
    To be called after model.to(hpu)
    Note: 'mark_only_scales_as_const' will be removed in future, please use 'mark_scales'
    """
    print(
        """WARNING: The argument 'mark_only_scales_as_const' will be removed soon. Please use 'mark_scales' instead.
        If mark_only_scales_as_const=True or mark_scales=True, then only scales are marked as const.
        If mark_non_scales=True, then non scale tensors are marked as constants.
        By default mark_only_scales_as_const=False, mark_scales=True, mark_non_scales=True
        """
    )
    if not hpu.is_inference_mode_enabled():
        hpu.enable_inference_mode()
    if not hpu.is_matmul3d_2d_reshape_enabled():
        hpu.enable_matmul3d_2d_reshape()

    mark_only_scales = mark_only_scales_as_const or mark_scales
    if mark_only_scales_as_const:
        mark_non_scales = False

    if model is not None:
        if getenv("PT_HPU_LAZY_MODE", "0") != "0":
            _mark_params_as_const(model=model, mark_scales=mark_only_scales, mark_non_scales=mark_non_scales)
            _check_params_as_const(model=model, mark_scales=mark_only_scales, mark_non_scales=mark_non_scales)
        else:
            hpu.set_mark_scale_const(mark_only_scales)
            hpu.set_mark_non_scale_const(mark_non_scales)
        _read_min_max_overwrite()
        _set_quantization_attributes(model)
        with _e_handler():
            _handle_quant_stats(model)


def hpu_reset_env():
    print(
        """
        [TO BE DEPRECATED] Please use hpu_inference_reset_env instead
        """
    )
    hpu_inference_reset_env()


def hpu_teardown_inference_env():
    print(
        """
        [TO BE DEPRECATED] Please use hpu_inference_reset_env instead
        """
    )
    hpu_inference_reset_env()


def hpu_inference_reset_env():
    """
    Disables inference mode
    To be called after model execution is done on HPU
    """
    hpu.disable_inference_mode()
    hpu.disable_matmul3d_2d_reshape()
