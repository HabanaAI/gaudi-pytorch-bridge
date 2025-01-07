import enum
import sys
import warnings

import habana_frameworks.torch.hpu as hpu
import habana_frameworks.torch.hpu.memory as htmem
from habana_frameworks.torch.utils import _experimental_C
from habana_frameworks.torch.utils._experimental_C import synDeviceType
from habana_frameworks.torch.utils.experimental.detect_recompilation import (
    const_shape_dataloader,
    data_dynamicity,
    detect_recompilation_auto_model,
)

_model_params_initialized = False
_optim_state_initialized = False
_available = False


def _is_available() -> bool:
    # Checks hpu.is_available() and caches it locally
    global _available
    if not _available:
        _available = hpu.is_available()
        hpu.init()
    return _available


def _data_ptr(t) -> int:
    # Note: Ensure whether _data_ptr(t) is returning a valid pointer, this function
    # can return null pointer as well.
    # Eg: if the size of tensor is [0], this will return null
    if _is_available():
        try:
            return _experimental_C.data_ptr(t)
        except:
            return 0
    else:
        return 0


def _get_device_type() -> int:
    return _experimental_C.get_device_type()


def _is_fp16_supported() -> bool:
    return _get_device_type() != synDeviceType.synDeviceGaudi


def _compute_stream() -> int:
    if _is_available():
        try:
            return _experimental_C.compute_stream()
        except:
            return 0
    else:
        return 0


def _record_param(name, t_start, t_size, is_param=False, is_grad=False, is_optim_state=False):
    if _is_available():
        try:
            _experimental_C.record_param(name, is_param, is_grad, is_optim_state, t_start, t_size)
        except:
            pass


def _is_model_param_initialized() -> bool:
    return _model_params_initialized


def _is_optim_state_initialized() -> bool:
    return _optim_state_initialized


def _record_params(model=None, optimizer=None, force_model_update=False):
    _is_optim_recorded = False
    if model is not None:
        for submodule_name, submodule in model.named_modules():
            for param_name, param in submodule.named_parameters(recurse=False):
                if not _is_model_param_initialized():
                    try:
                        _record_param(
                            submodule_name + "/" + param_name,
                            _data_ptr(param.data),
                            param.data.numel() * param.data.element_size(),
                            is_param=True,
                        )
                    except:
                        pass
                    if param.grad is not None:
                        try:
                            _record_param(
                                submodule_name + "/" + param_name + ".grad",
                                _data_ptr(param.grad),
                                param.grad.numel() * param.grad.element_size(),
                                is_grad=True,
                            )
                        except:
                            pass
                if optimizer is not None and not _is_optim_state_initialized():
                    try:
                        # TBD: Record other optimizer state dict also
                        mbuf = optimizer.state[param]["momentum_buffer"]
                        _record_param(
                            submodule_name + "/optim/" + param_name + "/momentum_buffer",
                            _data_ptr(mbuf),
                            mbuf.numel() * mbuf.element_size(),
                            is_optim_state=True,
                        )
                    except:
                        print("Exception in _record_param for optimizer buffer ", param_name)
                        pass
                    _is_optim_recorded = True
            for buffer_name, buffer in submodule.named_buffers(recurse=False):
                try:
                    _record_param(
                        submodule_name + "/buffer_" + buffer_name,
                        _data_ptr(buffer),
                        buffer.numel() * buffer.element_size(),
                        is_param=True,
                    )
                except:
                    pass
        _model_params_initialized = True
    _optim_state_initialized = _is_optim_recorded


def _set_profiler_tracer_memory(device_id) -> int:
    if _is_available():
        try:
            return _experimental_C.set_profiler_tracer_memory(device_id)
        except Exception as e:
            print("Call for set_profiler_tracer_memory failed with: ", e, file=sys.stderr)
            return 0
    else:
        return 0


def _reset_device_memory():
    warnings.warn("used only to reset the memory in case of testing")
    if _is_available:
        hpu.init()
        _experimental_C.reset_device_memory()
