import enum
import habana_frameworks.torch.hpu as hpu
from habana_frameworks.torch.utils._experimental_C import synDeviceType
from habana_frameworks.torch.utils import _experimental_C
import habana_frameworks.torch.hpu.memory as htmem

_model_params_initialized = False
_optim_state_initialized = False

def _data_ptr(t) -> int:
    if hpu.is_available():
        hpu.init()
        return _experimental_C.data_ptr(t)
    else:
        return 0

def _get_device_type() -> int:
    if hpu.is_available():
        hpu.init()
        return _experimental_C.get_device_type()
    else:
        return -1

def _compute_stream() -> int:
    if hpu.is_available():
        hpu.init()
        return _experimental_C.compute_stream()
    else:
        return 0

def _record_param(name, t_start, t_size, is_param=False, is_grad=False, is_optim_state=False):
    if hpu.is_available():
          hpu.init()
          _experimental_C.record_param(name, is_param, is_grad, is_optim_state, t_start, t_size)

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
                        _record_param(submodule_name + "/" + param_name,
                                      _data_ptr(param.data),
                                      param.data.numel() * param.data.element_size(),
                                      is_param=True)
                    except:
                        pass
                    if param.grad is not None:
                        try:
                            _record_param(submodule_name + "/" + param_name + ".grad",
                                          _data_ptr(param.grad),
                                          param.grad.numel() * param.grad.element_size(),
                                          is_grad=True)
                        except:
                            pass
                if optimizer is not None and not _is_optim_state_initialized():
                    try:
                        # TBD: Record other optimizer state dict also
                        mbuf = optimizer.state[param]['momentum_buffer']
                        _record_param(submodule_name + "/optim/" + param_name + "/momentum_buffer",
                                      _data_ptr(mbuf),
                                      mbuf.numel() * mbuf.element_size(),
                                      is_optim_state=True)
                    except:
                        print("Exception in _record_param for optimizer buffer ", param_name)
                        pass
                    _is_optim_recorded = True
            for buffer_name, buffer in submodule.named_buffers(recurse=False):
                try:
                    _record_param(submodule_name + "/buffer_" + buffer_name,
                                  _data_ptr(buffer),
                                  buffer.numel() * buffer.element_size(),
                                  is_param=True)
                except:
                    pass
        _model_params_initialized = True
    _optim_state_initialized = _is_optim_recorded
