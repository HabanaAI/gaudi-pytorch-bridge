import threading
import torch
import habana_frameworks.torch._core_C as htcore
from collections import deque
from functools import wraps
from typing import Union

from torch.functional import Tensor

_DEVICE_CONTEXTS = dict()
_DEVICE_CONTEXTS_LOCK = threading.Lock()

class _DeviceContext(object):
    def __init__(self, device):
        self.device = device

def _get_device_context(device=None):
    if device is None:
        device = htcore._hb_get_default_device()

    with _DEVICE_CONTEXTS_LOCK:
        devctx = _DEVICE_CONTEXTS.get(device, None)
        if devctx is None:
            devctx = _DeviceContext(device)
            _DEVICE_CONTEXTS[device] = devctx
        return devctx

def add_step_closure(closure, args=()):
    devctx = _get_device_context()
    step_closures = getattr(devctx, "step_closures", None)
    if step_closures is None:
        step_closures = []
        devctx.step_closures = step_closures
    step_closures.append(lambda a=args: closure(*a))

def _run_step_closures():
    devctx = _get_device_context()
    step_closures = getattr(devctx, "step_closures", None)
    if step_closures is not None:
        devctx.step_closures = []
        for closure in step_closures:
            closure()

def mark_step(device_str=""):
    htcore._mark_step(device_str)
    _run_step_closures()

name_stack = deque()

def gen_pre_fwd_hook(name):
    def pre_fwd_hook(module, input):
        new_name = name_stack[-1] + "/" + name if name_stack else name
        name_stack.append(new_name)
        htcore.set_module_name(new_name)
    return pre_fwd_hook

def gen_grad_hook(name):
    def grad_hook(grad):
        htcore.set_module_name(name)
        return grad
    return grad_hook

def post_fwd_hook(module, input, output):
    module_name = name_stack.pop()
    if (name_stack):
        name = name_stack[-1]
    else:
        name = ""
    grad_name = "gradient/" + module_name
    htcore.set_module_name(name)
    try:
        if isinstance(output, Tensor):
            if output.requires_grad:
                output.register_hook(gen_grad_hook(grad_name))
        else:
            for o in output:
                if isinstance(o, Tensor) and o.requires_grad:
                    o.register_hook(gen_grad_hook(grad_name))
    except:
        pass

add_module_orig = torch.nn.modules.Module.add_module

@wraps(torch.nn.modules.Module.add_module)
def wrap_add_module(self, name, module):
    module.register_forward_pre_hook(gen_pre_fwd_hook(name))
    module.register_forward_hook(post_fwd_hook)
    add_module_orig(self, name, module)

torch.nn.modules.Module.add_module = wrap_add_module

module_set_attr_orig = torch.nn.Module.__setattr__
@wraps(torch.nn.Module.__setattr__)
def wrap_set_attr(self, name: str, value: Union[torch.Tensor, 'torch.nn.Module']) -> None:
    if isinstance(value, torch.nn.Module):
        value.register_forward_pre_hook(gen_pre_fwd_hook(name))
        value.register_forward_hook(post_fwd_hook)
    module_set_attr_orig(self, name, value)

torch.nn.Module.__setattr__ = wrap_set_attr
