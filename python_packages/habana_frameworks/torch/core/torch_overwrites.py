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

import datetime
from collections import deque
from functools import wraps
from os import environ
from typing import Union

import habana_frameworks.torch.hpu.random as rand_hpu
import habana_frameworks.torch.utils.debug as htdebug
import torch
from torch.distributed.constants import default_pg_timeout
from torch.functional import Tensor

_name_stack = deque()
_module_dict = dict()

def _is_inference():
    return environ.get('PT_HPU_INFERENCE_MODE')

def _names_hook_already_registered(module):
    if hasattr(module, 'names_hook') and module.names_hook == True:
        return True
    return False

def _pre_fwd_hook(module, input):
    #handle the naming mismatch issue with a temp fix, till we
    #find a way to get unique module names from the calibration tool
    if _is_inference() and _name_stack and "relu" in  module.custom_name:
        ns = str(_name_stack[-1])
        if ns in _module_dict.keys():
            _module_dict[ns] += 1
            new_name = _name_stack[-1] + "/" + module.custom_name + "." + str(_module_dict[ns])
        else:
            _module_dict[ns] = 0
            new_name = _name_stack[-1] + "/" + module.custom_name if _name_stack else module.custom_name
    else:
        new_name = _name_stack[-1] + "/" + module.custom_name if _name_stack else module.custom_name

    _name_stack.append(new_name)
    htdebug._set_module_name(new_name)

def _gen_grad_hook(name):
    def grad_hook(grad):
        htdebug._set_module_name(name)
        return grad
    return grad_hook

def _post_fwd_hook(module, input, output):
    module_name = _name_stack.pop()
    if _is_inference() and module_name in _module_dict.keys():
        del _module_dict[module_name]
    if (_name_stack):
        name = _name_stack[-1]
    else:
        name = ""
    grad_name = "gradient/" + module_name
    htdebug._set_module_name(name)
    try:
        if isinstance(output, Tensor):
            if output.requires_grad:
                output.register_hook(_gen_grad_hook(grad_name))
        else:
            for o in output:
                if isinstance(o, Tensor) and o.requires_grad:
                    o.register_hook(_gen_grad_hook(grad_name))
    except:
        pass

def overwrite_torch_functions():
    # wrap torch.manual_seed

    manual_seed_orig = torch.manual_seed

    @wraps(torch.manual_seed)
    def wrap_manual_seed(seed):
        rand_hpu.manual_seed(seed)
        return manual_seed_orig(seed)

    torch.manual_seed = wrap_manual_seed

    # wrap torch.nn.modules.Module.add_module

    add_module_orig = torch.nn.modules.Module.add_module

    @wraps(torch.nn.modules.Module.add_module)
    def wrap_add_module(self, name, module):
        if isinstance(module, torch.nn.Module) and not _names_hook_already_registered(module):
            try:
                module.custom_name = name
                module.register_forward_pre_hook(_pre_fwd_hook)
                module.register_forward_hook(_post_fwd_hook)
                module.names_hook = True
            except (RuntimeError):
                pass
        add_module_orig(self, name, module)

    torch.nn.modules.Module.add_module = wrap_add_module

    # wrap torch.distributed.new_group and torch.distributed.init_process_group

    ranks_cache = {}
    new_group_orig = torch.distributed.new_group
    init_process_group_orig = torch.distributed.init_process_group

    @wraps(torch.distributed.new_group)
    def wrap_new_group(ranks=None, timeout=default_pg_timeout, backend=None, pg_options=None):
        nonlocal ranks_cache
        cache_enable = environ.get('PT_ENABLE_COMM_GROUP_CACHE', "False")
        if cache_enable.lower() == "true":
            nonlocal ranks_cache
            if ranks == None:
                actual_world_size = torch.distributed.distributed_c10d.get_world_size()
                ranks_tuple = tuple(list(range(0, actual_world_size)))
            else:
                ranks_tuple = tuple(sorted(tuple(ranks)))
            if ranks_tuple in ranks_cache:
                return ranks_cache[ranks_tuple]
            else:
                ranks_cache[ranks_tuple] = new_group_orig(ranks, timeout, backend, pg_options)
                return ranks_cache[ranks_tuple]
        else:
            return new_group_orig(ranks, timeout, backend, pg_options)

    @wraps(torch.distributed.init_process_group)
    def wrap_init_process_group(backend, init_method=None, timeout=datetime.timedelta(seconds=1800), world_size=- 1, rank=- 1, store=None, group_name='', pg_options=None):
        nonlocal ranks_cache
        cache_enable = environ.get('PT_ENABLE_COMM_GROUP_CACHE', "False")
        if cache_enable.lower() == "true":
            if len(ranks_cache) == 0:
                init_process_group_orig(backend, init_method, timeout, world_size, rank, store, group_name, pg_options)
            actual_world_size = torch.distributed.distributed_c10d.get_world_size()
            ranks_tuple = tuple(list(range(0, actual_world_size)))
            if ranks_tuple in ranks_cache:
                return ranks_cache[ranks_tuple]
            ranks_cache[ranks_tuple] = torch.distributed.distributed_c10d._get_default_group()
            return ranks_cache[ranks_tuple]
        else:
            return init_process_group_orig(backend, init_method, timeout, world_size, rank, store, group_name, pg_options)


    torch.distributed.new_group = wrap_new_group
    torch.distributed.init_process_group = wrap_init_process_group

    # wrap torch.nn.Module.__setattr__

    module_set_attr_orig = torch.nn.Module.__setattr__

    @wraps(torch.nn.Module.__setattr__)
    def wrap_set_attr(self, name: str, value: Union[torch.Tensor, 'torch.nn.Module']) -> None:
        if isinstance(value, torch.nn.Module) and not _names_hook_already_registered(value):
            try:
                value.custom_name = name
                value.register_forward_pre_hook(_pre_fwd_hook)
                value.register_forward_hook(_post_fwd_hook)
                value.names_hook = True
            except (RuntimeError):
                pass
        module_set_attr_orig(self, name, value)

    torch.nn.Module.__setattr__ = wrap_set_attr
