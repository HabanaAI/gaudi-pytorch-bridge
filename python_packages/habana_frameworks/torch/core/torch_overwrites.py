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
from os import environ, path
from typing import Union, Optional

import habana_frameworks.torch.hpu as ht
import habana_frameworks.torch.hpu.random as rand_hpu
import habana_frameworks.torch.utils.debug as htdebug
from habana_frameworks.torch.utils.internal import is_lazy
import torch
from torch.distributed.constants import default_pg_timeout
from torch.functional import Tensor
import threading

_name_stack = deque()
_module_dict = dict()
_lock = threading.Lock()


def _is_inference():
    return environ.get('PT_HPU_INFERENCE_MODE')


def _names_hook_already_registered(module):
    if hasattr(module, 'names_hook') and module.names_hook == True:
        return True
    return False


def _pre_fwd_hook(module, input):
    # handle the naming mismatch issue with a temp fix, till we
    # find a way to get unique module names from the calibration tool
    with _lock:
        try:
            if _is_inference() and _name_stack and "relu" in module.custom_name:
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
        except:
            new_name = module.custom_name


def _gen_grad_hook(name):
    def grad_hook(grad):
        htdebug._set_module_name(name)
        return grad
    return grad_hook


def _post_fwd_hook(module, input, output):
    with _lock:
        module_name = _name_stack.pop()
        if _is_inference() and module_name in _module_dict.keys():
            del _module_dict[module_name]
        if (_name_stack) and len(_name_stack):
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


class HpuMarker:
    pass


def _deserialize_habana_wrapper(func, marker, *args):
    return func(*args)


def overwrite_torch_functions():
    # wrap torch.manual_seed

    manual_seed_orig = torch.manual_seed

    @wraps(torch.manual_seed)
    def wrap_manual_seed(seed):
        if not is_lazy():
            from habana_frameworks.torch.utils import _debug_eager_C
            _debug_eager_C.join_pending_pipeline_threads()
        rand_hpu.manual_seed(seed)
        return manual_seed_orig(seed)

    torch.manual_seed = wrap_manual_seed

    # wrap torch.Tensor.record_stream

    record_orig = torch.Tensor.record_stream

    @wraps(torch.Tensor.record_stream)
    def wrap_record_stream(self, stream):
        if isinstance(self, Tensor) and self.device.type == 'hpu':
            ht.record_stream(self, stream)
        else:
            record_orig(self, stream)

    torch.Tensor.record_stream = wrap_record_stream

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

    # Install NNModule Hooks that assign name to the module, what is used for
    # debugging purposes. These hooks shouldn't be used with torch.compile due
    # to potential performance regressions that may be introduced by including
    # them, reference: https://pytorch.org/docs/master/compile/nn-module.html.
    if is_lazy():
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

    #Ie9b966962fca23e5118047c3e7a47545d4c87f11 needs to be merged to resolve https://github.com/pytorch/pytorch/issues/107460
    #torch compile will have issues with log/metric attributes until then.
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

    # Install NNModule Hooks that assign name to the module, what is used for
    # debugging purposes. These hooks shouldn't be used with torch.compile due
    # to potential performance regressions that may be introduced by including
    # them, reference: https://pytorch.org/docs/master/compile/nn-module.html.
    if is_lazy():
        torch.nn.Module.__setattr__ = wrap_set_attr

    # wrap torch.distributed.irecv

    dummy_mode = int(environ.get('P2P_DUMMY_MODE_PHASE', "0"))
    if dummy_mode == 1 or dummy_mode == 2:
        irecv_orig = torch.distributed.irecv
        distributed_c10d = torch.distributed.distributed_c10d

        def irecv_aux(tensor):
            dummy_mode = int(environ.get('P2P_DUMMY_MODE_PHASE', "0"))
            if not hasattr(irecv_aux, "dummy_mode_seq"):
                irecv_aux.dummy_mode_seq = 0  # it doesn't exist yet, so initialize it

            dummy_folder_path = environ.get("P2P_DUMMY_MODE_PATH") if environ.get(
                "P2P_DUMMY_MODE_PATH") != None else "./"
            tensor_file = dummy_folder_path + str(distributed_c10d.get_rank()) + \
                "_" + str(irecv_aux.dummy_mode_seq) + ".pt"

            if dummy_mode == 1:
                print("Dummy Mode: " + tensor_file + " saved.")
                torch.save(tensor.to('cpu'), tensor_file)

            if dummy_mode == 2:
                if path.exists(tensor_file):
                    tensor = torch.load(tensor_file).to('hpu')
                    print("Dummy Mode: " + tensor_file + " loaded.")
                else:
                    irecv_aux.dummy_mode_seq = 0
                    tensor_file = dummy_folder_path + str(distributed_c10d.get_rank()) + \
                        "_" + str(irecv_aux.dummy_mode_seq) + ".pt"
                    if path.exists(tensor_file):
                        print("Dummy Mode: " + tensor_file + " loaded.")
                        tensor = torch.load(tensor_file).to('hpu')
                    else:
                        raise Exception("Attempting to run HPU Dummy Mode but needed file " +
                                        tensor_file + " does not exist!")

            irecv_aux.dummy_mode_seq += 1
            return tensor

        @wraps(torch.distributed.irecv)
        def wrap_irecv(tensor: torch.Tensor, src: Optional[int] = None, group: Optional[distributed_c10d.ProcessGroup] = None, tag: int = 0) -> distributed_c10d.Work:
            res = irecv_orig(tensor, src, group, tag)
            tensor.copy_(irecv_aux(tensor))
            return res

        torch.distributed.irecv = wrap_irecv

    #wrap torch.Tensor._reduce_ex_internal

    _original_reduce_ex_internal = torch.Tensor._reduce_ex_internal

    @wraps(torch.Tensor._reduce_ex_internal)
    def _reduce_ex_internal_habana(self, proto):
        func, args = _original_reduce_ex_internal(self, proto)
        if self.device.type.startswith("hpu"):
            return _deserialize_habana_wrapper, (func, HpuMarker(), *args)

        return func, args

    torch.Tensor._reduce_ex_internal = _reduce_ex_internal_habana
